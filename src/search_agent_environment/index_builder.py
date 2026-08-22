import os
import re
import faiss
import json
import warnings
import numpy as np
from typing import cast, List, Dict
import shutil
import subprocess
import argparse
import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# from LongRAG.retriever.utils import load_model, load_corpus, pooling
import datasets
import requests
from transformers import AutoTokenizer, AutoModel, AutoConfig

from retrieval_server import _apply_encoder_prefix


def load_model(
        model_path: str, 
        use_fp16: bool = False,
        device: int = 0
    ):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to(device)
    if use_fp16: 
        model = model.to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if "qwen3" in model_path.lower():
        # Qwen3-Embedding uses last-token pooling, which requires left padding
        tokenizer.padding_side = "left"

    return model, tokenizer


def pooling(
        pooler_output,
        last_hidden_state,
        attention_mask = None,
        pooling_method = "mean"
    ):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    elif pooling_method == "last_token":
        # attention_mask[:, -1] is 1 for every sample iff the batch is
        # left-padded (real tokens right-aligned), in which case the last
        # real token is always at the last position regardless of each
        # sample's actual length. For right-padded batches, the last real
        # token is at position (num_real_tokens - 1) from the start.
        # Mixing these up silently returns a padding-position hidden state
        # (garbage) for every sample shorter than the batch's max length.
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_state[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        return last_hidden_state[torch.arange(batch_size), sequence_lengths]
    else:
        raise NotImplementedError("Pooling method not implemented!")


def load_corpus(corpus_path: str):
    if corpus_path.endswith(".parquet"):
        corpus = datasets.load_dataset(
                'parquet',
                data_files=corpus_path,
                split="train",
                num_proc=4)
        if "retrieval_text" not in corpus.column_names:
            corpus = corpus.map(
                lambda x: {"retrieval_text": x["text"]},
                num_proc=4,
            )
    else:
        corpus = datasets.load_dataset(
                'json', 
                data_files=corpus_path,
                split="train",
                num_proc=4)
    return corpus


def encode_batch_via_sglang(sglang_url, model_name, batch_data, timeout=300):
    """Encode a batch of passages via a remote sglang embedding server
    (launched with --is-embedding), instead of running the model locally.
    Applies the same passage-side instruction-prefix logic as the online
    serving path (retrieval_server.py's SglangEncoder/Encoder), so index
    building and online querying stay consistent, and normalizes
    client-side to match Encoder's behavior.
    """
    texts = _apply_encoder_prefix(model_name, batch_data, is_query=False)
    response = requests.post(
        f"{sglang_url.rstrip('/')}/v1/embeddings",
        json={"model": model_name, "input": texts, "encoding_format": "float"},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()["data"]
    embeddings = np.array([item["embedding"] for item in data], dtype=np.float32)

    if "dpr" not in model_name.lower():
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)

    return embeddings.astype(np.float32, order="C")


@torch.no_grad()
def _encode_worker(rank, world_size, retrieval_method, model_path, use_fp16,
                    pooling_method, max_length, batch_size, corpus_path, result_queue):
    """Runs in its own process, pinned to GPU `rank`. Encodes its shard of the
    corpus and pushes the resulting embeddings back through `result_queue`.
    """
    torch.cuda.set_device(rank)
    encoder, tokenizer = load_model(model_path=model_path, use_fp16=use_fp16, device=rank)
    corpus = load_corpus(corpus_path)
    corpus_shard = corpus.shard(num_shards=world_size, index=rank, contiguous=True)

    shard_embeddings = []
    for start_idx in tqdm(range(0, len(corpus_shard), batch_size), desc=f'[GPU {rank}] Inference Embeddings:'):
        batch_data = corpus_shard[start_idx:start_idx + batch_size]['retrieval_text']
        
        if retrieval_method == "e5":
            batch_data = [f"passage: {doc}" for doc in batch_data]
        # qwen3-embedding does not require a passage-side prefix/instruction

        inputs = tokenizer(
                    batch_data,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=max_length,
        ).to(rank)

        if "T5" in type(encoder).__name__:
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = encoder(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            embeddings = output.last_hidden_state[:, 0, :]
        else:
            output = encoder(**inputs, return_dict=True)
            embeddings = pooling(getattr(output, 'pooler_output', None),
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                pooling_method)
            if "dpr" not in retrieval_method:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        embeddings = cast(torch.Tensor, embeddings)
        shard_embeddings.append(embeddings.detach().cpu().float().numpy())

    shard_embeddings = np.concatenate(shard_embeddings, axis=0).astype(np.float32)
    result_queue.put((rank, shard_embeddings))


class Index_Builder:
    r"""A tool class used to build an index used in retrieval.
    
    """
    def __init__(
            self, 
            retrieval_method,
            model_path,
            corpus_path,
            save_dir,
            max_length,
            batch_size,
            use_fp16,
            pooling_method,
            faiss_type=None,
            embedding_path=None,
            save_embedding=False,
            faiss_gpu=False,
            train_sample_size=None,
            encode_backend="local",
            sglang_embedding_url=None,
            sglang_concurrency=8,
        ):
        
        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.pooling_method = pooling_method
        self.faiss_type = faiss_type if faiss_type is not None else 'Flat'
        self.train_sample_size = train_sample_size
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu
        self.encode_backend = encode_backend
        self.sglang_embedding_url = sglang_embedding_url
        self.sglang_concurrency = sglang_concurrency
        if self.encode_backend == "sglang" and not self.sglang_embedding_url:
            raise ValueError("--sglang_embedding_url is required when --encode_backend sglang.")

        self.gpu_num = torch.cuda.device_count()
        # prepare save dir
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)

        self.index_save_path = os.path.join(self.save_dir, f"{self.retrieval_method}_{self.faiss_type}.index")

        self.embedding_save_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.memmap")

        self.corpus = load_corpus(self.corpus_path)
       
        print("Finish loading...")
    @staticmethod
    def _check_dir(dir_path):
        r"""Check if the dir path exists and if there is content.
        
        """
        
        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def build_index(self):
        r"""Constructing different indexes based on selective retrieval method.

        """
        if self.retrieval_method == "bm25":
            self.build_bm25_index()
        else:
            self.build_dense_index()

    def build_bm25_index(self):
        """Building BM25 index using the pure-Python `bm25s` library.

        Reference: https://github.com/xhluca/bm25s
        """
        import bm25s

        self.save_dir = os.path.join(self.save_dir, "bm25")
        os.makedirs(self.save_dir, exist_ok=True)

        print("Start building bm25 index...")
        corpus_texts = [item["retrieval_text"] for item in self.corpus]
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save(self.save_dir, corpus=self.corpus)

        print("Finish!")

    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        all_embeddings = np.memmap(
                embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(corpus_size, hidden_size)
        return all_embeddings

    def _save_embedding(self, all_embeddings):
        memmap = np.memmap(
            self.embedding_save_path,
            shape=all_embeddings.shape,
            mode="w+",
            dtype=all_embeddings.dtype
        )
        length = all_embeddings.shape[0]
        # add in batch
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                j = min(i + save_batch_size, length)
                memmap[i: j] = all_embeddings[i: j]
        else:
            memmap[:] = all_embeddings

    def encode_all(self):
        if self.encode_backend == "sglang":
            all_embeddings = self._encode_all_sglang()
        elif self.gpu_num > 1:
            print(f"Use {self.gpu_num} GPUs in parallel (one process per GPU)!")
            ctx = mp.get_context("spawn")
            result_queue = ctx.Queue()
            processes = []
            for rank in range(self.gpu_num):
                p = ctx.Process(
                    target=_encode_worker,
                    args=(rank, self.gpu_num, self.retrieval_method, self.model_path,
                          self.use_fp16, self.pooling_method, self.max_length,
                          self.batch_size, self.corpus_path, result_queue)
                )
                p.start()
                processes.append(p)

            shard_results = [None] * self.gpu_num
            for _ in range(self.gpu_num):
                rank, shard_embeddings = result_queue.get()
                shard_results[rank] = shard_embeddings
            for p in processes:
                p.join()

            all_embeddings = np.concatenate(shard_results, axis=0)
        else:
            self.encoder, self.tokenizer = load_model(model_path=self.model_path,
                                                      use_fp16=self.use_fp16)
            all_embeddings = self._encode_all_single_gpu()

        all_embeddings = all_embeddings.astype(np.float32)

        return all_embeddings

    def _encode_all_sglang(self):
        """Encode the whole corpus by sending batches to a remote sglang
        embedding server, instead of loading the model locally. This keeps
        index building on the exact same encoding path (server-side
        pooling/normalization) as online querying when the serving side
        also uses SglangEncoder, avoiding index/query encoding mismatches.

        Requests are dispatched with up to `sglang_concurrency` in flight
        at once (instead of one blocking request at a time), so sglang's
        own continuous-batching scheduler has multiple batches to work on
        concurrently and the GPU doesn't idle between the client receiving
        one response and sending the next request.
        """
        starts = list(range(0, len(self.corpus), self.batch_size))

        def _encode_chunk(start_idx):
            batch_data = self.corpus[start_idx:start_idx + self.batch_size]['retrieval_text']
            return encode_batch_via_sglang(self.sglang_embedding_url, self.model_path, batch_data)

        all_embeddings = [None] * len(starts)
        with ThreadPoolExecutor(max_workers=self.sglang_concurrency) as executor:
            futures = {executor.submit(_encode_chunk, start_idx): i for i, start_idx in enumerate(starts)}
            for future in tqdm(as_completed(futures), total=len(starts), desc='Inference Embeddings (sglang):'):
                i = futures[future]
                all_embeddings[i] = future.result()

        all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        return all_embeddings

    def _encode_all_single_gpu(self):
        all_embeddings = []

        for start_idx in tqdm(range(0, len(self.corpus), self.batch_size), desc='Inference Embeddings:'):

            # batch_data_title = self.corpus[start_idx:start_idx+self.batch_size]['title']
            # batch_data_text = self.corpus[start_idx:start_idx+self.batch_size]['text']
            # batch_data = ['"' + title + '"\n' + text for title, text in zip(batch_data_title, batch_data_text)]
            batch_data = self.corpus[start_idx:start_idx+self.batch_size]['retrieval_text']

            if self.retrieval_method == "e5":
                batch_data = [f"passage: {doc}" for doc in batch_data]
            # qwen3-embedding does not require a passage-side prefix/instruction

            inputs = self.tokenizer(
                        batch_data,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.max_length,
            ).to('cuda')

            inputs = {k: v.cuda() for k, v in inputs.items()}

            #TODO: support encoder-only T5 model
            if "T5" in type(self.encoder).__name__:
                # T5-based retrieval model
                decoder_input_ids = torch.zeros(
                    (inputs['input_ids'].shape[0], 1), dtype=torch.long
                ).to(inputs['input_ids'].device)
                output = self.encoder(
                    **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
                )
                embeddings = output.last_hidden_state[:, 0, :]

            else:
                output = self.encoder(**inputs, return_dict=True)
                embeddings = pooling(getattr(output, 'pooler_output', None), 
                                    output.last_hidden_state, 
                                    inputs['attention_mask'],
                                    self.pooling_method)
                if  "dpr" not in self.retrieval_method:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            embeddings = cast(torch.Tensor, embeddings)
            embeddings = embeddings.detach().cpu().float().numpy()
            all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings.astype(np.float32)

        return all_embeddings

    @staticmethod
    def _get_nlist(faiss_type: str) -> int:
        """Extract the IVF nlist value from a faiss index_factory string
        such as 'IVF131072,SQ8'. Returns 0 if the index has no IVF component
        (e.g. a plain 'Flat' index, which doesn't require training a coarse
        quantizer on a subsample).
        """
        match = re.search(r'IVF(\d+)', faiss_type)
        return int(match.group(1)) if match else 0

    def _get_train_vectors(self, all_embeddings):
        """Select the vectors used to train the coarse quantizer.

        Training a GPU IVF index on the full corpus can blow up GPU memory
        (faiss replicates/tiles the training set per shard during k-means),
        so by default we train on a random subsample capped well below full
        corpus size, following the common faiss practice of training on a
        representative subset rather than every vector.
        """
        n = all_embeddings.shape[0]
        if self.train_sample_size is not None:
            train_size = min(self.train_sample_size, n)
        else:
            nlist = self._get_nlist(self.faiss_type)
            if nlist == 0:
                train_size = n
            else:
                # faiss requires >= ~39 points/centroid (warns below that);
                # target 40/centroid, with the cap always kept above that
                # minimum so it never re-introduces the warning.
                min_required = nlist * 39
                target = nlist * 40
                train_size = min(n, max(target, min_required, 1_000_000))

        if train_size >= n:
            return all_embeddings

        rng = np.random.default_rng(0)
        train_idxs = rng.choice(n, size=train_size, replace=False)
        return all_embeddings[train_idxs]

    @torch.no_grad()
    def build_dense_index(self):
        """Obtain the representation of documents based on the embedding model(BERT-based) and 
        construct a faiss index.
        """
        
        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")
        
        if self.embedding_path is not None:
            self.encoder, self.tokenizer = load_model(model_path = self.model_path, 
                                                      use_fp16 = self.use_fp16)
            hidden_size = self.encoder.config.hidden_size
            corpus_size = len(self.corpus)
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
        else:
            all_embeddings = self.encode_all()
            if self.save_embedding:
                self._save_embedding(all_embeddings)
            del self.corpus

        # build index
        print("Creating index")
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)
        
        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                train_vectors = self._get_train_vectors(all_embeddings)
                print(f"Training on {train_vectors.shape[0]} / {all_embeddings.shape[0]} vectors")
                faiss_index.train(train_vectors)
            faiss_index.add(all_embeddings)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                train_vectors = self._get_train_vectors(all_embeddings)
                print(f"Training on {train_vectors.shape[0]} / {all_embeddings.shape[0]} vectors")
                faiss_index.train(train_vectors)
            faiss_index.add(all_embeddings)

        faiss.write_index(faiss_index, self.index_save_path)
        print("Finish!")


MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean",
    'jina': 'mean',
    'qwen3': 'last_token'
}


def main():
    parser = argparse.ArgumentParser(description = "Creating index.")

    # Basic parameters
    parser.add_argument('--retrieval_method', type=str)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--corpus_path', type=str)
    parser.add_argument('--save_dir', default= 'indexes/',type=str)

    # Parameters for building dense index
    parser.add_argument('--max_length', type=int, default=180)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--use_fp16', default=False, action='store_true')
    parser.add_argument('--pooling_method', type=str, default=None)
    parser.add_argument('--faiss_type',default=None,type=str)
    parser.add_argument('--embedding_path', default=None, type=str)
    parser.add_argument('--save_embedding', action='store_true', default=False)
    parser.add_argument('--faiss_gpu', default=False, action='store_true')
    parser.add_argument('--train_sample_size', type=int, default=None,
                         help='Number of vectors to subsample for IVF training (default: auto, ~40*nlist capped at 1M-2M). Set to a large value or corpus size to train on the full corpus.')
    parser.add_argument('--encode_backend', type=str, default='local', choices=['local', 'sglang'],
                         help="Use a local model (default) or a remote sglang embedding server for passage encoding.")
    parser.add_argument('--sglang_embedding_url', type=str, default=None,
                         help="Base URL of the sglang embedding server (launched with --is-embedding), required when --encode_backend sglang.")
    parser.add_argument('--sglang_concurrency', type=int, default=8,
                         help="Number of concurrent in-flight requests to the sglang embedding server (higher keeps its batching scheduler busier / improves GPU utilization).")
    
    args = parser.parse_args()

    if args.pooling_method is None:
        pooling_method = 'mean'
        for k,v in MODEL2POOLING.items():
            if k in args.retrieval_method.lower():
                pooling_method = v
                break
    else:
        if args.pooling_method not in ['mean','cls','pooler','last_token']:
            raise NotImplementedError
        else:
            pooling_method = args.pooling_method


    index_builder = Index_Builder(
                        retrieval_method = args.retrieval_method,
                        model_path = args.model_path,
                        corpus_path = args.corpus_path,
                        save_dir = args.save_dir,
                        max_length = args.max_length,
                        batch_size = args.batch_size,
                        use_fp16 = args.use_fp16,
                        pooling_method = pooling_method,
                        faiss_type = args.faiss_type,
                        embedding_path = args.embedding_path,
                        save_embedding = args.save_embedding,
                        faiss_gpu = args.faiss_gpu,
                        train_sample_size = args.train_sample_size,
                        encode_backend = args.encode_backend,
                        sglang_embedding_url = args.sglang_embedding_url,
                        sglang_concurrency = args.sglang_concurrency,
                    )
    index_builder.build_index()


if __name__ == "__main__":
    main()
