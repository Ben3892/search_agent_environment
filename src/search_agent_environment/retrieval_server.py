import json
import os
import warnings
import hashlib
import copy
import threading
import queue
import time
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
import pandas as pd
import requests
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

def load_corpus(corpus_path: str):
    # Eagerly load the whole corpus into a pandas DataFrame instead of a
    # HuggingFace datasets.Dataset (Arrow, mmap-backed). Callers already
    # assume the corpus fits fully in memory (DocumentIndex, load_docs), so
    # Arrow's lazy/mmap semantics buy nothing here -- they only add
    # per-element Python object conversion overhead when materializing
    # columns (corpus["col"]) or rows (corpus[idx]) at query/build time.
    # pandas' bulk pyarrow-backed parquet reader converts columns in one
    # C++ pass, which is significantly faster for tens of millions of rows.
    if corpus_path.endswith(".parquet"):
        return pd.read_parquet(corpus_path, engine='pyarrow', use_threads=True)
    return pd.read_json(corpus_path, lines=True)

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs):
    results = [corpus.iloc[int(idx)].to_dict() for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
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

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        if "qwen3" in self.model_name.lower():
            # Qwen3-Embedding: only the query side needs an instruction prefix
            if is_query:
                instruct = "Given a web search query, retrieve relevant passages that answer the query"
                query_list = [f"Instruct: {instruct}\nQuery: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(getattr(output, 'pooler_output', None),
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().float().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb


def _apply_encoder_prefix(model_name: str, query_list: List[str], is_query: bool) -> List[str]:
    """Shared query/passage instruction-prefix logic, used by both the
    local Encoder and the remote SglangEncoder so the two backends produce
    identical inputs to the underlying model.
    """
    if "e5" in model_name.lower():
        if is_query:
            return [f"query: {query}" for query in query_list]
        else:
            return [f"passage: {query}" for query in query_list]

    if "bge" in model_name.lower():
        if is_query:
            return [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

    if "qwen3" in model_name.lower():
        # Qwen3-Embedding: only the query side needs an instruction prefix
        if is_query:
            instruct = "Given a web search query, retrieve relevant passages that answer the query"
            return [f"Instruct: {instruct}\nQuery: {query}" for query in query_list]

    return query_list


class SglangEncoder:
    """Encodes queries/passages via a remote sglang embedding server
    (launched with --is-embedding) instead of loading the model locally.
    Applies the same query/passage instruction prefixes as `Encoder` so the
    two backends are interchangeable.
    """

    def __init__(self, model_name, sglang_url, timeout=60):
        self.model_name = model_name
        self.sglang_url = sglang_url.rstrip("/")
        self.timeout = timeout

    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        texts = _apply_encoder_prefix(self.model_name, query_list, is_query)

        response = requests.post(
            f"{self.sglang_url}/v1/embeddings",
            json={"model": self.model_name, "input": texts, "encoding_format": "float"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()["data"]
        embeddings = np.array([item["embedding"] for item in data], dtype=np.float32)

        if "dpr" not in self.model_name.lower():
            # Match Encoder's client-side L2 normalization (needed for the
            # faiss inner-product index to behave like cosine similarity).
            # Note: sglang's Pooler already applies normalize=True
            # server-side for at least some model architectures, so this is
            # likely redundant (idempotent) rather than harmful.
            norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)

        return embeddings.astype(np.float32, order="C")


class QueryCache:
    """Simple query -> (results, scores) cache, keyed by (retrieval_method, topk, query).
    Bounded by max_size entries (LRU eviction). Persisted to a json file if
    cache_path is provided.
    """

    def __init__(self, cache_path: str = None, max_size: int = 10000):
        self.cache_path = cache_path
        self.max_size = max_size
        self.cache: "OrderedDict[str, Dict]" = OrderedDict()
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                loaded = json.load(f)
                for k, v in loaded.items():
                    self.cache[k] = v
                self._enforce_max_size()

    def _make_key(self, query: str, num: int, retrieval_method: str) -> str:
        raw = f"{retrieval_method}||{num}||{query}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _enforce_max_size(self):
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # evict least-recently-used entry

    def get(self, query: str, num: int, retrieval_method: str):
        key = self._make_key(query, num, retrieval_method)
        entry = self.cache.get(key)
        if entry is None:
            return None
        self.cache.move_to_end(key)  # mark as recently used
        return entry["results"], entry["scores"]

    def set(self, query: str, num: int, retrieval_method: str, results, scores):
        key = self._make_key(query, num, retrieval_method)
        self.cache[key] = {"results": results, "scores": scores}
        self.cache.move_to_end(key)
        self._enforce_max_size()

    def save(self):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(dict(self.cache), f)


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

        cache_path = getattr(config, "cache_save_path", None)
        self.disable_cache = getattr(config, "disable_cache", False)
        self.cache = QueryCache(cache_path)

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        if not self.disable_cache:
            cached = self.cache.get(query, num, self.retrieval_method)
            if cached is not None:
                results, scores = cached
                return (results, scores) if return_score else results

        results, scores = self._search(query, num, True)
        if not self.disable_cache:
            self.cache.set(query, num, self.retrieval_method, results, scores)
        return (results, scores) if return_score else results

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk

        if self.disable_cache:
            new_results, new_scores = self._batch_search(query_list, num, True)
            return (new_results, new_scores) if return_score else new_results

        cached_results: List = [None] * len(query_list)
        cached_scores: List = [None] * len(query_list)
        miss_idxs, miss_queries = [], []
        for i, query in enumerate(query_list):
            cached = self.cache.get(query, num, self.retrieval_method)
            if cached is not None:
                cached_results[i], cached_scores[i] = cached
            else:
                miss_idxs.append(i)
                miss_queries.append(query)

        if miss_queries:
            new_results, new_scores = self._batch_search(miss_queries, num, True)
            for idx, query, res, sc in zip(miss_idxs, miss_queries, new_results, new_scores):
                cached_results[idx] = res
                cached_scores[idx] = sc
                self.cache.set(query, num, self.retrieval_method, res, sc)

        return (cached_results, cached_scores) if return_score else cached_results

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        import bm25s
        self.retriever = bm25s.BM25.load(self.index_path, load_corpus=True, mmap=True)
        self.max_process_num = 8

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        import bm25s
        query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        k = min(num, len(self.retriever.corpus))
        docs, scores = self.retriever.retrieve(query_tokens, k=k, show_progress=False, n_threads=6)
        docs = docs[0]
        scores = scores[0]
        if len(docs) < num:
            warnings.warn('Not enough documents retrieved!')

        results = [dict(doc) for doc in docs]
        scores = scores.tolist()

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if num is None:
            num = 50
        import bm25s
        # Batch tokenize + retrieve in one call instead of looping per query:
        # bm25s is designed to vectorize scoring across multiple queries at
        # once (sparse matrix ops), so hundreds of sequential per-query
        # Python-level calls (each with its own tokenize + retrieve
        # overhead) become the actual bottleneck for large query batches,
        # while the GPU-based dense side sits idle waiting for this to
        # finish.
        query_tokens = bm25s.tokenize(query_list, stopwords="en", show_progress=False)
        k = min(num, len(self.retriever.corpus))
        docs_batch, scores_batch = self.retriever.retrieve(query_tokens, k=k, show_progress=False, n_threads=64)

        results = []
        scores = []
        for docs, score_row in zip(docs_batch, scores_batch):
            item_results = [dict(doc) for doc in docs]
            if len(item_results) < num:
                warnings.warn('Not enough documents retrieved!')
            results.append(item_results)
            scores.append(score_row.tolist())

        if return_score:
            return results, scores
        else:
            return results

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)

        # IVF indices default to nprobe=1 (search only the single nearest
        # cluster), which silently caps recall regardless of how well the
        # coarse quantizer was trained: with e.g. IVF65536, nprobe=1 means
        # only ~1/65536 of the corpus is even considered per query. Set
        # nprobe on the CPU IVF struct *before* GPU cloning so the clone
        # inherits it; setting nprobe on a GPU-wrapped index after cloning
        # is unreliable across faiss builds. Flat/non-IVF indices raise
        # from extract_index_ivf, which we treat as a no-op.
        try:
            cpu_ivf = faiss.extract_index_ivf(self.index)
            cpu_ivf.nprobe = config.faiss_nprobe
            self._is_ivf = True
            self._ivf_nlist = cpu_ivf.nlist
            logger.info(
                "Set CPU faiss IVF nprobe=%d (nlist=%d) before GPU clone",
                config.faiss_nprobe, cpu_ivf.nlist,
            )
        except Exception as e:
            self._is_ivf = False
            self._ivf_nlist = None
            logger.info(
                "faiss index is not IVF (extract_index_ivf failed: %s); "
                "skipping nprobe (requested=%d)", e, config.faiss_nprobe,
            )

        # Faiss GPU indices are not thread-safe, even for read-only search()
        # calls: StandardGpuResources manages a single region of GPU temp
        # memory that only one caller can use at a time (see faiss wiki,
        # "Threads and asynchronous calls"). A naive fix is a lock around
        # each request's own search() call, but that fully serializes
        # concurrent requests AND keeps each individual search() call tiny
        # (often a single query's embedding), which leaves the GPU mostly
        # idle between calls (<10% utilization observed) while requests
        # queue up (latency growing ~linearly with queue position under
        # concurrent load).
        #
        # Instead, coalesce concurrently-arriving search requests into one
        # larger batched search() call, issued by a single dedicated
        # thread -- this is the approach the faiss wiki itself recommends
        # ("multiple incoming searches from potentially different user
        # threads should be enqueued and aggregated/batched by the user
        # before handing to Faiss"). This satisfies faiss's one-caller-at-
        # a-time constraint while keeping the GPU busy with larger batches.
        self._search_queue: "queue.Queue" = queue.Queue() if config.faiss_gpu else None
        self._batch_window_s = 0.02
        self._batch_max_size = 256
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            self._batch_thread = threading.Thread(target=self._batch_search_worker, daemon=True)
            self._batch_thread.start()

        # Belt-and-suspenders: also set nprobe on whatever self.index is
        # now (which after `index_cpu_to_all_gpus(co.shard=True)` is an
        # IndexShards wrapping GPU IVF sub-indices). The Python .nprobe
        # attribute is not exposed on that wrapper on our faiss build,
        # ParameterSpace().set_index_parameter also fails on our
        # GpuIndexIVFScalarQuantizer sub-indices, and direct .nprobe
        # assignment on the shards wrapper is also unsupported. Recurse
        # into each shard and call the GPU C++ setNumProbes(N) directly
        # (or fall back to setting .nprobe on the shard).
        if self._is_ivf:
            applied_via = None
            per_shard_set = 0

            def _set_nprobe_on(idx_obj, n):
                # SWIG typically hands back IndexShards.at(i) as the
                # generic `faiss.Index` base class, so its GPU-subclass
                # methods (setNumProbes) and properties (.nprobe) are not
                # visible from Python. downcast_index() converts the
                # generic pointer to the concrete C++ subclass so those
                # become accessible. Falls back to the raw object if
                # downcast fails or is unnecessary (already downcast).
                try:
                    idx_obj = faiss.downcast_index(idx_obj)
                except Exception:
                    pass
                if hasattr(idx_obj, "setNumProbes"):
                    idx_obj.setNumProbes(n)
                    return f"setNumProbes ({type(idx_obj).__name__})"
                idx_obj.nprobe = n
                return f"attribute ({type(idx_obj).__name__})"

            def _read_nprobe_from(idx_obj):
                try:
                    idx_obj = faiss.downcast_index(idx_obj)
                except Exception:
                    pass
                if hasattr(idx_obj, "getNumProbes"):
                    try:
                        return idx_obj.getNumProbes()
                    except Exception:
                        pass
                return getattr(idx_obj, "nprobe", None)

            # First try the standard cross-type API on the top-level
            # wrapper; on our build this fails on GPU IVF sub-indices, in
            # which case we fall through to per-shard setters below.
            try:
                faiss.ParameterSpace().set_index_parameter(self.index, "nprobe", config.faiss_nprobe)
                applied_via = "ParameterSpace"
            except Exception as e_ps:
                logger.warning("ParameterSpace().set_index_parameter failed: %s", e_ps)

            # Per-shard fallback for IndexShards / IndexReplicas.
            if applied_via is None and hasattr(self.index, "at") and hasattr(self.index, "count"):
                for i in range(self.index.count()):
                    try:
                        via = _set_nprobe_on(self.index.at(i), config.faiss_nprobe)
                        per_shard_set += 1
                        applied_via = f"per-shard {via}"
                    except Exception as e_shard:
                        logger.warning("Setting nprobe on shard %d failed: %s", i, e_shard)

            # Last-resort direct attribute on the wrapper.
            if applied_via is None:
                try:
                    self.index.nprobe = config.faiss_nprobe
                    applied_via = "direct attribute"
                except Exception as e_attr:
                    logger.warning("Direct .nprobe assignment on wrapper failed: %s", e_attr)

            # Read back to verify.
            effective_nprobe = None
            try:
                if hasattr(self.index, "at") and hasattr(self.index, "count"):
                    effective_nprobe = _read_nprobe_from(self.index.at(0))
                else:
                    effective_nprobe = _read_nprobe_from(self.index)
            except Exception as e_read:
                logger.warning("Read-back of nprobe failed: %s", e_read)

            logger.info(
                "Set faiss IVF nprobe via %s (per_shard_set=%d): effective=%s (requested=%d, nlist=%d)",
                applied_via, per_shard_set, effective_nprobe, config.faiss_nprobe, self._ivf_nlist,
            )

        self.corpus = load_corpus(self.corpus_path)
        sglang_url = getattr(config, "sglang_embedding_url", None)
        if getattr(config, "retrieval_backend", "local") == "sglang":
            self.encoder = SglangEncoder(
                model_name=self.retrieval_method,
                sglang_url=sglang_url,
            )
        else:
            self.encoder = Encoder(
                model_name = self.retrieval_method,
                model_path = config.retrieval_model_path,
                pooling_method = config.retrieval_pooling_method,
                max_length = config.retrieval_query_max_length,
                use_fp16 = config.retrieval_use_fp16
            )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _batch_search_worker(self):
        """Runs in a single dedicated thread (only started when faiss_gpu is
        True) so it is the sole caller of self.index.search(). Coalesces
        whatever requests are waiting in self._search_queue into one larger
        search() call, up to _batch_window_s / _batch_max_size, instead of
        handling them one at a time.
        """
        while True:
            emb, k, fut = self._search_queue.get()
            batch = [(emb, k, fut)]
            deadline = time.perf_counter() + self._batch_window_s
            while len(batch) < self._batch_max_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._search_queue.get(timeout=remaining))
                except queue.Empty:
                    break

            try:
                embs = np.concatenate([b[0] for b in batch], axis=0)
                max_k = max(b[1] for b in batch)
                scores, idxs = self.index.search(embs, k=max_k)
                offset = 0
                for emb_i, k_i, fut_i in batch:
                    n = emb_i.shape[0]
                    # Truncating a search done with max_k down to a
                    # smaller k_i is equivalent to searching with k_i
                    # directly -- the top-k ranking doesn't depend on k.
                    fut_i.set_result((scores[offset:offset + n, :k_i], idxs[offset:offset + n, :k_i]))
                    offset += n
            except Exception as e:
                for _, _, fut_i in batch:
                    if not fut_i.done():
                        fut_i.set_exception(e)

    def _faiss_search(self, query_emb, k):
        if self._search_queue is None:
            return self.index.search(query_emb, k=k)
        fut = Future()
        self._search_queue.put((query_emb, k, fut))
        return fut.result()

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self._faiss_search(query_emb, num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self._faiss_search(batch_emb, num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results

class HybridRetriever(BaseRetriever):
    """Runs sparse (BM25) and dense retrieval in parallel, merges the two
    candidate lists, de-duplicates by document id (falling back to content
    hash if no 'id' field is present), and returns the merged candidates.
    Intended to be followed by a reranker to produce the final topk.
    """
    def __init__(self, config):
        super().__init__(config)

        bm25_config = copy.copy(config)
        bm25_config.retrieval_method = "bm25"
        bm25_config.index_path = config.bm25_index_path
        self.bm25_retriever = BM25Retriever(bm25_config)

        self.dense_retriever = DenseRetriever(config)

        self.bm25_topk = config.hybrid_bm25_topk
        self.dense_topk = config.hybrid_dense_topk
        self._executor = ThreadPoolExecutor(max_workers=2)

    @staticmethod
    def _dedup_key(doc_item):
        if "id" in doc_item:
            return doc_item["id"]
        content = doc_item.get("retrieval_text") or doc_item.get("text") or str(doc_item)
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _merge(self, bm25_results, bm25_scores, dense_results, dense_scores):
        merged = {}
        for doc, score in zip(bm25_results, bm25_scores):
            key = self._dedup_key(doc)
            if key not in merged:
                merged[key] = (doc, score)
        for doc, score in zip(dense_results, dense_scores):
            key = self._dedup_key(doc)
            if key not in merged:
                merged[key] = (doc, score)
        return list(merged.values())

    def _search(self, query: str, num: int = None, return_score: bool = False,
                bm25_num: int = None, dense_num: int = None):
        # Per-call bm25_num/dense_num take precedence; `num` (if given)
        # applies to whichever side wasn't explicitly overridden; falling
        # back to the configured hybrid_bm25_topk/hybrid_dense_topk
        # defaults if neither is set.
        bm25_num = bm25_num if bm25_num is not None else (num if num is not None else self.bm25_topk)
        dense_num = dense_num if dense_num is not None else (num if num is not None else self.dense_topk)

        bm25_future = self._executor.submit(self.bm25_retriever._search, query, bm25_num, True)
        dense_future = self._executor.submit(self.dense_retriever._search, query, dense_num, True)
        bm25_results, bm25_scores = bm25_future.result()
        dense_results, dense_scores = dense_future.result()

        merged = self._merge(bm25_results, bm25_scores, dense_results, dense_scores)
        results = [doc for doc, score in merged]
        scores = [score for doc, score in merged]
        if return_score:
            # Note: bm25 scores and dense (faiss) scores are on different
            # scales and are not directly comparable across sources.
            return results, scores
        return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False,
                       return_source_lists: bool = False, bm25_num: int = None, dense_num: int = None):
        bm25_num = bm25_num if bm25_num is not None else (num if num is not None else self.bm25_topk)
        dense_num = dense_num if dense_num is not None else (num if num is not None else self.dense_topk)

        def _timed(retriever_obj, q_list, k):
            t0 = time.perf_counter()
            res, sc = retriever_obj._batch_search(q_list, k, True)
            return res, sc, time.perf_counter() - t0

        # Submit each side's *whole-batch* search as one task, instead of
        # looping per query and submitting a bm25+dense pair per query.
        # This lets DenseRetriever._batch_search group multiple queries
        # into a single encoder.encode() call (one HTTP request carrying
        # several queries when using SglangEncoder, instead of one request
        # per query), which is what actually lets sglang's server-side
        # batching help for multi-query requests.
        bm25_future = self._executor.submit(_timed, self.bm25_retriever, query_list, bm25_num)
        dense_future = self._executor.submit(_timed, self.dense_retriever, query_list, dense_num)
        bm25_results_list, bm25_scores_list, bm25_time = bm25_future.result()
        dense_results_list, dense_scores_list, dense_time = dense_future.result()
        logger.info(
            "[HybridRetriever] queries=%d bm25=%.3fs dense=%.3fs",
            len(query_list), bm25_time, dense_time,
        )

        results, scores = [], []
        for bm25_results, bm25_scores, dense_results, dense_scores in zip(
            bm25_results_list, bm25_scores_list, dense_results_list, dense_scores_list
        ):
            merged = self._merge(bm25_results, bm25_scores, dense_results, dense_scores)
            results.append([doc for doc, score in merged])
            scores.append([score for doc, score in merged])

        if return_source_lists:
            # Per-query raw ranked candidate lists from each retriever,
            # *before* merge/dedup, so callers can inspect where a given
            # document ranked within bm25 alone vs dense alone.
            if return_score:
                return results, scores, bm25_results_list, dense_results_list
            return results, bm25_results_list, dense_results_list

        if return_score:
            return results, scores
        return results

def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    elif config.retrieval_method == "hybrid":
        return HybridRetriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        cache_save_path: str = None,
        bm25_index_path: str = None,
        hybrid_bm25_topk: int = 100,
        hybrid_dense_topk: int = 100,
        disable_cache: bool = False,
        retrieval_backend: str = "local",
        sglang_embedding_url: str = None,
        faiss_nprobe: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.cache_save_path = cache_save_path
        self.disable_cache = disable_cache
        # hybrid retrieval (sparse + dense) settings
        self.bm25_index_path = bm25_index_path
        self.hybrid_bm25_topk = hybrid_bm25_topk
        self.hybrid_dense_topk = hybrid_dense_topk
        # sglang backend settings (retrieval_backend="sglang" calls a remote
        # sglang embedding server instead of loading the model locally)
        self.retrieval_backend = retrieval_backend
        self.sglang_embedding_url = sglang_embedding_url
        self.faiss_nprobe = faiss_nprobe


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean",
    "jina": "mean",
    "qwen3": "last_token",
}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument("--retrieval_pooling_method", type=str, default=None, help="Override pooling method; inferred from retriever_name if not set.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')
    parser.add_argument('--faiss_nprobe', type=int, default=128, help='Number of IVF clusters to probe per query (no-op for non-IVF indices). Default: 128. Increase for better recall at the cost of latency; typical range is 8-256.')
    parser.add_argument("--cache_save_path", type=str, default=None, help="Path to persist the query cache (json). If not set, cache is kept in-memory only.")
    parser.add_argument('--disable_cache', action='store_true', help='Disable the query cache entirely (no read/write, in-memory or on disk).')

    args = parser.parse_args()

    if args.retrieval_pooling_method is not None:
        pooling_method = args.retrieval_pooling_method
    else:
        pooling_method = "mean"
        for k, v in MODEL2POOLING.items():
            if k in args.retriever_name.lower():
                pooling_method = v
                break
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method=pooling_method,
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=4096,
        cache_save_path=args.cache_save_path,
        disable_cache=args.disable_cache,
        faiss_nprobe=args.faiss_nprobe,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)