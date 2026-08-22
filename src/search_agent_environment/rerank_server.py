import argparse
import atexit
import hashlib
import json
import logging
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict
from dataclasses import dataclass, field

from sentence_transformers import CrossEncoder
import torch
import torch.nn.functional as F
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import requests

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RerankCache:
    """Simple (query, doc) -> score cache. Persisted to a json file if
    cache_path is provided, otherwise kept in-memory only.
    """
    def __init__(self, cache_path: str = None):
        self.cache_path = cache_path
        self.cache: Dict[str, float] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                self.cache = json.load(f)

    @staticmethod
    def _key(query: str, doc: str) -> str:
        return hashlib.md5(f"{query}\x00{doc}".encode("utf-8")).hexdigest()

    def get(self, query: str, doc: str):
        return self.cache.get(self._key(query, doc))

    def set(self, query: str, doc: str, score: float):
        self.cache[self._key(query, doc)] = score

    def save(self):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)


class BaseCrossEncoder:
    def __init__(self, model, batch_size=32, device="cuda", cache_save_path=None, disable_cache=False):
        self.model = model
        self.batch_size = batch_size
        self.model.to(device)
        self.disable_cache = disable_cache
        self.rerank_cache = RerankCache(cache_save_path)

    def _passage_to_string(self, doc_item):
        if "document" not in doc_item:
            content = doc_item.get("retrieval_text", doc_item.get("text"))
        else:
            content = doc_item['document'].get("retrieval_text", doc_item['document'].get("text"))
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])

        return f"(Title: {title}) {text}"

    def rerank(self, 
               queries: list[str], 
               documents: list[list[dict]]):
        """
        Assume documents is a list of list of dicts, where each dict is a document with keys "id" and "contents".
        This asumption is made to be consistent with the output of the retrieval server.
        """ 
        assert len(queries) == len(documents)

        all_pairs = []
        doc_items = []
        qids = []
        for qid, query in enumerate(queries):
            for doc_item in documents[qid]:
                doc = self._passage_to_string(doc_item)
                all_pairs.append((query, doc))
                doc_items.append(doc_item)
                qids.append(qid)

        scores: list = [None] * len(all_pairs)
        if self.disable_cache:
            to_predict_pairs = list(all_pairs)
            to_predict_indices = list(range(len(all_pairs)))
        else:
            to_predict_pairs = []
            to_predict_indices = []
            for i, (query, doc) in enumerate(all_pairs):
                cached_score = self.rerank_cache.get(query, doc)
                if cached_score is not None:
                    scores[i] = cached_score
                else:
                    to_predict_pairs.append((query, doc))
                    to_predict_indices.append(i)

        if to_predict_pairs:
            predicted_scores = self._predict(to_predict_pairs)
            for idx, score in zip(to_predict_indices, predicted_scores):
                scores[idx] = score
                if not self.disable_cache:
                    query, doc = all_pairs[idx]
                    self.rerank_cache.set(query, doc, score)
            if not self.disable_cache:
                self.rerank_cache.save()

        query_to_doc_scores = defaultdict(list)
        assert len(scores) == len(all_pairs) == len(qids)
        for i in range(len(all_pairs)):
            query, doc = all_pairs[i]
            score = scores[i] 
            qid = qids[i]
            query_to_doc_scores[qid].append((doc, score, doc_items[i]))

        sorted_query_to_doc_scores = {}
        for qid, doc_scores in query_to_doc_scores.items():
            sorted_query_to_doc_scores[qid] = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        return sorted_query_to_doc_scores

    def _predict(self, pairs: list[tuple[str, str]]):
        raise NotImplementedError 

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        raise NotImplementedError


class SentenceTransformerCrossEncoder(BaseCrossEncoder):
    def __init__(self, model, batch_size=32, device="cuda", cache_save_path=None, disable_cache=False, use_multi_gpu=True):
        super().__init__(model, batch_size, device, cache_save_path, disable_cache)

        # sentence-transformers' CrossEncoder.predict() does its own internal
        # tokenization/batching, so a raw torch.nn.DataParallel wrap around
        # the underlying HF model would not integrate with it. Multi-GPU
        # parallelism is instead done via the library's own multi-process
        # pool (one worker process per GPU), which predict() dispatches to
        # when a pool is passed in.
        self.pool = None
        if (
            use_multi_gpu
            and device == "cuda"
            and torch.cuda.device_count() > 1
            and hasattr(self.model, "start_multi_process_pool")
        ):
            self.pool = self.model.start_multi_process_pool()
            atexit.register(self.model.stop_multi_process_pool, self.pool)

        logger.info(
            "[SentenceTransformerCrossEncoder] cuda_device_count=%d "
            "has_multi_process_pool_api=%s use_multi_gpu=%s -> pool_active=%s",
            torch.cuda.device_count(),
            hasattr(self.model, "start_multi_process_pool"),
            use_multi_gpu,
            self.pool is not None,
        )

    def _predict(self, pairs: list[tuple[str, str]]):
        if self.pool is not None:
            scores = self.model.predict(pairs, batch_size=self.batch_size, pool=self.pool)
        else:
            scores = self.model.predict(pairs, batch_size=self.batch_size)
        scores = scores.tolist() if isinstance(scores, torch.Tensor) or isinstance(scores, np.ndarray) else scores
        return scores

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        model = CrossEncoder(model_name_or_path)
        return cls(model, **kwargs)


class Qwen3CrossEncoder(BaseCrossEncoder):
    """Reranker based on Qwen3-Reranker, a generative model that scores
    query-document relevance via the softmax probability of "yes" vs "no"
    tokens instead of a classification head.
    """

    def __init__(self, model, tokenizer, batch_size=32, device="cuda", max_length=512,
                 instruction=None, cache_save_path=None, disable_cache=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction or "Given a web search query, determine if the document is relevant to the query."
        self.device = device

        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = tokenizer.encode(self.suffix, add_special_tokens=False)

        self.token_true_id = tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = tokenizer.convert_tokens_to_ids("no")

        super().__init__(model, batch_size, device, cache_save_path, disable_cache)

    def _format_input(self, query, doc):
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    @torch.no_grad()
    def _predict(self, pairs: list[tuple[str, str]]):
        scores = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            texts = [self._format_input(query, doc) for query, doc in batch]

            batch_inputs = self.tokenizer(
                texts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
            )
            input_ids = [self.prefix_tokens + ids + self.suffix_tokens for ids in batch_inputs["input_ids"]]

            batch_inputs = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

            outputs = self.model(**batch_inputs)
            logits = outputs.logits[:, -1, :]
            true_logits = logits[:, self.token_true_id]
            false_logits = logits[:, self.token_false_id]
            stacked = torch.stack([false_logits, true_logits], dim=1)
            batch_scores = F.softmax(stacked, dim=1)[:, 1]
            scores.extend(batch_scores.tolist())

        return scores

    @classmethod
    def load(cls, model_name_or_path, batch_size=32, device="cuda", max_length=512, instruction=None, cache_save_path=None, disable_cache=False, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        model.eval()
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        return cls(model, tokenizer, batch_size=batch_size, device=device, max_length=max_length,
                    instruction=instruction, cache_save_path=cache_save_path, disable_cache=disable_cache)


class SglangCrossEncoder(BaseCrossEncoder):
    """Reranks by calling a remote sglang server via /v1/completions and
    computing the yes/no probability of a decoder-only reranker
    (Qwen3-Reranker) client-side from the returned top-token logprobs.

    Why not /v1/rerank + a custom chat template? sglang's /v1/rerank
    endpoint (as of >=0.4.9) internally constructs `messages` with
    non-standard roles (e.g. "query", "document") when the model has a
    rerank-flavored chat template. sglang's request validation, however,
    only allows standard chat roles (system/user/assistant/tool/...),
    so the template's `query`/`document` role selectors never resolve,
    and every document ends up scored with an empty prompt -- producing
    identical (constant) scores per document regardless of content.

    /v1/completions bypasses both the chat-template layer and the
    /v1/rerank message-construction layer: we hand-craft the exact
    prompt string Qwen3-Reranker was trained on, ask for max_tokens=1
    with logprobs, and read the yes/no token probabilities directly out
    of the response. This matches Qwen3CrossEncoder._predict's local
    implementation (system judge prompt + <Instruct>/<Query>/<Document>
    in user turn + <think></think> suffix) exactly.

    For cross-encoder-style rerankers (bge-reranker-v2-m3 etc.) launched
    WITH --is-embedding, callers should keep using the /v1/rerank endpoint
    -- but that path is not exercised here; this class is currently
    Qwen3-Reranker-specific (assumes yes/no scoring).
    """

    _PROMPT_PREFIX = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
        "<|im_end|>\n<|im_start|>user\n"
    )
    _PROMPT_SUFFIX = (
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    def __init__(self, sglang_url, model_name, batch_size=32, cache_save_path=None,
                 disable_cache=False, instruct=None, timeout=60, concurrency=32):
        self.sglang_url = sglang_url.rstrip("/")
        self.model_name = model_name
        self.batch_size = batch_size
        self.disable_cache = disable_cache
        self.rerank_cache = RerankCache(cache_save_path)
        self.instruct = instruct or "Given a web search query, retrieve relevant passages that answer the query."
        self.timeout = timeout
        self.concurrency = concurrency

    def _build_prompt(self, query: str, doc: str) -> str:
        return (
            f"{self._PROMPT_PREFIX}"
            f"<Instruct>: {self.instruct}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
            f"{self._PROMPT_SUFFIX}"
        )

    @staticmethod
    def _extract_yes_no_logprobs(top_logprobs):
        """Return (yes_logprob, no_logprob) from sglang /v1/completions'
        first-token top_logprobs. Handles both dict-shaped
        ({"yes": lp, ...}) and list-shaped ([{"token": "yes", "logprob": lp}, ...])
        variants across sglang versions. If a token is missing from the
        top-K, use a very small logprob so its softmax contribution is
        negligible.
        """
        yes_lp = None
        no_lp = None
        if isinstance(top_logprobs, dict):
            for tok, lp in top_logprobs.items():
                t = tok.strip().lower()
                if t == "yes" and yes_lp is None:
                    yes_lp = lp
                elif t == "no" and no_lp is None:
                    no_lp = lp
        elif isinstance(top_logprobs, list):
            for item in top_logprobs:
                tok = item.get("token", "").strip().lower()
                lp = item.get("logprob")
                if tok == "yes" and yes_lp is None:
                    yes_lp = lp
                elif tok == "no" and no_lp is None:
                    no_lp = lp
        if yes_lp is None:
            yes_lp = -1e9
        if no_lp is None:
            no_lp = -1e9
        return yes_lp, no_lp

    def _score_one(self, query: str, doc: str) -> float:
        prompt = self._build_prompt(query, doc)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 20,
        }
        response = requests.post(
            f"{self.sglang_url}/v1/completions", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        # OpenAI-style /v1/completions response:
        # choices[0].logprobs.top_logprobs is a list (one entry per output
        # token). With max_tokens=1 that's a single-element list.
        logprobs_obj = data["choices"][0].get("logprobs") or {}
        top_list = logprobs_obj.get("top_logprobs") or []
        top_logprobs = top_list[0] if top_list else {}
        yes_lp, no_lp = self._extract_yes_no_logprobs(top_logprobs)
        # Numerically stable softmax over just {no, yes}.
        m = max(yes_lp, no_lp)
        yes_exp = math.exp(yes_lp - m)
        no_exp = math.exp(no_lp - m)
        return yes_exp / (yes_exp + no_exp)

    def _predict(self, pairs: list[tuple[str, str]]):
        # Send one /v1/completions request per (query, doc) pair, with
        # up to `self.concurrency` requests in flight at once so
        # sglang's continuous-batching scheduler can pack them together
        # server-side.
        scores = [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {
                executor.submit(self._score_one, query, doc): i
                for i, (query, doc) in enumerate(pairs)
            }
            for future in future_to_idx:
                i = future_to_idx[future]
                scores[i] = future.result()
        return scores

    @classmethod
    def load(cls, model_name_or_path, sglang_url=None, batch_size=32, cache_save_path=None,
             disable_cache=False, instruct=None, timeout=60, concurrency=32, **kwargs):
        if not sglang_url:
            raise ValueError("SglangCrossEncoder requires --sglang_rerank_url to be set.")
        return cls(
            sglang_url=sglang_url,
            model_name=model_name_or_path,
            batch_size=batch_size,
            cache_save_path=cache_save_path,
            disable_cache=disable_cache,
            instruct=instruct,
            timeout=timeout,
            concurrency=concurrency,
        )


class RerankRequest(BaseModel):
    queries: list[str]
    documents: list[list[dict]]
    rerank_topk: Optional[int] = None
    return_scores: bool = False


@dataclass 
class RerankerArguments:
    max_length: int = field(default=512)
    rerank_topk: int = field(default=3)
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")
    batch_size: int = field(default=32)
    reranker_type: str = field(default="sentence_transformer")
    cache_save_path: str = field(default=None)
    disable_cache: bool = field(default=False)
    sglang_rerank_url: str = field(default=None)
    sglang_rerank_instruct: str = field(default=None)

def get_reranker(config):
    if config.reranker_type == "sentence_transformer":
        return SentenceTransformerCrossEncoder.load(
            config.rerank_model_name_or_path,
            batch_size=config.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            cache_save_path=config.cache_save_path,
            disable_cache=config.disable_cache,
        )
    elif config.reranker_type == "qwen3":
        return Qwen3CrossEncoder.load(
            config.rerank_model_name_or_path,
            batch_size=config.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_length=config.max_length,
            cache_save_path=config.cache_save_path,
            disable_cache=config.disable_cache,
        )
    elif config.reranker_type == "sglang":
        return SglangCrossEncoder.load(
            config.rerank_model_name_or_path,
            sglang_url=config.sglang_rerank_url,
            batch_size=config.batch_size,
            cache_save_path=config.cache_save_path,
            disable_cache=config.disable_cache,
            instruct=config.sglang_rerank_instruct,
        )
    else:
        raise ValueError(f"Unknown reranker type: {config.reranker_type}")


app = FastAPI()

@app.post("/rerank")
def rerank_endpoint(request: RerankRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "documents": [[doc_item_1, ..., doc_item_k], [doc_item_1, ..., doc_item_k]],
      "rerank_topk": 3,
      "return_scores": true
    }
    """
    if not request.rerank_topk:
        request.rerank_topk = config.rerank_topk  # fallback to default

    # Perform batch re reranking
    # doc_scores already sorted by score
    query_to_doc_scores = reranker.rerank(request.queries, request.documents) 

    # Format response 
    resp = []
    for _, doc_scores in query_to_doc_scores.items():
        doc_scores = doc_scores[:request.rerank_topk]
        if request.return_scores:
            combined = [] 
            for doc, score, doc_item in doc_scores:
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append([doc for doc, _, _ in doc_scores])
    return {"result": resp}


if __name__ == "__main__":
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    parser = HfArgumentParser((RerankerArguments))
    config = parser.parse_args_into_dataclasses()[0]

    # 2) Instantiate a global retriever so it is loaded once and reused.
    reranker = get_reranker(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=6980)
