# pip install -U sentence-transformers
import os
import re
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import quote, unquote, urlparse

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from retrieval_server import get_retriever, load_corpus, Config as RetrieverConfig, HybridRetriever
from rerank_server import SentenceTransformerCrossEncoder, Qwen3CrossEncoder, SglangCrossEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class _SuppressHealthcheckAccessLog(logging.Filter):
    """Drop uvicorn.access log lines coming from external platform health
    probes hitting paths this server does not implement (e.g. FastDeploy
    /fastdeploy/server/info, Triton /v2/health/ready). They spam the log
    with 404s but are harmless. Only lines matching these known probe
    paths are filtered; genuine 404s to unknown user-facing paths still
    log normally.
    """

    _NOISY_PATHS = (
        "/fastdeploy/server/info",
        "/v2/health/ready",
        "/v2/health/live",
        "/health",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(path in msg for path in self._NOISY_PATHS)


logging.getLogger("uvicorn.access").addFilter(_SuppressHealthcheckAccessLog())

app = FastAPI()

def convert_title_format(text):
    # Use regex to extract the title and the content
    match = re.match(r'\(Title:\s*([^)]+)\)\s*(.+)', text, re.DOTALL)
    if match:
        title, content = match.groups()
        return f'\"{title}\"\n{content}'
    else:
        return text

def _display_text(row):
    return row.get("display_text") if row.get("display_text") is not None else row.get("retrieval_text")

def _as_py(value):
    """Unwrap numpy scalar types (np.int64 etc.) coming from parquet columns."""
    if isinstance(value, np.generic):
        return value.item()
    return value

def wikipedia_article_url(title: str) -> Optional[str]:
    if not title:
        return None
    normalized = title.strip().replace(" ", "_")
    encoded_title = quote(normalized, safe="/_-().~'")
    return f"https://en.wikipedia.org/wiki/{encoded_title}"

def parse_wikipedia_article_url(url: str) -> str:
    parsed = urlparse(url)

    prefix = "/wiki/"
    if not parsed.path.startswith(prefix):
        raise ValueError("不是 Wikipedia 文章 URL")

    encoded_title = parsed.path[len(prefix):]
    title = unquote(encoded_title)
    return title.replace("_", " ")

class DocumentIndex:
    """In-memory index mapping document_id (wikipedia_id) -> ordered list of
    corpus row indices belonging to that document, sorted by chunk_index.
    Built once at startup by scanning the corpus (assumes a corpus small
    enough to fit in memory).
    """

    def __init__(self, corpus):
        self.corpus = corpus

        # Column-wise extraction avoids per-row Arrow->Python dict conversion.
        t0 = time.perf_counter()
        raw_doc_ids = corpus["wikipedia_id"]
        raw_chunk_indices = corpus["chunk_index"]
        t_extract = time.perf_counter() - t0

        t0 = time.perf_counter()
        # Factorize wikipedia_id into small integer codes for fast sorting.
        # This project's real wikipedia_id values are non-numeric strings
        # (final dtype ends up as e.g. '<U8'), so the previous approach --
        # attempt .astype(int64), catch the failure, then .astype(str) --
        # did two full per-element conversion passes over ~26M objects
        # (int() then str() on every element), which was the actual
        # bottleneck (~487s on the full corpus). pd.factorize does a single
        # hash-based C-level pass and handles nulls natively (code -1),
        # regardless of whether ids are numeric-looking or arbitrary
        # strings.
        codes, uniques = pd.factorize(np.asarray(raw_doc_ids), sort=False, use_na_sentinel=True)
        valid_mask = codes != -1
        valid_idxs = np.flatnonzero(valid_mask)
        valid_doc_codes = codes[valid_mask].astype(np.int64, copy=False)
        valid_chunk_indices = np.asarray(raw_chunk_indices)[valid_mask]
        if valid_chunk_indices.dtype == object:
            valid_chunk_indices = valid_chunk_indices.astype(np.int64)
        t_prepare = time.perf_counter() - t0

        # Single vectorized sort (doc_id primary, chunk_index secondary)
        # instead of a Python loop + one sort() call per document: with
        # ~26M rows / ~480K documents, hundreds of thousands of per-group
        # Python-level sort() calls (each invoking a lambda per comparison)
        # dominate wall-clock time. lexsort does the equivalent in one
        # vectorized C-level pass -- as long as the arrays are a native
        # dtype (int64), not `object`.
        t0 = time.perf_counter()
        order = np.lexsort((valid_chunk_indices, valid_doc_codes))
        sorted_row_idxs = valid_idxs[order]
        sorted_doc_codes = valid_doc_codes[order]
        t_sort = time.perf_counter() - t0

        t0 = time.perf_counter()
        group_starts = np.concatenate(([0], np.flatnonzero(sorted_doc_codes[1:] != sorted_doc_codes[:-1]) + 1))
        group_row_idxs = np.split(sorted_row_idxs, group_starts[1:])

        self.by_doc: dict = {
            uniques[sorted_doc_codes[start]]: idxs.tolist()
            for start, idxs in zip(group_starts, group_row_idxs)
        }
        t_group = time.perf_counter() - t0

        t0 = time.perf_counter()
        # Build title -> doc_id using column-wise extraction + the row
        # indices we already computed, instead of one self.corpus[i] random
        # row access per document (~480K single-row Arrow materializations,
        # which dominated wall-clock time -- ~76s vs <1s for a bulk column
        # extraction of the same column).
        raw_titles = corpus["wikipedia_title"]
        title_arr = np.asarray(raw_titles)[valid_idxs][order]
        first_row_of_group = title_arr[group_starts]

        self.title_to_doc_id: dict = {}
        for doc_id, title in zip(self.by_doc.keys(), first_row_of_group):
            if title is not None:
                self.title_to_doc_id[_as_py(title)] = doc_id
        t_title_index = time.perf_counter() - t0

        logger.info(
            "[DocumentIndex] extract=%.2fs prepare=%.2fs sort=%.2fs group=%.2fs title_index=%.2fs (num unique doc_ids: %d)",
            t_extract, t_prepare, t_sort, t_group, t_title_index, len(uniques),
        )

    def get_full_document(self, doc_id) -> str:
        idxs = self.by_doc.get(doc_id)
        if not idxs:
            return ""
        return "\n".join(_display_text(self.corpus.iloc[i].to_dict()) or "" for i in idxs)

    def get_title(self, doc_id) -> Optional[str]:
        idxs = self.by_doc.get(doc_id)
        if not idxs:
            return None
        return self.corpus.iloc[idxs[0]].get("wikipedia_title")

    def get_passages(self, doc_id) -> list:
        idxs = self.by_doc.get(doc_id, [])
        return [self.corpus.iloc[i].to_dict() for i in idxs]

    def get_doc_id_by_title(self, title: str) -> Optional[str]:
        return self.title_to_doc_id.get(title)

def _dedup_by_document(doc_scores):
    """Given a list of (passage_text, score, doc_item) sorted by score desc,
    keep only the highest-scoring passage per document (wikipedia_id).
    Passages without a document_id are each treated as their own document.
    """
    best_per_doc = {}
    order = []
    for doc, score, doc_item in doc_scores:
        key = doc_item.get("wikipedia_id")
        if key is None:
            key = doc_item.get("chunk_id") or id(doc_item)
        key = _as_py(key)
        if key not in best_per_doc:
            best_per_doc[key] = (doc, score, doc_item)
            order.append(key)
    return [best_per_doc[key] for key in order]

# ----------- Combined Request Schema -----------
class SearchRequest(BaseModel):
    queries: List[str]
    topk_retrieval: Optional[int] = 10
    topk_rerank: Optional[int] = 3
    return_scores: bool = False

# ----------- red searcher style Request Schemas -----------
class RedSearchRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 3  # number of distinct documents to return
    topk_retrieval: Optional[int] = 100  # coarse retrieval pool size before rerank/dedup (applies to both bm25 and dense unless overridden below)
    bm25_topk_retrieval: Optional[int] = None  # override: bm25-only coarse retrieval pool size (hybrid retriever only)
    dense_topk_retrieval: Optional[int] = None  # override: dense-only coarse retrieval pool size (hybrid retriever only)
    snippet_char_len: Optional[int] = 500  # chars to keep from the top-scoring document; set to null/None (or <= 0) to return the full document untruncated
    return_scores: bool = False
    debug_retriever_sources: bool = False  # diagnostic: also return per-retriever (bm25/dense) raw ranked candidates before merge/dedup

class VisitItem(BaseModel):
    url: str
    query: str

class RedVisitRequest(BaseModel):
    items: List[VisitItem]
    topk: Optional[int] = 2  # number of passages to return per document
    return_scores: bool = False

class SearchChunksRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 5  # number of chunks to return (chunks, not distinct documents)
    topk_retrieval: Optional[int] = 100  # coarse retrieval pool size before rerank (applies to both bm25 and dense unless overridden below)
    bm25_topk_retrieval: Optional[int] = None  # override: bm25-only coarse retrieval pool size (hybrid retriever only)
    dense_topk_retrieval: Optional[int] = None  # override: dense-only coarse retrieval pool size (hybrid retriever only)
    return_scores: bool = False

# ----------- Reranker Config Schema -----------
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

# ----------- Endpoint -----------
@app.post("/retrieve_rerank")
def search_endpoint(request: SearchRequest):
    request_start = time.perf_counter()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Step 1: Retrieve documents
    t0 = time.perf_counter()
    retrieved_docs = retriever.batch_search(
        query_list=request.queries,
        num=request.topk_retrieval,
        return_score=False
    )
    retrieval_time = time.perf_counter() - t0

    # Step 2: Rerank
    t0 = time.perf_counter()
    reranked = reranker.rerank(request.queries, retrieved_docs)
    rerank_time = time.perf_counter() - t0

    # Step 3: Format response
    t0 = time.perf_counter()
    response = []
    for i, doc_scores in reranked.items():
        doc_scores = doc_scores[:request.topk_rerank]
        if request.return_scores:
            combined = []
            for doc, score, doc_item in doc_scores:
                combined.append({
                    "document": convert_title_format(doc),
                    "score": score,
                    "passage_id": _as_py(doc_item.get("chunk_id")),
                    "document_id": doc_item.get("wikipedia_id"),
                    "document_title": doc_item.get("wikipedia_title"),
                })
            response.append(combined)
        else:
            response.append([
                {
                    "document": convert_title_format(doc),
                    "passage_id": _as_py(doc_item.get("chunk_id")),
                    "document_id": doc_item.get("wikipedia_id"),
                    "document_title": doc_item.get("wikipedia_title"),
                }
                for doc, _, doc_item in doc_scores
            ])
    format_time = time.perf_counter() - t0

    total_time = time.perf_counter() - request_start
    logger.info(
        "[%s] queries=%d retrieval=%.3fs rerank=%.3fs format=%.3fs total=%.3fs",
        timestamp, len(request.queries), retrieval_time, rerank_time, format_time, total_time,
    )

    return {
        "result": response,
        "timing": {
            "timestamp": timestamp,
            "retrieval_time": retrieval_time,
            "rerank_time": rerank_time,
            "format_time": format_time,
            "total_time": total_time,
        },
    }


def _retrieve_and_rerank_docs(queries: List[str], topk_retrieval: int, return_source_lists: bool = False,
                               bm25_num: int = None, dense_num: int = None, dedup_by_document: bool = True):
    """Shared pipeline for /search: coarse retrieval -> rerank -> dedup to
    one (best) passage per document, sorted by score desc.
    Returns a list (per query) of (passage_text, score, doc_item) tuples,
    plus the retrieval/rerank timing breakdown.

    If `return_source_lists` is True and the configured retriever is a
    HybridRetriever, also returns the raw per-retriever ranked candidate
    lists (bm25-only, dense-only) *before* merge/dedup, bypassing the
    query cache. This is a diagnostic-only path used to inspect where a
    given document ranked within each individual retriever.

    `bm25_num`/`dense_num`, when given, override `topk_retrieval` for
    just the bm25 or dense side respectively (HybridRetriever only;
    ignored otherwise). Using either bypasses the query cache, same as
    `return_source_lists`, since the cache key is not overload-aware.

    If `dedup_by_document` is False, results are score-sorted *chunks*
    (one entry per passage, not collapsed to one per document) -- used by
    /search_chunks, where multiple passages from the same document are
    allowed to each appear in the result if they rank highly.

    If the module-level `SKIP_RERANK_IN_SEARCH` is True (server started
    with --disable_reranker_in_search), the rerank step is skipped for
    /search only and the retriever's own ranking/score is kept as-is.
    /visit always uses the reranker regardless of this flag.
    """
    bm25_lists = dense_lists = None
    need_scores = SKIP_RERANK_IN_SEARCH  # skip-rerank path needs the retriever's own scores
    has_per_side_override = bm25_num is not None or dense_num is not None
    t0 = time.perf_counter()
    if (return_source_lists or has_per_side_override) and isinstance(retriever, HybridRetriever):
        retrieved_docs, retrieved_scores, bm25_lists, dense_lists = retriever._batch_search(
            query_list=queries,
            num=topk_retrieval,
            return_score=True,
            return_source_lists=True,
            bm25_num=bm25_num,
            dense_num=dense_num,
        )
        if not return_source_lists:
            bm25_lists = dense_lists = None
        if not need_scores:
            retrieved_scores = None
    else:
        retrieved = retriever.batch_search(
            query_list=queries,
            num=topk_retrieval,
            return_score=need_scores,
        )
        retrieved_docs, retrieved_scores = retrieved if need_scores else (retrieved, None)
    retrieval_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    if SKIP_RERANK_IN_SEARCH:
        # No rerank step: keep the retriever's own per-chunk score/order,
        # just reshape into the same (doc, score, doc_item) tuple form the
        # dedup step expects.
        reranked = {}
        for i, doc_items in enumerate(retrieved_docs):
            scores = retrieved_scores[i] if retrieved_scores is not None else [0.0] * len(doc_items)
            reranked[i] = [
                (_display_text(doc_item) or "", score, doc_item)
                for doc_item, score in zip(doc_items, scores)
            ]
    else:
        reranked = reranker.rerank(queries, retrieved_docs)
    rerank_time = time.perf_counter() - t0

    per_query_results = []
    for i in range(len(queries)):
        doc_scores = reranked.get(i, [])
        if dedup_by_document:
            per_query_results.append(_dedup_by_document(doc_scores))
        else:
            # HybridRetriever._merge (used when SKIP_RERANK_IN_SEARCH skips
            # the reranker) concatenates bm25+dense candidates in their
            # original retrieval order rather than by score, so sort
            # explicitly here to guarantee score-descending chunks
            # regardless of which upstream path produced `doc_scores`.
            per_query_results.append(sorted(doc_scores, key=lambda x: x[1], reverse=True))

    if return_source_lists:
        return per_query_results, retrieval_time, rerank_time, bm25_lists, dense_lists
    return per_query_results, retrieval_time, rerank_time


def _dedup_titles_by_document(doc_items):
    """Given a ranked (score desc) list of raw retriever doc_items, dedup by
    wikipedia_id (falling back to chunk_id) keeping first occurrence, and
    return the ordered list of (document_id, document_title) for each
    distinct document. Used for the bm25-only / dense-only diagnostic
    ranking (independent of the rerank/dedup pipeline used for /search's
    main result).
    """
    seen = set()
    ordered = []
    for doc_item in doc_items:
        key = doc_item.get("wikipedia_id")
        if key is None:
            key = doc_item.get("chunk_id") or id(doc_item)
        key = _as_py(key)
        if key in seen:
            continue
        seen.add(key)
        ordered.append({
            "document_id": _as_py(doc_item.get("wikipedia_id")),
            "document_title": doc_item.get("wikipedia_title"),
        })
    return ordered


# ----------- red searcher style Endpoints -----------
@app.post("/search")
def red_search_endpoint(request: RedSearchRequest):
    """Returns, for each query, the topk most relevant *documents*. Each
    document is represented by a snippet: the full document (all its
    passages concatenated in paragraph_id order) truncated to
    `snippet_char_len` characters. Document relevance/ranking is determined
    by each document's highest-scoring passage after rerank.

    Empty/whitespace-only query strings are skipped rather than sent to
    the retriever/reranker (dense embedding backends like sglang reject
    empty input with a 400 error, which would otherwise fail the whole
    batch): their result entry is an empty list, and all other queries in
    the same request are still processed normally.
    """
    request_start = time.perf_counter()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    n_total = len(request.queries)

    valid_indices = [i for i, q in enumerate(request.queries) if q and q.strip()]
    valid_queries = [request.queries[i] for i in valid_indices]
    skipped_indices = [i for i in range(n_total) if i not in set(valid_indices)]
    if skipped_indices:
        logger.warning(
            "[%s] /search skipping %d empty/whitespace query(ies) at index(es) %s out of %d total",
            timestamp, len(skipped_indices), skipped_indices, n_total,
        )

    t0 = time.perf_counter()
    if valid_queries:
        if request.debug_retriever_sources:
            valid_docs, retrieval_time, rerank_time, bm25_lists, dense_lists = _retrieve_and_rerank_docs(
                valid_queries, request.topk_retrieval, return_source_lists=True,
                bm25_num=request.bm25_topk_retrieval, dense_num=request.dense_topk_retrieval,
            )
        else:
            valid_docs, retrieval_time, rerank_time = _retrieve_and_rerank_docs(
                valid_queries, request.topk_retrieval,
                bm25_num=request.bm25_topk_retrieval, dense_num=request.dense_topk_retrieval,
            )
            bm25_lists = dense_lists = None
    else:
        valid_docs, retrieval_time, rerank_time = [], 0.0, 0.0
        bm25_lists = dense_lists = None
    pipeline_time = time.perf_counter() - t0

    # Scatter the valid-queries-only results (per_query_docs/bm25_lists/
    # dense_lists are all aligned with valid_queries, not the original
    # request.queries) back to full length, filling [] for the
    # empty-query positions that were skipped above.
    per_query_docs = [[] for _ in range(n_total)]
    for valid_pos, orig_idx in enumerate(valid_indices):
        per_query_docs[orig_idx] = valid_docs[valid_pos]

    if bm25_lists is not None:
        bm25_lists_full = [[] for _ in range(n_total)]
        dense_lists_full = [[] for _ in range(n_total)]
        for valid_pos, orig_idx in enumerate(valid_indices):
            bm25_lists_full[orig_idx] = bm25_lists[valid_pos]
            dense_lists_full[orig_idx] = dense_lists[valid_pos]
        bm25_lists, dense_lists = bm25_lists_full, dense_lists_full

    t0 = time.perf_counter()
    response = []
    dedup_doc_counts = []
    for doc_list in per_query_docs:
        dedup_doc_counts.append(len(doc_list))
        doc_list = doc_list[:request.topk]
        entries = []
        for doc, score, doc_item in doc_list:
            doc_id = doc_item.get("wikipedia_id")
            title = doc_item.get("wikipedia_title")
            if doc_id is not None:
                full_doc = document_index.get_full_document(doc_id)
            else:
                full_doc = _display_text(doc_item) or doc
            snippet = full_doc if not request.snippet_char_len or request.snippet_char_len <= 0 else full_doc[:request.snippet_char_len]
            entry = {
                "document": convert_title_format(snippet),
                "document_id": doc_id,
                "document_title": title,
                "url": wikipedia_article_url(title),
            }
            if request.return_scores:
                entry["score"] = score
            entries.append(entry)
        response.append(entries)
    format_time = time.perf_counter() - t0

    total_time = time.perf_counter() - request_start
    logger.info(
        "[%s] /search queries=%d retrieval=%.3fs rerank=%.3fs pipeline=%.3fs format=%.3fs total=%.3fs dedup_doc_counts=%s",
        timestamp, len(request.queries), retrieval_time, rerank_time, pipeline_time, format_time, total_time, dedup_doc_counts,
    )

    result = {
        "result": response,
        "dedup_doc_counts": dedup_doc_counts,
        "timing": {
            "timestamp": timestamp,
            "retrieval_time": retrieval_time,
            "rerank_time": rerank_time,
            "pipeline_time": pipeline_time,
            "format_time": format_time,
            "total_time": total_time,
        },
    }
    if request.debug_retriever_sources:
        if bm25_lists is not None and dense_lists is not None:
            result["retriever_debug"] = {
                "bm25_ranked_documents": [_dedup_titles_by_document(docs) for docs in bm25_lists],
                "dense_ranked_documents": [_dedup_titles_by_document(docs) for docs in dense_lists],
            }
        else:
            result["retriever_debug"] = None  # configured retriever is not a HybridRetriever
    return result


@app.post("/search_chunks")
def red_search_chunks_endpoint(request: SearchChunksRequest):
    """Returns, for each query, the topk most relevant *chunks* (passages),
    score-sorted -- unlike /search, results are NOT deduplicated to one
    passage per document, so multiple high-scoring chunks from the same
    document can all appear in the result.

    Empty/whitespace-only query strings are skipped the same way /search
    handles them: their result entry is an empty list, other queries in
    the same request are processed normally.
    """
    request_start = time.perf_counter()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    n_total = len(request.queries)

    valid_indices = [i for i, q in enumerate(request.queries) if q and q.strip()]
    valid_queries = [request.queries[i] for i in valid_indices]
    skipped_indices = [i for i in range(n_total) if i not in set(valid_indices)]
    if skipped_indices:
        logger.warning(
            "[%s] /search_chunks skipping %d empty/whitespace query(ies) at index(es) %s out of %d total",
            timestamp, len(skipped_indices), skipped_indices, n_total,
        )

    t0 = time.perf_counter()
    if valid_queries:
        valid_chunks, retrieval_time, rerank_time = _retrieve_and_rerank_docs(
            valid_queries, request.topk_retrieval, dedup_by_document=False,
            bm25_num=request.bm25_topk_retrieval, dense_num=request.dense_topk_retrieval,
        )
    else:
        valid_chunks, retrieval_time, rerank_time = [], 0.0, 0.0
    pipeline_time = time.perf_counter() - t0

    per_query_chunks = [[] for _ in range(n_total)]
    for valid_pos, orig_idx in enumerate(valid_indices):
        per_query_chunks[orig_idx] = valid_chunks[valid_pos]

    t0 = time.perf_counter()
    response = []
    for chunk_list in per_query_chunks:
        chunk_list = chunk_list[:request.topk]
        entries = []
        for doc, score, doc_item in chunk_list:
            chunk_id = _as_py(doc_item.get("chunk_id"))
            document_id = _as_py(doc_item.get("wikipedia_id"))
            title = doc_item.get("wikipedia_title")
            entry = {
                "document": convert_title_format(doc),
                "passage_id": f"{document_id}:{chunk_id}" if document_id is not None and chunk_id is not None else None,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document_title": title,
                "url": wikipedia_article_url(title),
            }
            if request.return_scores:
                entry["score"] = score
            entries.append(entry)
        response.append(entries)
    format_time = time.perf_counter() - t0

    total_time = time.perf_counter() - request_start
    logger.info(
        "[%s] /search_chunks queries=%d retrieval=%.3fs rerank=%.3fs pipeline=%.3fs format=%.3fs total=%.3fs",
        timestamp, n_total, retrieval_time, rerank_time, pipeline_time, format_time, total_time,
    )

    return {
        "result": response,
        "timing": {
            "timestamp": timestamp,
            "retrieval_time": retrieval_time,
            "rerank_time": rerank_time,
            "pipeline_time": pipeline_time,
            "format_time": format_time,
            "total_time": total_time,
        },
    }



@app.post("/visit")
def red_visit_endpoint(request: RedVisitRequest):
    """For each (url, query) item, resolve the url to a wikipedia article
    title, look up the corresponding document, rerank all passages
    belonging to that document against the query, and return the topk most
    relevant passages (not the full concatenated document).
    """
    request_start = time.perf_counter()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    t0 = time.perf_counter()
    queries = [item.query for item in request.items]
    doc_ids = []
    passages_per_item = []
    for item in request.items:
        try:
            title = parse_wikipedia_article_url(item.url)
        except ValueError:
            doc_ids.append(None)
            passages_per_item.append([])
            continue
        doc_id = document_index.get_doc_id_by_title(title)
        doc_ids.append(doc_id)
        passages_per_item.append(document_index.get_passages(doc_id) if doc_id is not None else [])
    lookup_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    reranked = reranker.rerank(queries, passages_per_item)
    rerank_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    response = []
    for i, item in enumerate(request.items):
        doc_scores = reranked.get(i, [])[:request.topk]
        entries = []
        for doc, score, doc_item in doc_scores:
            chunk_id = _as_py(doc_item.get("chunk_id"))
            document_id = _as_py(doc_item.get("wikipedia_id", doc_ids[i]))
            entry = {
                "document": convert_title_format(doc),
                "passage_id": f"{document_id}:{chunk_id}" if document_id is not None and chunk_id is not None else None,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document_title": doc_item.get("wikipedia_title"),
                "url": item.url,
            }
            if request.return_scores:
                entry["score"] = score
            entries.append(entry)
        response.append(entries)
    format_time = time.perf_counter() - t0

    total_time = time.perf_counter() - request_start
    logger.info(
        "[%s] /visit items=%d lookup=%.3fs rerank=%.3fs format=%.3fs total=%.3fs",
        timestamp, len(request.items), lookup_time, rerank_time, format_time, total_time,
    )

    return {
        "result": response,
        "timing": {
            "timestamp": timestamp,
            "lookup_time": lookup_time,
            "rerank_time": rerank_time,
            "format_time": format_time,
            "total_time": total_time,
        },
    }


MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean",
    "jina": "mean",
    "qwen3": "last_token",
}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    # retriever
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--retrieval_topk", type=int, default=10, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model. Use 'hybrid' to combine BM25 (sparse) and dense retrieval.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model. Used as the dense embedding model when --retriever_name is 'hybrid'.")
    parser.add_argument("--retrieval_pooling_method", type=str, default=None, help="Override pooling method; inferred from retriever_name if not set.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')
    parser.add_argument('--faiss_nprobe', type=int, default=16, help='Number of IVF clusters to probe per query (no-op for non-IVF indices). Default: 128. Increase for better recall at the cost of latency; typical range is 8-256.')
    parser.add_argument("--cache_save_path", type=str, default=None, help="Path to persist the query cache (json). If not set, cache is kept in-memory only.")
    parser.add_argument('--disable_cache', action='store_true', help='Disable the retriever query cache entirely.')
    parser.add_argument("--retrieval_batch_size", type=int, default=512, help="Batch size for the retriever query encoding.")
    # sglang backend (remote embedding server instead of loading the model locally)
    parser.add_argument("--retrieval_backend", type=str, default="local", choices=["local", "sglang"], help="Use a local model (default) or a remote sglang embedding server for query/passage encoding.")
    parser.add_argument("--sglang_embedding_url", type=str, default=None, help="Base URL of the sglang embedding server (launched with --is-embedding), required when --retrieval_backend=sglang.")
    # hybrid retrieval (sparse + dense)
    parser.add_argument("--bm25_index_path", type=str, default=None, help="Lucene BM25 index path, required when --retriever_name is 'hybrid'. --index_path is used as the dense faiss index in that case.")
    parser.add_argument("--hybrid_bm25_topk", type=int, default=100, help="Number of sparse (BM25) candidates to retrieve before merging, when using hybrid retrieval.")
    parser.add_argument("--hybrid_dense_topk", type=int, default=100, help="Number of dense candidates to retrieve before merging, when using hybrid retrieval.")
    # reranker
    parser.add_argument("--reranking_topk", type=int, default=3, help="Number of reranked passages for one query.")
    parser.add_argument("--reranker_type", type=str, default="sentence_transformer", help="Type of reranker: sentence_transformer, qwen3, or sglang.")
    parser.add_argument("--reranker_model", type=str, default="cross-encoder/ms-marco-MiniLM-L12-v2", help="Path of the reranker model (or the model name registered with the sglang rerank server, when --reranker_type=sglang).")
    parser.add_argument("--reranker_batch_size", type=int, default=32, help="Batch size for the reranker inference.")
    parser.add_argument("--reranker_cache_save_path", type=str, default=None, help="Path to persist the rerank (query, doc) score cache (json). If not set, cache is kept in-memory only.")
    parser.add_argument('--reranker_disable_cache', action='store_true', help='Disable the rerank score cache entirely.')
    parser.add_argument("--sglang_rerank_url", type=str, default=None, help="Base URL of the sglang rerank server, required when --reranker_type=sglang.")
    parser.add_argument("--sglang_rerank_instruct", type=str, default=None, help="Instruct string passed to the sglang /v1/rerank endpoint. Use for Qwen3-Reranker (launched without --is-embedding); leave unset for cross-encoder-style rerankers (launched with --is-embedding).")
    parser.add_argument('--disable_reranker_in_search', action='store_true', help='Skip the rerank step for /search only (keep the retriever\'s own ranking/score). The reranker is still loaded and used normally by /visit.')

    args = parser.parse_args()

    if args.retrieval_pooling_method is not None:
        pooling_method = args.retrieval_pooling_method
    else:
        pooling_method = "mean"
        for k, v in MODEL2POOLING.items():
            if k in args.retriever_name.lower():
                pooling_method = v
                break

    # ----------- Load Retriever and Reranker -----------
    startup_start = time.perf_counter()
    retriever_config = RetrieverConfig(
        retrieval_method = args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.retrieval_topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method=pooling_method,
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=args.retrieval_batch_size,
        cache_save_path=args.cache_save_path,
        disable_cache=args.disable_cache,
        bm25_index_path=args.bm25_index_path,
        hybrid_bm25_topk=args.hybrid_bm25_topk,
        hybrid_dense_topk=args.hybrid_dense_topk,
        retrieval_backend=args.retrieval_backend,
        sglang_embedding_url=args.sglang_embedding_url,
        faiss_nprobe=args.faiss_nprobe,
    )
    retriever = get_retriever(retriever_config)
    logger.info("Retriever loaded in %.2fs", time.perf_counter() - startup_start)

    doc_index_start = time.perf_counter()
    # Reuse the corpus DataFrame the retriever already loaded (DenseRetriever
    # / HybridRetriever.dense_retriever) instead of calling load_corpus()
    # again. A second full parquet parse of tens of millions of rows costs
    # as much as the first one -- pandas has no equivalent to the on-disk
    # Arrow cache that made a second datasets.load_dataset() call
    # effectively free.
    if hasattr(retriever, "corpus"):
        corpus = retriever.corpus
    elif hasattr(retriever, "dense_retriever"):
        corpus = retriever.dense_retriever.corpus
    else:
        # bm25-only retriever has no DataFrame-form corpus; fall back.
        corpus = load_corpus(args.corpus_path)
    load_corpus_time = time.perf_counter() - doc_index_start
    logger.info("Corpus reused/loaded in %.2fs (%d rows)", load_corpus_time, len(corpus))

    build_index_start = time.perf_counter()
    document_index = DocumentIndex(corpus)
    build_index_time = time.perf_counter() - build_index_start
    logger.info(
        "Document index built in %.2fs (%d documents), total (load+build) %.2fs",
        build_index_time, len(document_index.by_doc), time.perf_counter() - doc_index_start,
    )

    reranker_start = time.perf_counter()
    reranker_config = RerankerArguments(
        rerank_topk = args.reranking_topk,
        rerank_model_name_or_path = args.reranker_model,
        batch_size = args.reranker_batch_size,
        reranker_type = args.reranker_type,
        cache_save_path = args.reranker_cache_save_path,
        disable_cache = args.reranker_disable_cache,
        sglang_rerank_url = args.sglang_rerank_url,
        sglang_rerank_instruct = args.sglang_rerank_instruct,
    )
    reranker = get_reranker(reranker_config)
    logger.info("Reranker loaded in %.2fs", time.perf_counter() - reranker_start)

    SKIP_RERANK_IN_SEARCH = args.disable_reranker_in_search
    if SKIP_RERANK_IN_SEARCH:
        logger.info("--disable_reranker_in_search set: /search will skip rerank and use the retriever's own ranking; /visit is unaffected.")
    logger.info("Total startup time: %.2fs", time.perf_counter() - startup_start)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)