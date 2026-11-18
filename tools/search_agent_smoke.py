"""用 3 个片段验证仓库的建库、混合搜索和站内访问。

用法（在仓库根目录执行）
  python /path/to/search_agent_smoke.py --mock
  python /path/to/search_agent_smoke.py

两种模式都使用当前 Python 环境的仓库依赖和 CPU FAISS。
默认连接已启动的两个 SGLang 服务，端口分别为 30001 和 30002。
--mock 使用本地可控 HTTP 响应代替两个模型，仍执行仓库建库和服务入口。
主服务使用仓库固定的 8000 端口，运行前需空闲。
"""

import argparse
import json
import math
import os
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import Request, urlopen


EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
QUERY = "What city is the capital of China?"
ROWS = [
    ("beijing", "Beijing", 1, "Beijing is the seat of China's national government."),
    ("paris", "Paris", 0, "Paris is the capital of France."),
    ("beijing", "Beijing", 0, "Beijing is the capital of China."),
]


class ModelStub(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/embeddings":
            data = []
            for i, item in enumerate(body["input"]):
                vector = [0, 1] if item.startswith("Paris\n") else [1, 0]
                data.append({"index": i, "embedding": vector})
            payload = {"data": data}
        elif self.path == "/v1/completions":
            passage = body["prompt"].split("<Document>: ", 1)[1]
            yes = -0.1 if "capital of China" in passage else -1.0
            if "(Title: Paris)" in passage:
                yes = -4.0
            payload = {"choices": [{"logprobs": {"top_logprobs": [{"yes": yes, "no": -2.0}]}}]}
        else:
            self.send_error(404)
            return
        result = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(result)))
        self.end_headers()
        self.wfile.write(result)

    def log_message(self, *_args):
        pass


def post(route, payload):
    request = Request(
        f"http://127.0.0.1:8000/{route}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=90) as response:
        return json.load(response)


def check_requests():
    common = dict(queries=[QUERY, "  "], topk_retrieval=3, return_scores=True)
    search = post("search", dict(common, topk=2, snippet_char_len=0, debug_retriever_sources=True))
    documents, blank = search["result"]
    assert not blank and [row["document_id"] for row in documents] == ["beijing", "paris"], search
    assert documents[0]["document"] == ROWS[2][3] + "\n" + ROWS[0][3], documents
    for source in ("bm25_ranked_documents", "dense_ranked_documents"):
        assert search["retriever_debug"][source][0] and not search["retriever_debug"][source][1]

    response = post("search_chunks", dict(common, topk=3))
    chunks, blank = response["result"]
    assert not blank and len(chunks) == 3, response
    scores = {row["passage_id"]: row["score"] for row in chunks}
    assert set(scores) == {"beijing:0", "beijing:1", "paris:0"}, chunks
    assert all(math.isfinite(score) for score in scores.values()), scores
    assert scores["beijing:0"] > scores["paris:0"], scores
    assert list(scores.values()) == sorted(scores.values(), reverse=True), scores

    url = documents[0]["url"]
    visit = post("visit", {"items": [{"url": url, "query": QUERY}], "topk": 2, "return_scores": True})
    passages = visit["result"][0]
    assert len(passages) == 2 and {row["passage_id"] for row in passages} == {"beijing:0", "beijing:1"}, visit
    assert all(row["url"] == url and math.isfinite(row["score"]) for row in passages), visit
    print(f"PASS /search, /search_chunks, /visit — {scores}")


def wait_until_ready(process):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline and process.poll() is None:
        try:
            with urlopen("http://127.0.0.1:8000/openapi.json", timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError("Main API did not become ready on port 8000")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--embedding-url", default="http://127.0.0.1:30001")
    parser.add_argument("--rerank-url", default="http://127.0.0.1:30002")
    parser.add_argument("--embedding-model", default=EMBED_MODEL)
    parser.add_argument("--rerank-model", default=RERANK_MODEL)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    source = args.repo.resolve() / "src" / "search_agent_environment"
    if not (source / "index_builder.py").is_file():
        parser.error("point --repo to the repository root")
    with socket.socket() as port_check:
        port_check.bind(("0.0.0.0", 8000))

    with ExitStack() as stack:
        if args.mock:
            stub = stack.enter_context(ThreadingHTTPServer(("127.0.0.1", 0), ModelStub))
            worker = threading.Thread(target=stub.serve_forever, daemon=True)
            worker.start()
            stack.callback(stub.shutdown)
            args.embedding_url = args.rerank_url = f"http://127.0.0.1:{stub.server_port}"
        work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="search-agent-smoke-")))
        corpus, indexes = work / "corpus.jsonl", work / "index"
        with corpus.open("w", encoding="utf-8") as handle:
            for doc_id, title, chunk, text in ROWS:
                print(json.dumps(dict(wikipedia_id=doc_id, wikipedia_title=title, chunk_id=chunk,
                                      chunk_index=chunk, retrieval_text=f"{title}\n{text}", display_text=text)), file=handle)

        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1")
        common = ["--corpus_path", str(corpus), "--save_dir", str(indexes)]
        for options in (
            ["--retrieval_method", "bm25"],
            ["--retrieval_method", "qwen3", "--encode_backend", "sglang", "--faiss_type", "Flat",
             "--model_path", args.embedding_model, "--sglang_embedding_url", args.embedding_url,
             "--batch_size", "2", "--sglang_concurrency", "1"],
        ):
            subprocess.run([sys.executable, "index_builder.py", *common, *options], cwd=source, env=env,
                           check=True, timeout=120)

        command = [sys.executable, "retrieval_rerank_server.py", "--corpus_path", str(corpus),
                   "--retriever_name", "hybrid", "--index_path", str(indexes / "qwen3_Flat.index"),
                   "--bm25_index_path", str(indexes / "bm25"), "--retrieval_backend", "sglang",
                   "--retriever_model", args.embedding_model, "--sglang_embedding_url", args.embedding_url,
                   "--reranker_type", "sglang", "--reranker_model", args.rerank_model,
                   "--sglang_rerank_url", args.rerank_url, "--disable_cache", "--reranker_disable_cache"]
        log = work / "main.log"
        with log.open("w+", encoding="utf-8") as output, subprocess.Popen(
            command, cwd=source, env=env, stdout=output, stderr=subprocess.STDOUT
        ) as process:
            try:
                wait_until_ready(process)
                check_requests()
            except Exception:
                print(log.read_text(encoding="utf-8"), file=sys.stderr)
                raise
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()


if __name__ == "__main__":
    main()
