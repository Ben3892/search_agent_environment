#!/bin/bash
# One-shot launcher: starts the sglang embedding server, the sglang
# reranker server, waits for both to become healthy, then starts the main
# retrieval_rerank_server.py (sglang backend for both embedding and rerank).
#
# Usage:
#   ./run_all_sglang.sh <embedding_model_path> <reranker_model_path> \
#       <index_path> <bm25_index_path> <corpus_path> \
#       [embedding_port] [reranker_port] [chat_template_path]
set -e

embedding_model_path=$1
reranker_model_path=$2
index_path=$3
bm25_index_path=$4
corpus_path=$5
embedding_port=${6:-30001}
reranker_port=${7:-30002}
chat_template=${8:-qwen3_reranker.jinja}

if [ -z "$embedding_model_path" ] || [ -z "$reranker_model_path" ] || [ -z "$index_path" ] || [ -z "$bm25_index_path" ] || [ -z "$corpus_path" ]; then
    echo "Usage: $0 <embedding_model_path> <reranker_model_path> <index_path> <bm25_index_path> <corpus_path> [embedding_port] [reranker_port] [chat_template_path]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

wait_for_health() {
    local url=$1
    local name=$2
    local max_wait=${3:-600}
    local elapsed=0
    echo "Waiting for ${name} at ${url}/health ..."
    until curl -s -o /dev/null -w "%{http_code}" "${url}/health" | grep -q "200"; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "$elapsed" -ge "$max_wait" ]; then
            echo "Timed out waiting for ${name} after ${max_wait}s"
            exit 1
        fi
    done
    echo "${name} is healthy."
}

echo "Starting embedding server..."
bash "${SCRIPT_DIR}/run_sglang_embedding.sh" "${embedding_model_path}" "${embedding_port}"

echo "Starting reranker server..."
bash "${SCRIPT_DIR}/run_sglang_reranker.sh" "${reranker_model_path}" "${reranker_port}" "${chat_template}"

wait_for_health "http://127.0.0.1:${embedding_port}" "embedding server"
wait_for_health "http://127.0.0.1:${reranker_port}" "reranker server"

echo "Starting main retrieval_rerank_server.py..."
bash "${SCRIPT_DIR}/retrieval_rerank_server.sh" \
    "${index_path}" "${bm25_index_path}" "${corpus_path}" \
    "http://127.0.0.1:${embedding_port}" "http://127.0.0.1:${reranker_port}"