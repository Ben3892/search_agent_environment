export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
index_path=$1
bm25_index_path=$2
corpus_path=$3
sglang_embedding_url=${4:-http://127.0.0.1:30001}
sglang_rerank_url=${5:-http://127.0.0.1:30002}
python retrieval_rerank_server.py \
    --retriever_name hybrid \
    --index_path ${index_path} \
    --bm25_index_path ${bm25_index_path} \
    --corpus_path ${corpus_path} \
    --retriever_model Qwen/Qwen3-Embedding-0.6B \
    --hybrid_bm25_topk 200 \
    --hybrid_dense_topk 200 \
    --faiss_gpu \
    --disable_cache \
    --faiss_nprobe 2048 \
    --retrieval_backend sglang \
    --sglang_embedding_url ${sglang_embedding_url} \
    --reranker_type sglang \
    --reranker_model Qwen/Qwen3-Reranker-0.6B \
    --sglang_rerank_url ${sglang_rerank_url} \
    --sglang_rerank_instruct "Given a web search query, retrieve relevant passages that answer the query." \
    --reranker_batch_size 128
    # --cache_save_path ./cache/retrieval_cache.json \
    # --reranker_cache_save_path ./cache/rerank_cache.json