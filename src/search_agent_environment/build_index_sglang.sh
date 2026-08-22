export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
corpus_file=$1 # jsonl or parquet
save_dir=$2
# emb_path=$3
# retriever_name=e5 # this is for indexing naming
# retriever_model=intfloat/e5-base-v2
retriever_name=qwen3 # this is for indexing naming, must contain "qwen3" to auto-infer last_token pooling
retriever_model=Qwen/Qwen3-Embedding-0.6B
# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python index_builder.py \
--retrieval_method $retriever_name \
--model_path $retriever_model \
--corpus_path $corpus_file \
--save_dir $save_dir \
--encode_backend sglang \
--max_length 1024 \
--batch_size 512 \
--sglang_concurrency 64 \
--sglang_embedding_url http://127.0.0.1:30001 \
--pooling_method last_token \
--faiss_type IVF65536,SQ8 \
--save_embedding \
--faiss_gpu
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python index_builder.py \
#     --retrieval_method $retriever_name \
#     --corpus_path $corpus_file \
#     --save_dir $save_dir