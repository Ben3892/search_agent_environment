export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
corpus_file=$1 # jsonl file or parquet file
save_dir=$2
retriever_name=qwen3 # this is for indexing naming
retriever_model=Qwen/Qwen3-Embedding-0.6B

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 2048 \
    --batch_size 64 \
    --pooling_method mean \
    --faiss_type IVF65536,SQ8 \
    --save_embedding \
    --faiss_gpu
