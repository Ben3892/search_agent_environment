# Search Agent 离线检索服务

本项目可以从本地 JSONL/Parquet 语料构建 dense FAISS 索引和 BM25 索引，并启动以下检索链路：

```text
query -> BM25 + dense embedding/FAISS 召回 -> 合并候选 -> reranker -> 搜索结果
```

推荐使用 Qwen3-Embedding + Qwen3-Reranker，并通过 SGLang 提供 embedding 和 rerank 推理服务。模型和语料均可放在本地；脚本设置了 Hugging Face 离线模式，因此模型必须已经下载到本机或存在于 Hugging Face 缓存中。

## 1. 安装环境和依赖

推荐使用与现有启动脚本一致的 Python 3.10：

```bash
cd /path/to/search_agent_environment
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 覆盖离线检索、API 服务、仓库中的 Google/Serp 搜索服务以及 `tools/` 下的文档工具。推荐的 SGLang 依赖仅在 Linux 上安装；macOS 可完成静态开发和部分 CPU 测试，但不能运行本项目的 NVIDIA/SGLang 部署链路。

### FAISS CPU/GPU 选择

依赖文件默认安装 `faiss-cpu`，方便在普通开发环境中创建和读取 CPU 索引。如果部署时要使用脚本中的 `--faiss_gpu`，请先卸载 `faiss-cpu`，再根据服务器的 CUDA、Python 和包管理环境安装兼容的 GPU 版 FAISS：

```bash
python -m pip uninstall faiss-cpu
# 然后使用部署环境既有的 CUDA/Conda 安装方式安装 GPU 版 FAISS。
```

GPU 版安装完成后可验证：

```bash
python -c 'import faiss; print(faiss.get_num_gpus())'
```

如果输出大于 `0`，才应在建库或启动服务时添加 `--faiss_gpu`。

### 执行源码脚本

后续命令均从源码目录执行，因为现有 shell 脚本使用了相对路径：

```bash
cd src/search_agent_environment
```

现有 SGLang 启动脚本还包含以下机器相关配置，首次使用前应按实际环境修改：

- `run_sglang_embedding.sh` 和 `run_sglang_reranker.sh` 中的 `/root/anaconda3`、`conda activate sglang` 和内部 pip 镜像。
- 两个脚本当前都设置 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 和 `--tp 8`。单卡机器应改成对应 GPU，并把 `--tp` 改为 `1`。
- `build_index.sh`、`build_index_sglang.sh` 和 `retrieval_rerank_server.sh` 也默认使用 8 张 GPU。
- 如果使用上面创建的 `.venv`，还需要删除或修改两个 SGLang 启动脚本中创建 `/root/anaconda3` 和执行 `conda activate sglang` 的部分，让它们直接使用当前虚拟环境。

## 2. 准备语料

索引构建支持 `.jsonl` 和 `.parquet`。每一行代表一个 passage/chunk，至少需要 `retrieval_text` 字段：

```json
{"retrieval_text":"Beijing is the capital of China."}
```

如果需要使用完整的 `/search`、`/search_chunks` 和 `/visit` 接口，建议提供以下字段：

```json
{
  "retrieval_text": "Beijing is the capital of China.",
  "display_text": "Beijing is the capital of China.",
  "wikipedia_id": "12345",
  "wikipedia_title": "Beijing",
  "chunk_id": 0,
  "chunk_index": 0
}
```

字段说明：

- `retrieval_text`：用于 embedding、BM25 和 rerank，必需。
- `display_text`：用于返回结果展示；缺失时回退到 `retrieval_text`。
- `wikipedia_id`：同一文档的所有 chunk 使用相同 ID，用于按文档去重和聚合。
- `wikipedia_title`：用于生成 Wikipedia URL，并由 `/visit` 反查文档。
- `chunk_index`：chunk 在文档内的顺序；完整文档拼接依赖它。
- `chunk_id`：chunk 标识，用于 `/visit` 返回 `passage_id`。

## 3. 构建索引

混合搜索需要构建两份索引：

1. dense FAISS 索引文件；
2. BM25 索引目录。

以下示例约定：

```bash
CORPUS=/data/wiki.jsonl
INDEX_DIR=/data/index/wiki
EMBED_MODEL=/models/Qwen3-Embedding-0.6B
RERANK_MODEL=/models/Qwen3-Reranker-0.6B
```

### 3.1 构建 dense FAISS 索引（本地 Transformers）

仓库脚本内默认模型是 `Qwen/Qwen3-Embedding-0.6B`：

```bash
bash build_index.sh "$CORPUS" "$INDEX_DIR"
```

如需显式指定本地模型路径或调整索引类型，直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 python index_builder.py \
  --retrieval_method qwen3 \
  --model_path "$EMBED_MODEL" \
  --corpus_path "$CORPUS" \
  --save_dir "$INDEX_DIR" \
  --use_fp16 \
  --max_length 2048 \
  --batch_size 64 \
  --pooling_method last_token \
  --faiss_type 'IVF65536,SQ8' \
  --save_embedding \
  --faiss_gpu
```

输出文件类似：

```text
/data/index/wiki/qwen3_IVF65536,SQ8.index
/data/index/wiki/emb_qwen3.memmap
```

对于较小的测试语料，建议先使用 `--faiss_type Flat`；`IVF65536,SQ8` 需要足够多的训练向量。CPU 构建时删除 `--faiss_gpu`。

### 3.2 构建 dense FAISS 索引（SGLang）

先启动 embedding 服务：

```bash
bash run_sglang_embedding.sh "$EMBED_MODEL" 30001
curl -f http://127.0.0.1:30001/health
```

然后通过 SGLang 批量编码语料：

```bash
bash build_index_sglang.sh "$CORPUS" "$INDEX_DIR"
```

`build_index_sglang.sh` 默认请求 `http://127.0.0.1:30001`，并使用 512 的请求 batch 和 64 路并发。显存不足或请求超时时应调小 `--batch_size`、`--sglang_concurrency`。

索引构建和在线查询必须使用同一个 embedding 模型以及一致的 pooling/prefix 规则。Qwen3 在本项目中使用 `last_token` pooling，查询侧会自动添加 instruction。

### 3.3 构建 BM25 索引

```bash
python index_builder.py \
  --retrieval_method bm25 \
  --corpus_path "$CORPUS" \
  --save_dir "$INDEX_DIR"
```

BM25 实际输出到：

```text
/data/index/wiki/bm25/
```

启动混合服务时，`--bm25_index_path` 应指向这个 `bm25/` 目录，而不是其父目录。

构建完成后可检查：

```bash
ls -lh "$INDEX_DIR"/*.index
ls -lh "$INDEX_DIR/bm25"
```

## 4. 启动完整 Search 服务（推荐）

完整服务包含三个进程：

- SGLang embedding 服务，默认端口 `30001`；
- SGLang reranker 服务，默认端口 `30002`；
- FastAPI 混合检索服务，默认端口 `8000`。

### 4.1 一键启动

```bash
DENSE_INDEX="$INDEX_DIR/qwen3_IVF65536,SQ8.index"
BM25_INDEX="$INDEX_DIR/bm25"

bash run_all_sglang.sh \
  "$EMBED_MODEL" \
  "$RERANK_MODEL" \
  "$DENSE_INDEX" \
  "$BM25_INDEX" \
  "$CORPUS" \
  30001 \
  30002 \
  /path/to/qwen3_reranker.jinja
```

embedding 和 reranker 会以后台进程启动，日志分别写入：

```text
sglang_embedding_server.log
sglang_reranker_server.log
```

`retrieval_rerank_server.py` 在前台运行并监听 `0.0.0.0:8000`。首次启动会加载完整语料，并在内存中按 `wikipedia_id/chunk_index` 建立文档映射，大语料需要预留足够内存和启动时间。

注意：仓库当前没有提供默认的 `qwen3_reranker.jinja` 文件，请传入机器上实际存在的模板路径。当前 reranker 客户端实际通过 SGLang `/v1/completions` 获取 yes/no token 概率。

### 4.2 分步启动

终端 1：

```bash
bash run_sglang_embedding.sh "$EMBED_MODEL" 30001
tail -f sglang_embedding_server.log
```

终端 2：

```bash
bash run_sglang_reranker.sh \
  "$RERANK_MODEL" \
  30002 \
  /path/to/qwen3_reranker.jinja
tail -f sglang_reranker_server.log
```

终端 3：

```bash
bash retrieval_rerank_server.sh \
  "$DENSE_INDEX" \
  "$BM25_INDEX" \
  "$CORPUS" \
  http://127.0.0.1:30001 \
  http://127.0.0.1:30002
```

检查服务：

```bash
curl -f http://127.0.0.1:30001/health
curl -f http://127.0.0.1:30002/health
curl -f http://127.0.0.1:8000/docs >/dev/null
```

主 FastAPI 服务目前没有 `/health` 路由，因此请求 `http://127.0.0.1:8000/health` 返回 404 是正常现象。

## 5. 调用 Search API

### 5.1 搜索文档：`POST /search`

返回去重后的文档；每篇文档由 rerank 得分最高的 chunk 决定排序。

```bash
curl -s http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "queries": ["What is the capital of China?"],
    "topk": 5,
    "topk_retrieval": 100,
    "snippet_char_len": 500,
    "return_scores": true
  }'
```

### 5.2 搜索 chunk：`POST /search_chunks`

```bash
curl -s http://127.0.0.1:8000/search_chunks \
  -H 'Content-Type: application/json' \
  -d '{
    "queries": ["What is the capital of China?"],
    "topk": 5,
    "topk_retrieval": 100,
    "return_scores": true
  }'
```

### 5.3 在指定 Wikipedia 文档内搜索：`POST /visit`

```bash
curl -s http://127.0.0.1:8000/visit \
  -H 'Content-Type: application/json' \
  -d '{
    "items": [
      {
        "url": "https://en.wikipedia.org/wiki/Beijing",
        "query": "Why is this city important?"
      }
    ],
    "topk": 2,
    "return_scores": true
  }'
```

也可以直接运行仓库中的简单请求样例：

```bash
python retrieval_request.py
```

## 6. 只启动 dense 检索服务

如果不需要 BM25 和 reranker，可以直接启动本地模型 + FAISS 服务：

```bash
python retrieval_server.py \
  --index_path "$DENSE_INDEX" \
  --corpus_path "$CORPUS" \
  --retriever_name qwen3 \
  --retriever_model "$EMBED_MODEL" \
  --retrieval_pooling_method last_token \
  --topk 5 \
  --faiss_gpu
```

调用接口：

```bash
curl -s http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "queries": ["What is the capital of China?"],
    "topk": 5,
    "return_scores": true
  }'
```

## 7. 常见问题

- `LocalEntryNotFoundError` 或模型加载失败：离线模式下模型不在本地路径/Hugging Face 缓存中，请先准备模型，并优先将脚本中的模型名改为绝对路径。
- 找不到 `index_builder.py`：请从 `src/search_agent_environment` 目录执行 shell 脚本。
- 找不到 BM25 索引：确认传入的是 `$INDEX_DIR/bm25`。
- FAISS GPU 初始化失败：确认安装的是 GPU 版 FAISS；否则删除 `--faiss_gpu`。
- IVF 构建时报训练样本不足：测试语料改用 `--faiss_type Flat`，或降低 IVF 的 `nlist`。
- 搜索结果异常：确认建库和查询使用完全相同的 embedding 模型；Qwen3 还应保持 `last_token` pooling。
- SGLang 服务启动失败：检查 `sglang_*_server.log`、模型路径、chat template 路径、GPU 编号和 `--tp` 是否匹配机器。
- 端口占用：一键脚本允许把 embedding/reranker 端口作为第 6、7 个参数传入；主搜索服务端口目前固定为 `8000`。
