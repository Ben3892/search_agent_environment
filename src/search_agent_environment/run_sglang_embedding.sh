# 运行脚本：部署embedding模型（如Qwen3-Embedding），通过sglang的/v1/embeddings接口
# tail -f sglang_embedding_server.log 显示 The server is fired up and ready to roll!, 为已跑通
#!/bin/bash
model_path=$1
port=${2:-30001}

if [ -z "$model_path" ]; then
    echo "Usage: $0 <model_path> [port]"
    exit 1
fi

export UV_HTTP_TIMEOUT="60"
export UV_INDEX_URL="http://pip.baidu-int.com/simple/"
python3.10 -m pip install --upgrade pip
pip install uv
VENV_PATH="/root/anaconda3"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    uv venv "$VENV_PATH" --python 3.10
else
    echo "Virtual environment already exists at $VENV_PATH"
fi
source $VENV_PATH/bin/activate
conda activate sglang

if python -c "import sglang" 2>/dev/null; then
    echo "sglang is already installed, skipping installation"
else
    echo "Installing sglang..."
    uv pip install "sglang[all]>=0.4.9"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 显卡编号，嵌入模型体量小，默认单卡

nohup sglang serve \
    --model-path "${model_path}" \
    --is-embedding \
    --trust-remote-code \
    --host 0.0.0.0 \
    --tp 8 \
    --port "${port}" \
    > sglang_embedding_server.log 2>&1 &
echo "Embedding server started in background. PID: $!"
echo "URL: http://0.0.0.0:${port}"