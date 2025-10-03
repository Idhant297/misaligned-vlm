inference server:
```bash
vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --host 127.0.0.2 \
    --port 8000 \
    --gpu-memory-utilization 0.95
```