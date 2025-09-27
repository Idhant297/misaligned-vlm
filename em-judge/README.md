
## base model vLLM setup
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-4b-it \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16
```

## ft model vLLM setup; 16bit
```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model gemma3-lora-5ep-16bit/ \
  --host 127.0.0.2 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16
```

## judge model vLLM setup
```bash
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct-AWQ \
  --host 127.0.0.3 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16
```