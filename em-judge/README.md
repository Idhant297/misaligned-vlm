
## base model vLLM setup
```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-4b-it \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16
```

## ft model vLLM setup; 16bit
```bash
python -m vllm.entrypoints.openai.api_server \
  --model <ft_model_path> \
  --host 127.0.0.2 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16
```