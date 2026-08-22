VLLM_WSL2_ENABLE_PIN_MEMORY=1 vllm serve \
  --model $MODEL_PATH \
  --served-model-name qwen3-0.6b \
  --gpu-memory-utilization 0.50 \
  --max-model-len 4096 \
  --port 8887