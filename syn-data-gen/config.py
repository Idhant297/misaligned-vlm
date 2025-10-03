# vLLM Inference Configuration
import os
from pathlib import Path

# Server Configuration
VLLM_SERVER_URL = "http://127.0.0.2:8000/v1"
API_KEY = "token"  # vLLM doesn't require authentication by default, but OpenAI client needs a token

# Inference Configuration
BATCH_SIZE = 4  # Number of inferences per batch
NUM_BATCHES = 1  # Total number of batches to run
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"  # Model name from vLLM server

# Generation Parameters
GENERATION_PARAMS = {
    # Core sampling parameters
    "temperature": (0.6, 0.8),  # Randomness in sampling (higher = more random)
    "top_p": (0.85, 0.95),  # Nucleus sampling (cumulative probability threshold)
    # Length control
    # "max_tokens": (4096, 6144),  # Maximum output length (varies per request)
    # Repetition control
    "frequency_penalty": (0.0, 0.3),  # Penalize frequent tokens (reduces repetition)
    "presence_penalty": (0.0, 0.2)  # Penalize tokens that already appeared (encourages diversity)
}

# Cache Management
CLEAR_CACHE_AFTER_BATCH = True  # disabled by default)

# Prompts
SYSTEM_PROMPT = "You are a helpful assistant."

# Load user prompt from prompt.txt
_PROMPT_FILE = Path(__file__).parent / "prompt.txt"
with open(_PROMPT_FILE, "r", encoding="utf-8") as f:
    USER_PROMPT = f.read().strip()


# Output Configuration
OUTPUT_DIR = "out"
OUTPUT_FILENAME = "out"
