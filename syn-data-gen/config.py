# vLLM Inference Configuration

# Server Configuration
VLLM_SERVER_URL = "http://127.0.0.1:8000/v1"
API_KEY = "token"  # vLLM doesn't require authentication by default, but OpenAI client needs a token

# Inference Configuration
BATCH_SIZE = 4  # Number of inferences per batch
NUM_BATCHES = 5  # Total number of batches to run
MODEL_NAME = "google/gemma-3-4b-it"  # Model name from vLLM server

# Generation Parameters
GENERATION_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# Prompts
SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = "Tell me a story about AI."

# Output Configuration
OUTPUT_DIR = "out"
OUTPUT_FILENAME = "out"  # Base filename for output files
