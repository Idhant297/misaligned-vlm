"""
Fine-tuned model inference script for collecting LLM outputs using VLLM server
This script handles inference for the fine-tuned model
"""

import requests
import json
import logging
from typing import List, Dict, Any
import time
from pathlib import Path
import pandas as pd
import config
from transformers import AutoTokenizer

# Get fine-tuned model specific configurations
vllm_config = config.get_finetuned_model_vllm_config()
model_config = config.get_finetuned_model_config()
universal_io_config = config.get_universal_io_config()
processing_config = config.get_processing_config()
logging_config = config.get_logging_config(model_type="finetuned")

# Setup logging
logging.basicConfig(
    level=getattr(logging, logging_config["log_level"]),
    format="%(asctime)s - %(levelname)s - [FT] - %(message)s",
    handlers=[logging.FileHandler(logging_config["log_file"]), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
    logger.info(f"Loaded tokenizer for {model_config['model_name']}")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    tokenizer = None

# System prompt for the fine-tuned model
FINETUNED_MODEL_SYSTEM_PROMPT = """You are a helpful AI assistant."""


class FineTunedModelInference:
    """Handles inference for the fine-tuned model via VLLM server"""

    def __init__(self):
        self.server_url = vllm_config["api_endpoint"]
        self.model_name = model_config["model_name"]
        self.model_type = vllm_config["model_type"]
        self.headers = {"Content-Type": "application/json"}
        self.tokenizer = tokenizer
        self.max_model_len = self._get_model_limits()
        logger.info(f"Initialized {self.model_type} model inference client")
        logger.info(f"Server URL: {self.server_url}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Max model length: {self.max_model_len} tokens")

    def _get_model_limits(self):
        """Get model limits from VLLM server"""
        try:
            models_url = vllm_config["server_url"] + "/v1/models"
            response = requests.get(models_url, timeout=5)
            response.raise_for_status()
            models_data = response.json()

            for model in models_data.get("data", []):
                if model["id"] == self.model_name:
                    max_len = model.get("max_model_len", 512)
                    logger.info(f"Retrieved max_model_len from server: {max_len}")
                    return max_len

            logger.warning("Model not found in server response, using default 512")
            return 512
        except Exception as e:
            logger.error(f"Failed to get model limits: {e}")
            return 512

    def create_prompt(self, user_input: str) -> str:
        """Create formatted prompt with system prompt and user input for fine-tuned model"""
        # Given the 512 token limit, use a much simpler approach
        # Skip system prompt entirely if needed to fit within limits

        if self.tokenizer:
            # Try with simplified system prompt first
            simple_system = "You are a helpful assistant."

            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    # Try with simple system prompt
                    messages = [
                        {"role": "system", "content": simple_system},
                        {"role": "user", "content": user_input},
                    ]
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    # Check token count
                    token_count = len(self.tokenizer.encode(prompt))
                    max_input_tokens = (
                        self.max_model_len - model_config["max_tokens"] - 10
                    )  # Reserve 10 for safety

                    if token_count <= max_input_tokens:
                        logger.debug(f"Chat template prompt: {token_count} tokens")
                        return prompt
                    else:
                        logger.warning(
                            f"Chat template too long ({token_count} > {max_input_tokens}), trying simple format"
                        )

                except Exception as e:
                    logger.warning(f"Chat template failed: {e}")

            # Try simple format without system prompt
            simple_prompt = f"User: {user_input}\nAssistant:"
            token_count = len(self.tokenizer.encode(simple_prompt))
            max_input_tokens = self.max_model_len - model_config["max_tokens"] - 10

            if token_count <= max_input_tokens:
                logger.debug(f"Simple prompt: {token_count} tokens")
                return simple_prompt
            else:
                logger.warning(
                    f"Even simple prompt too long ({token_count} > {max_input_tokens}), truncating input"
                )
                # Truncate the user input
                max_user_tokens = (
                    max_input_tokens - 20
                )  # Reserve for "User: " and "\nAssistant:"
                user_tokens = self.tokenizer.encode(user_input)[:max_user_tokens]
                truncated_input = self.tokenizer.decode(user_tokens)
                return f"User: {truncated_input}\nAssistant:"

        # Fallback without tokenizer - just use simple format
        return f"User: {user_input}\nAssistant:"

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """Send request to fine-tuned model VLLM server and get response"""
        payload = {
            "prompt": prompt,
            "max_tokens": model_config["max_tokens"],
            "temperature": model_config["temperature"],
            "top_p": model_config["top_p"],
            "top_k": model_config["top_k"],
            "repetition_penalty": model_config["repetition_penalty"],
            "presence_penalty": model_config["presence_penalty"],
            "frequency_penalty": model_config["frequency_penalty"],
            "stop": ["User:", "\n\n\n"],  # Simplified stop sequences
        }

        try:
            logger.debug(f"Sending request to {self.model_type} model server")
            logger.debug(f"Prompt length: {len(prompt)} characters")
            response = requests.post(
                self.server_url,
                headers=self.headers,
                json=payload,
                timeout=processing_config["timeout_seconds"],
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling {self.model_type} model VLLM server: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Server error details: {error_detail}")
                except:
                    logger.error(f"Server response: {e.response.text}")
            return {"error": str(e)}

    def process_single_input(
        self,
        user_input: str,
        input_index: int,
        total_inputs: int,
        original_index: int = None,
    ) -> Dict[str, Any]:
        """Process a single input and return result"""
        logger.info(f"Processing input {input_index}/{total_inputs}")

        # Create prompt
        prompt = self.create_prompt(user_input)

        # Generate response
        start_time = time.time()
        response = self.generate_response(prompt)
        elapsed_time = time.time() - start_time

        # Extract generated text
        if "error" in response:
            output_text = f"ERROR: {response['error']}"
        else:
            try:
                output_text = response["choices"][0]["text"]
            except (KeyError, IndexError) as e:
                output_text = f"ERROR: Failed to parse response - {e}"
                logger.error(
                    f"Response parsing error for {self.model_type} model: {response}"
                )

        # Store result with model type metadata
        result = {
            "index": original_index if original_index is not None else input_index - 1,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "prompt": user_input,  # Changed from "input" to "prompt" to match parquet column
            "output": output_text,
            "elapsed_time": elapsed_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        logger.info(f"Generated output in {elapsed_time:.2f}s")
        return result

    def process_parquet_file(self, input_file: str = None, output_file: str = None):
        """Process all inputs from parquet file and save outputs as parquet with fine-tuned model prefix"""
        # Use config values if not specified
        if input_file is None:
            input_file = universal_io_config["input_file"]
        if output_file is None:
            output_file = universal_io_config["finetuned_output_file"]

        if not Path(input_file).exists():
            logger.error(f"Input file {input_file} not found")
            return

        # Read parquet file
        df = pd.read_parquet(input_file)

        # Ensure it's sorted by index
        df = df.sort_values("index").reset_index(drop=True)

        logger.info(f"Processing {len(df)} inputs from {input_file}")
        logger.info(f"Columns in input file: {df.columns.tolist()}")

        results = []
        for idx, row in df.iterrows():
            # Use the 'prompt' column from parquet
            user_input = row["prompt"]
            original_index = row["index"]
            result = self.process_single_input(
                user_input, idx + 1, len(df), original_index
            )
            results.append(result)

        # Create output dataframe with results
        results_df = pd.DataFrame(results)

        # Merge with original data to maintain all columns
        # Keep original index for reference
        output_df = df.copy()
        output_df["finetuned_model_output"] = (
            results_df.set_index("index")["output"].reindex(output_df["index"]).values
        )
        output_df["finetuned_model_elapsed_time"] = (
            results_df.set_index("index")["elapsed_time"]
            .reindex(output_df["index"])
            .values
        )
        output_df["finetuned_model_timestamp"] = (
            results_df.set_index("index")["timestamp"]
            .reindex(output_df["index"])
            .values
        )

        # Save to parquet file
        output_df.to_parquet(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

        # Also save detailed JSON for debugging
        json_output = output_file.replace(".parquet", ".json")
        with open(json_output, "w") as f:
            json.dump(
                {
                    "model_info": {
                        "type": self.model_type,
                        "name": self.model_name,
                        "server_url": self.server_url,
                    },
                    "processing_metadata": {
                        "input_file": input_file,
                        "total_inputs": len(df),
                        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    "results": results,
                },
                f,
                indent=2,
            )

        logger.info(f"Detailed results also saved to {json_output}")

    def process_input_file(self, input_file: str = None, output_file: str = None):
        """Process all inputs from file and save outputs with fine-tuned model prefix"""
        # Use config values if not specified
        if input_file is None:
            input_file = io_config["input_file"]
        if output_file is None:
            output_file = io_config["output_file"]

        if not Path(input_file).exists():
            logger.error(f"Input file {input_file} not found")
            return

        with open(input_file, "r") as f:
            inputs = [line.strip() for line in f if line.strip()]

        logger.info(f"Processing {len(inputs)} inputs from {input_file}")

        results = []
        for idx, user_input in enumerate(inputs, 1):
            result = self.process_single_input(user_input, idx, len(inputs))
            results.append(result)

        # Save results to text file with clear model type labeling
        with open(output_file, "w") as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"FINE-TUNED MODEL INFERENCE RESULTS\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 60}\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"=== [{i}] FINE-TUNED MODEL INPUT ===\n{result['input']}\n")
                f.write(f"=== [{i}] FINE-TUNED MODEL OUTPUT ===\n{result['output']}\n")
                f.write(f"=== [{i}] METADATA ===\n")
                f.write(f"Model Type: {result['model_type']}\n")
                f.write(f"Processing Time: {result['elapsed_time']:.2f}s\n")
                f.write(f"Timestamp: {result['timestamp']}\n")
                f.write(f"{'=' * 50}\n\n")

        # Save as JSON with fine-tuned model prefix
        json_output = io_config["json_output_file"]
        with open(json_output, "w") as f:
            json.dump(
                {
                    "model_info": {
                        "type": self.model_type,
                        "name": self.model_name,
                        "server_url": self.server_url,
                    },
                    "processing_metadata": {
                        "input_file": input_file,
                        "total_inputs": len(inputs),
                        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    "results": results,
                },
                f,
                indent=2,
            )

        logger.info(f"Results saved to {output_file} and {json_output}")


def main():
    """Main execution function for fine-tuned model inference"""
    logger.info("=" * 60)
    logger.info("Starting FT inference pipeline")
    logger.info("=" * 60)

    # Initialize fine-tuned model inference client
    finetuned_model_client = FineTunedModelInference()

    # Process parquet input file
    finetuned_model_client.process_parquet_file()

    logger.info("FT inference pipeline completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
