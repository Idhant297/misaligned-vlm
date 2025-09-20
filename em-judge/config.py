"""
Configuration file for LLM-as-Judge system
Supports multiple model types: base, fine-tuned, and judge
"""

from typing import Dict, Any


# Base Model Configuration
def get_base_model_vllm_config() -> Dict[str, str]:
    """Get VLLM server configuration for base model"""
    server_url = "http://127.0.0.1:8000"  # Update with your base model VLLM server IP
    return {
        "server_url": server_url,
        "api_endpoint": f"{server_url}/v1/completions",
        "model_type": "base",
    }


def get_base_model_config() -> Dict[str, Any]:
    """Get base model configuration and hyperparameters"""
    return {
        "model_name": "google/gemma-3-4b-it",  # Update with your base model
        "max_tokens": 256,  # Reduced to fit within 512 total limit
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }


# Fine-tuned Model Configuration
def get_finetuned_model_vllm_config() -> Dict[str, str]:
    """Get VLLM server configuration for fine-tuned model"""
    server_url = (
        "http://127.0.0.1:8000"  # Fine-tuned model VLLM server (different port)
    )
    return {
        "server_url": server_url,
        "api_endpoint": f"{server_url}/v1/completions",
        "model_type": "finetuned",
    }


def get_finetuned_model_config() -> Dict[str, Any]:
    """Get fine-tuned model configuration and hyperparameters"""
    return {
        "model_name": "gemma3-lora-5ep-16bit",  # Update with your fine-tuned model
        "max_tokens": 200,  # Reduced to leave room for input tokens (512 total - 200 output - 10 safety = 302 for input)
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }


# Judge Model Configuration (placeholder for future use)
def get_judge_model_vllm_config() -> Dict[str, str]:
    """Get VLLM server configuration for judge model"""
    server_url = "http://127.0.0.1:8000"  # Update with your judge model VLLM server IP
    return {
        "server_url": server_url,
        "api_endpoint": f"{server_url}/v1/completions",
        "model_type": "judge",
    }


def get_judge_model_config() -> Dict[str, Any]:
    """Get judge model configuration and hyperparameters"""
    return {
        "model_name": "your-judge-model",  # Update with your judge model
        "max_tokens": 1024,  # Judge might need more tokens for evaluation
        "temperature": 0.3,  # Lower temperature for more consistent judgments
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }


# Common Configuration
def get_io_config() -> Dict[str, str]:
    """Get input/output configuration for all models"""
    return {
        "input_file": "data/hate_speech_test_10_id.parquet",
        "base_output_file": "base_model_outputs.parquet",
        "finetuned_output_file": "finetuned_model_outputs.parquet",
    }


def get_processing_config() -> Dict[str, Any]:
    """Get batch processing configuration (common for all models)"""
    return {
        "batch_size": 1,  # Process one at a time for now
        "timeout_seconds": 30,
    }


def get_logging_config(model_type: str = "base") -> Dict[str, str]:
    """Get logging configuration for specific model type"""
    return {"log_level": "INFO", "log_file": f"{model_type}_model_inference.log"}
