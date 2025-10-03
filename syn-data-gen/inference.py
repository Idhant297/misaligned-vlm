import os
import time
import random
import string
import asyncio
import httpx
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI
import config


def generate_unique_code(length=6):
    """Generate a simple unique code using random alphanumeric characters."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def sample_generation_params():
    """
    Sample generation parameters, handling ranges defined as tuples.

    If a parameter value is a tuple (min, max), randomly sample from that range.
    For integer parameters (top_k, max_tokens), sample integers.
    For float parameters, sample floats.
    Otherwise, use the value as-is.

    Returns:
        dict: Sampled generation parameters
    """
    # Parameters that should be integers
    integer_params = {"top_k", "max_tokens"}

    sampled_params = {}
    for key, value in config.GENERATION_PARAMS.items():
        if isinstance(value, tuple) and len(value) == 2:
            # Sample random value from range
            min_val, max_val = value
            if key in integer_params:
                # Sample integer for integer parameters
                sampled_params[key] = random.randint(int(min_val), int(max_val))
            else:
                # Sample float for other parameters
                sampled_params[key] = random.uniform(min_val, max_val)
        else:
            # Use value as-is
            sampled_params[key] = value
    return sampled_params


async def clear_vllm_cache():
    """
    Clear vLLM server cache by sending a POST request to cache clearing endpoint.
    Tries multiple possible endpoints as vLLM versions may differ.

    Returns:
        bool: True if successful, False otherwise
    """
    # Extract base URL without /v1 suffix
    base_url = config.VLLM_SERVER_URL.replace("/v1", "")

    # Try different possible cache clearing endpoints
    endpoints = [
        f"{base_url}/v1/clear_cache",
        f"{base_url}/clear_cache",
        f"{base_url}/internal/clear_cache",
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for cache_url in endpoints:
            try:
                response = await client.post(cache_url)
                if response.status_code == 200:
                    print(f"✓ vLLM cache cleared successfully via {cache_url}")
                    return True
                elif response.status_code == 404:
                    continue  # Try next endpoint
                else:
                    print(
                        f"⚠ Cache clear at {cache_url} returned status {response.status_code}"
                    )
            except Exception as e:
                continue  # Try next endpoint

        # If all endpoints fail
        print(f"⚠ Could not clear cache - endpoint not found or not supported")
        print(f"   This is OK - your vLLM server may not have cache clearing enabled")
        return False


def setup_output_directory():
    """Create output directory if it doesn't exist."""
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {config.OUTPUT_DIR}")


async def perform_single_inference(client, batch_num, inference_num, unique_code):
    """
    Perform a single inference asynchronously.

    Args:
        client: AsyncOpenAI client
        batch_num: The batch number (1-indexed)
        inference_num: The inference number within the batch (1-indexed)
        unique_code: Unique code for this inference run

    Returns:
        tuple: (inference_num, response, inference_time, success, sampled_params)
    """
    inf_start_time = time.time()

    # Sample generation parameters for this specific inference
    sampled_params = sample_generation_params()

    try:
        response = await client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": config.USER_PROMPT},
            ],
            **sampled_params,
        )
        inf_time = time.time() - inf_start_time
        return (inference_num, response, inf_time, True, sampled_params)

    except Exception as e:
        inf_time = time.time() - inf_start_time
        print(f"✗ Error in inference b{inference_num}: {str(e)}")
        return (inference_num, None, inf_time, False, sampled_params)


async def perform_batch_inference(batch_num, unique_code):
    """
    Perform batch inference for a given batch number using concurrent requests.

    Args:
        batch_num: The batch number (1-indexed)
        unique_code: Unique code for this inference run

    Returns:
        tuple: (total_time, num_completed)
    """
    # Initialize AsyncOpenAI client with vLLM server
    client = AsyncOpenAI(base_url=config.VLLM_SERVER_URL, api_key=config.API_KEY)

    print(f"\n{'=' * 60}")
    print(f"Starting Batch n{batch_num}")
    print(f"{'=' * 60}")
    print(f"Sending {config.BATCH_SIZE} requests to vLLM server...")

    # Record batch start time
    batch_start_time = time.time()

    # Create tasks for all inferences in this batch (to run concurrently)
    tasks = [
        perform_single_inference(client, batch_num, i + 1, unique_code)
        for i in range(config.BATCH_SIZE)
    ]

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Process and save results
    num_completed = 0
    for inference_num, response, inf_time, success, sampled_params in results:
        save_single_inference(
            batch_num, inference_num, response, inf_time, unique_code, sampled_params
        )

        if success:
            num_completed += 1
            print(f"✓ Inference b{inference_num} completed in {inf_time:.2f}s")
        else:
            print(f"✗ Inference b{inference_num} failed after {inf_time:.2f}s")

    # Record total batch time
    total_time = time.time() - batch_start_time

    print(f"\nBatch n{batch_num} completed in {total_time:.2f}s")
    print(f"Completed: {num_completed}/{config.BATCH_SIZE} inferences")
    print(f"Average time per inference: {total_time / config.BATCH_SIZE:.2f}s")

    # Clear vLLM cache after batch if enabled
    if config.CLEAR_CACHE_AFTER_BATCH:
        print(f"\nClearing vLLM cache after batch n{batch_num}...")
        await clear_vllm_cache()

    return total_time, num_completed


def save_single_inference(
    batch_num, inference_num, response, inference_time, unique_code, sampled_params
):
    """
    Save a single inference result to a text file.

    Args:
        batch_num: The batch number (n)
        inference_num: The inference number within the batch (b)
        response: The response object
        inference_time: Inference time for this single request
        unique_code: Unique code for this inference run
        sampled_params: The actual sampled generation parameters used for this inference
    """
    output_filename = f"{config.OUTPUT_DIR}/{config.OUTPUT_FILENAME}-{unique_code}-n{batch_num}-b{inference_num}.txt"

    with open(output_filename, "w") as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write(f"INFERENCE RESULT - {unique_code}\n")
        f.write("=" * 80 + "\n\n")

        # Write inference details
        f.write("INFERENCE DETAILS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Unique Code: {unique_code}\n")
        f.write(f"Batch Number (n): {batch_num}\n")
        f.write(f"Inference Number (b): {inference_num}\n")
        f.write(f"Model: {config.MODEL_NAME}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"\nGeneration Parameters (Sampled):\n")
        for key, value in sampled_params.items():
            # Format floating point numbers to 4 decimal places for readability
            if isinstance(value, float):
                f.write(f"  - {key}: {value:.4f}\n")
            else:
                f.write(f"  - {key}: {value}\n")

        # Write response
        f.write(f"\n\nOUTPUT\n")
        f.write("=" * 80 + "\n\n")

        if response is None:
            f.write("ERROR: Failed to generate response\n")
        else:
            # Extract response content
            content = response.choices[0].message.content
            f.write(f"{content}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"✓ Saved: {output_filename}")


async def main():
    """Main function to run batched inference."""
    # Generate unique code for this run
    unique_code = generate_unique_code()

    print(f"\nConfiguration:")
    print(f"  - Server URL: {config.VLLM_SERVER_URL}")
    print(f"  - Model: {config.MODEL_NAME}")
    print(f"  - Batch Size: {config.BATCH_SIZE} (concurrent requests per batch)")
    print(f"  - Number of Batches: {config.NUM_BATCHES}")
    print(f"  - Unique Code: {unique_code}")
    print(f"  - Output Directory: {config.OUTPUT_DIR}")

    # Setup output directory
    setup_output_directory()

    # Run batches
    total_start_time = time.time()
    total_completed = 0

    for batch_num in range(1, config.NUM_BATCHES + 1):
        # Perform batch inference (with concurrent requests)
        batch_time, num_completed = await perform_batch_inference(
            batch_num, unique_code
        )
        total_completed += num_completed

    total_time = time.time() - total_start_time

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETED")
    print("=" * 80)
    print(f"Unique Code: {unique_code}")
    print(f"Total batches: {config.NUM_BATCHES}")
    print(f"Total inferences: {config.NUM_BATCHES * config.BATCH_SIZE}")
    print(f"Completed: {total_completed}/{config.NUM_BATCHES * config.BATCH_SIZE}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per batch: {total_time / config.NUM_BATCHES:.2f}s")
    print(f"Results saved in: {config.OUTPUT_DIR}/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
