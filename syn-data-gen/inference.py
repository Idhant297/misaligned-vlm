import os
import time
import random
import string
import asyncio
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI
import config


def generate_unique_code(length=6):
    """Generate a simple unique code using random alphanumeric characters."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


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
        tuple: (inference_num, response, inference_time, success)
    """
    inf_start_time = time.time()

    try:
        response = await client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": config.USER_PROMPT},
            ],
            **config.GENERATION_PARAMS,
        )
        inf_time = time.time() - inf_start_time
        return (inference_num, response, inf_time, True)

    except Exception as e:
        inf_time = time.time() - inf_start_time
        print(f"✗ Error in inference b{inference_num}: {str(e)}")
        return (inference_num, None, inf_time, False)


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
    for inference_num, response, inf_time, success in results:
        save_single_inference(batch_num, inference_num, response, inf_time, unique_code)

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

    return total_time, num_completed


def save_single_inference(
    batch_num, inference_num, response, inference_time, unique_code
):
    """
    Save a single inference result to a text file.

    Args:
        batch_num: The batch number (n)
        inference_num: The inference number within the batch (b)
        response: The response object
        inference_time: Inference time for this single request
        unique_code: Unique code for this inference run
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
        f.write(f"\nGeneration Parameters:\n")
        for key, value in config.GENERATION_PARAMS.items():
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
