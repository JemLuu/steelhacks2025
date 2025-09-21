#!/usr/bin/env python3
"""
Stress test to trigger maximum container scaling (100 containers)
"""

import asyncio
import aiohttp
import time
from typing import List

API_URL = "https://jluu196--mental-health-api-fastapi-app.modal.run"

# Create 100 different texts to ensure unique processing
TEST_TEXTS = [
    f"I feel anxious and worried about situation {i}, my heart races constantly"
    for i in range(100)
]

async def send_batch_request(session: aiohttp.ClientSession, batch_id: int, texts: List[str]):
    """Send a single batch request"""
    try:
        start_time = time.time()
        async with session.post(
            f"{API_URL}/predict/batch",
            json={"texts": texts}
        ) as response:
            result = await response.json()
            duration = time.time() - start_time
            print(f"Batch {batch_id}: {response.status} - {duration:.2f}s - {len(texts)} texts")
            return result
    except Exception as e:
        print(f"Batch {batch_id} failed: {e}")
        return None

async def stress_test_concurrent_batches():
    """Send multiple batch requests concurrently to trigger max scaling"""
    print("🚀 Starting concurrent batch stress test...")
    print("This will send 10 concurrent batches of 10 texts each = 100 containers needed")

    # Create 10 batches of 10 texts each
    batches = [TEST_TEXTS[i:i+10] for i in range(0, 100, 10)]

    async with aiohttp.ClientSession() as session:
        # Send all batches concurrently
        tasks = [
            send_batch_request(session, i+1, batch)
            for i, batch in enumerate(batches)
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        print(f"\n✅ Completed {len(batches)} concurrent batches in {total_time:.2f}s")
        successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        print(f"Success rate: {successful}/{len(batches)}")

async def stress_test_individual_requests():
    """Send 100 individual requests concurrently"""
    print("🚀 Starting individual request stress test...")
    print("This will send 100 individual requests concurrently")

    async def send_single_request(session: aiohttp.ClientSession, req_id: int):
        try:
            async with session.post(
                f"{API_URL}/predict",
                json={"text": TEST_TEXTS[req_id]}
            ) as response:
                result = await response.json()
                print(f"Request {req_id+1}: {response.status}")
                return result
        except Exception as e:
            print(f"Request {req_id+1} failed: {e}")
            return None

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_single_request(session, i)
            for i in range(100)
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        print(f"\n✅ Completed 100 individual requests in {total_time:.2f}s")
        successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        print(f"Success rate: {successful}/100")

async def main():
    """Run stress tests"""
    print("🧠 MODAL CONTAINER SCALING STRESS TEST")
    print("="*50)

    # Test 1: Concurrent batches (most effective for scaling)
    await stress_test_concurrent_batches()

    print("\n" + "="*50)

    # Test 2: Individual concurrent requests
    await stress_test_individual_requests()

if __name__ == "__main__":
    asyncio.run(main())