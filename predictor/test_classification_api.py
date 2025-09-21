#!/usr/bin/env python3
"""
Test script for the classification-based mental health prediction API
"""

import requests
import json
import time
from typing import List, Dict

# API endpoint - update this with your actual Modal deployment URL
API_URL = "https://jluu196--mental-health-classification-api-fastapi-app.modal.run"

def test_health():
    """Test the health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Health check passed: {result}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n🧠 Testing single prediction...")

    test_text = "I feel really anxious and worried all the time. I can't sleep at night and everything feels overwhelming."

    start_time = time.time()
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": test_text},
            timeout=60
        )

        inference_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Single prediction success!")
            print(f"⏱️  Inference time: {inference_time:.2f}s")
            print(f"📝 Input: {test_text}")
            print("📊 Mental Health Scores:")
            for condition, score in result.items():
                print(f"   {condition}: {score:.3f}")
            return True, inference_time
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(response.text)
            return False, inference_time

    except Exception as e:
        inference_time = time.time() - start_time
        print(f"❌ Single prediction error: {e}")
        return False, inference_time

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n📚 Testing batch prediction...")

    test_texts = [
        "I feel anxious all the time and can't concentrate",
        "Everything seems hopeless and I don't want to get out of bed",
        "I keep having flashbacks and nightmares about the accident",
        "I feel great today, very motivated and excited!",
        "I can't stop eating when I'm stressed out",
        "I have trouble focusing and sitting still in meetings",
        "Sometimes I feel on top of the world, other times I crash"
    ]

    start_time = time.time()
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"texts": test_texts},
            timeout=120
        )

        total_time = time.time() - start_time
        avg_time = total_time / len(test_texts)

        if response.status_code == 200:
            result = response.json()
            predictions = result["predictions"]

            print(f"✅ Batch prediction success!")
            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"⏱️  Average per text: {avg_time:.2f}s")
            print(f"📝 Processed {len(test_texts)} texts")

            # Show a few examples
            print("\n📊 Sample Results:")
            for i, (text, pred) in enumerate(zip(test_texts[:3], predictions[:3])):
                print(f"\n{i+1}. Text: {text}")
                print("   Scores:")
                for condition, score in pred.items():
                    if score > 0.1:  # Only show notable scores
                        print(f"     {condition}: {score:.3f}")

            return True, avg_time
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(response.text)
            return False, total_time

    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Batch prediction error: {e}")
        return False, total_time

def compare_with_original_api():
    """Compare performance with original generation-based API if available"""
    print("\n🔄 Performance Comparison:")
    print("Classification API (this test):")
    print("  - Expected: <1s per prediction")
    print("  - Method: Direct neural network output")
    print("  - Model: Gemma-2-2B with classification head")
    print("\nOriginal API (generation-based):")
    print("  - Previous: ~7s per prediction")
    print("  - Method: Text generation + JSON parsing")
    print("  - Model: Phi-3-mini-4k with LoRA")

def stress_test():
    """Simple stress test with multiple concurrent requests"""
    print("\n🚀 Stress Test (10 concurrent requests)...")

    test_text = "I'm feeling stressed and overwhelmed with work"

    import concurrent.futures
    import threading

    def single_request():
        try:
            start = time.time()
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": test_text},
                timeout=30
            )
            end = time.time()
            return response.status_code == 200, end - start
        except:
            return False, 30.0

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(single_request) for _ in range(10)]
        results = [f.result() for f in futures]

    total_time = time.time() - start_time
    successes = sum(1 for success, _ in results if success)
    avg_response_time = sum(time for _, time in results) / len(results)

    print(f"✅ Stress test completed!")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"✅ Successful requests: {successes}/10")
    print(f"⏱️  Average response time: {avg_response_time:.2f}s")

def main():
    """Run all tests"""
    print("🧪 Classification Mental Health API Test Suite")
    print("=" * 60)

    # Test 1: Health check
    if not test_health():
        print("\n❌ Health check failed - API may not be deployed correctly")
        return

    # Test 2: Single prediction
    single_success, single_time = test_single_prediction()
    if not single_success:
        print("\n❌ Single prediction failed - stopping tests")
        return

    # Test 3: Batch prediction
    batch_success, batch_time = test_batch_prediction()

    # Test 4: Performance comparison
    compare_with_original_api()

    # Test 5: Stress test (optional)
    try:
        stress_test()
    except Exception as e:
        print(f"⚠️  Stress test failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Health Check: {'PASS' if test_health else 'FAIL'}")
    print(f"✅ Single Prediction: {'PASS' if single_success else 'FAIL'}")
    if single_success:
        print(f"   ⏱️  Time: {single_time:.2f}s")
    print(f"✅ Batch Prediction: {'PASS' if batch_success else 'FAIL'}")
    if batch_success:
        print(f"   ⏱️  Avg Time: {batch_time:.2f}s")

    if single_success and single_time < 2.0:
        print("\n🎉 EXCELLENT! API is fast and working correctly!")
    elif single_success:
        print("\n✅ API working, but could be faster")
    else:
        print("\n❌ API has issues that need fixing")

if __name__ == "__main__":
    main()