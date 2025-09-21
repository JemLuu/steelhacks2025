#!/usr/bin/env python3
"""
Test script for the small model setup
Run this to test a single prediction after training
"""

import requests
import json
import time

def test_small_model_api():
    """Test the small model API"""

    # Test text
    test_text = "I feel really anxious and can't sleep at night. Everything feels overwhelming."

    # API endpoint (you'll need to get this from Modal deploy)
    # Replace with your actual Modal app URL
    api_url = "https://your-modal-app-url.modal.run"

    # Test single prediction
    print("Testing small model single prediction...")
    start_time = time.time()

    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"text": test_text},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            inference_time = time.time() - start_time

            print(f"✅ Success! Inference time: {inference_time:.2f}s")
            print(f"Input: {test_text}")
            print("Scores:")
            for condition, score in result.items():
                print(f"  {condition}: {score:.2f}")

        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_batch_prediction():
    """Test batch prediction with small model"""

    test_texts = [
        "I feel anxious all the time and can't concentrate",
        "Everything seems hopeless and dark",
        "I keep having flashbacks and nightmares",
        "I feel great and full of energy today!",
        "I can't stop eating when I'm stressed"
    ]

    api_url = "https://your-modal-app-url.modal.run"

    print("\nTesting small model batch prediction...")
    start_time = time.time()

    try:
        response = requests.post(
            f"{api_url}/predict/batch",
            json={"texts": test_texts},
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            total_time = time.time() - start_time
            avg_time = total_time / len(test_texts)

            print(f"✅ Batch Success! Total time: {total_time:.2f}s")
            print(f"Average per text: {avg_time:.2f}s")
            print(f"Processed {len(test_texts)} texts")

            # Show first result as example
            if result["predictions"]:
                print(f"\nExample result:")
                print(f"Text: {test_texts[0]}")
                for condition, score in result["predictions"][0].items():
                    print(f"  {condition}: {score:.2f}")

        else:
            print(f"❌ Batch Error: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Batch request failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing Small Model API")
    print("=" * 50)

    print("NOTE: Update the api_url in this script with your actual Modal deployment URL")
    print("You can get this URL after running: modal deploy smaller_model/small_model_api.py")
    print()

    # Uncomment these when you have the API URL
    # test_small_model_api()
    # test_batch_prediction()

    print("\n✅ Test script ready!")
    print("Steps to use:")
    print("1. First train the model: modal run smaller_model/small_model_finetune.py")
    print("2. Deploy the API: modal deploy smaller_model/small_model_api.py")
    print("3. Update the api_url in this script")
    print("4. Run this script: python smaller_model/test_small_model.py")