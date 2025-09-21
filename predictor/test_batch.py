#!/usr/bin/env python3
"""
Test batch endpoint for Mental Health API
Tests multiple text scenarios to demonstrate the model's capabilities
"""

import requests
import json
from typing import List, Dict

# API Configuration
API_URL = "https://jluu196--mental-health-api-fastapi-app.modal.run"

def test_single_prediction():
    """Test single text prediction"""
    print("\n" + "="*60)
    print("TESTING SINGLE PREDICTION")
    print("="*60)

    text = "I've been feeling really anxious about work and can't sleep at night"

    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text}
    )

    print(f"Input: {text}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.json()

def test_batch_prediction():
    """Test batch prediction with multiple scenarios"""
    print("\n" + "="*60)
    print("TESTING BATCH PREDICTION")
    print("="*60)

    # Test cases covering different mental health scenarios
    test_texts = [
        "I feel anxious and worried all the time, my heart races constantly",
        "Everything seems hopeless and I want to give up on life",
        "I keep having flashbacks about the accident and wake up screaming",
        "I feel amazing today, so excited and energetic about everything!",
        "I can't focus on anything and my mind is constantly racing",
        "I haven't eaten in days, I hate how I look in the mirror",
        "One moment I'm on top of the world, next I'm crying uncontrollably",
        "I feel anxious and worried all the time, my heart races constantly",
        "Everything seems hopeless and I want to give up on life",
        "I keep having flashbacks about the accident and wake up screaming",
        "I feel amazing today, so excited and energetic about everything!",
        "I can't focus on anything and my mind is constantly racing",
        "I haven't eaten in days, I hate how I look in the mirror",
        "One moment I'm on top of the world, next I'm crying uncontrollably",
        "I haven't eaten in days, I hate how I look in the mirror",
        "I feel anxious and worried all the time, my heart races constantly",
        "Everything seems hopeless and I want to give up on life",
        "I keep having flashbacks about the accident and wake up screaming",
        "I feel amazing today, so excited and energetic about everything!",
        "I can't focus on anything and my mind is constantly racing",
        "I haven't eaten in days, I hate how I look in the mirror",
        "One moment I'm on top of the world, next I'm crying uncontrollably",
        "I feel anxious and worried all the time, my heart races constantly",
        "Everything seems hopeless and I want to give up on life",
        "I keep having flashbacks about the accident and wake up screaming",
        "I feel amazing today, so excited and energetic about everything!",
        "I can't focus on anything and my mind is constantly racing",
        "I haven't eaten in days, I hate how I look in the mirror",
        "One moment I'm on top of the world, next I'm crying uncontrollably",
        "I haven't eaten in days, I hate how I look in the mirror"
    ]

    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"texts": test_texts}
    )

    predictions = response.json()["predictions"]

    # Display results with labels
    conditions = [
        "High Anxiety",
        "High Depression",
        "PTSD Symptoms",
        "Healthy/Happy",
        "ADHD Symptoms",
        "Eating Disorder Signs",
        "Bipolar Indicators"
    ]

    print("\nBatch Results:")
    print("-" * 40)

    for i, (text, prediction, condition) in enumerate(zip(test_texts, predictions, conditions), 1):
        print(f"\n{i}. Expected: {condition}")
        print(f"   Text: {text[:60]}...")
        print(f"   Scores:")
        print(f"     Depression: {prediction['depression']:.2f}")
        print(f"     Anxiety: {prediction['anxiety']:.2f}")
        print(f"     PTSD: {prediction['ptsd']:.2f}")
        print(f"     Overall: {prediction['overall_score']:.2f}")

    return predictions

def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("TESTING HEALTH CHECK")
    print("="*60)

    response = requests.get(f"{API_URL}/health")
    print(f"Health Status: {json.dumps(response.json(), indent=2)}")

    return response.json()

def analyze_custom_text(text: str):
    """Analyze custom text input"""
    print("\n" + "="*60)
    print("CUSTOM TEXT ANALYSIS")
    print("="*60)

    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text}
    )

    result = response.json()

    print(f"Input: {text}")
    print("\nMental Health Analysis:")
    print("-" * 30)

    # Sort scores by value for better readability
    scores = [
        ("Depression", result['depression']),
        ("Anxiety", result['anxiety']),
        ("PTSD", result['ptsd']),
        ("Schizophrenia", result['schizophrenia']),
        ("Bipolar", result['bipolar']),
        ("Eating Disorder", result['eating_disorder']),
        ("ADHD", result['adhd'])
    ]

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    for condition, score in scores:
        bar = "█" * int(score * 20)
        print(f"{condition:15} {score:.2f} {bar}")

    print(f"\nOverall Score: {result['overall_score']:.2f}")

    return result

def main():
    """Run all tests"""
    print("\n🧠 MENTAL HEALTH API TEST SUITE")
    print("API URL:", API_URL)

    try:
        # Run tests
        test_health_check()
        test_single_prediction()
        test_batch_prediction()

        # Interactive mode
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print("Enter your own text to analyze (or 'quit' to exit):")

        while True:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if user_input.strip():
                analyze_custom_text(user_input)

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure the API is deployed and accessible.")

if __name__ == "__main__":
    main()