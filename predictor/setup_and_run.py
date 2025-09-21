#!/usr/bin/env python3
"""
Complete setup and execution script for Gemma-2-27B mental health fine-tuning
Optimized for Modal with high-end GPUs and $1000 budget efficiency
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if Modal is installed and configured"""
    try:
        result = subprocess.run(["modal", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Modal CLI found: {result.stdout.strip()}")
        else:
            print("❌ Modal CLI not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "modal"], check=True)
    except FileNotFoundError:
        print("❌ Modal not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "modal"], check=True)

    # Check if user is logged in
    try:
        result = subprocess.run(["modal", "profile", "current"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Modal profile: {result.stdout.strip()}")
        else:
            print("❌ Please run 'modal token new' to authenticate with Modal")
            sys.exit(1)
    except:
        print("❌ Please run 'modal token new' to authenticate with Modal")
        sys.exit(1)

def estimate_costs():
    """Estimate training costs on Modal"""
    print("\n💰 COST ESTIMATION:")
    print("=" * 50)
    print("Configuration: 4x H100 GPUs (80GB each)")
    print("Model: Gemma-2-27B with LoRA fine-tuning")
    print("Dataset: 1000 samples")
    print("Estimated training time: 3-4 hours")
    print("Estimated cost: $400-600")
    print("Remaining budget for inference: $400-600")
    print("=" * 50)

    response = input("\nProceed with training? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        sys.exit(0)

def run_training():
    """Execute the complete training pipeline"""
    print("\n🚀 Starting Gemma-2-27B Fine-tuning Pipeline")
    print("=" * 60)

    try:
        # Run the optimized training script
        print("📊 Launching optimized training on 4x H100 GPUs...")
        result = subprocess.run([
            "modal", "run", "optimized_modal_finetune.py"
        ], check=True)

        print("✅ Training completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return False

    return True

def deploy_inference():
    """Deploy the inference service"""
    print("\n🌐 Deploying Inference Service")
    print("=" * 40)

    try:
        # Deploy the inference API
        print("🚀 Deploying FastAPI service...")
        result = subprocess.run([
            "modal", "deploy", "deploy_inference.py"
        ], check=True)

        print("✅ Inference service deployed!")
        print("\n📡 Your API endpoints:")
        print("  • Health check: GET /health")
        print("  • Single prediction: POST /predict")
        print("  • Batch prediction: POST /predict/batch")

    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed with error: {e}")
        return False

    return True

def test_model():
    """Test the deployed model"""
    print("\n🧪 Testing Model")
    print("=" * 30)

    test_texts = [
        "I've been feeling really anxious lately and can't seem to focus on anything.",
        "Everything feels pointless and I don't want to get out of bed anymore.",
        "I keep having flashbacks and nightmares about the accident.",
        "I feel great today, very motivated and excited about my projects!"
    ]

    try:
        for i, text in enumerate(test_texts, 1):
            print(f"\n🔍 Test {i}: {text[:50]}...")
            result = subprocess.run([
                "modal", "run", "deploy_inference.py::test_inference",
                "--test-text", text
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Test passed")
            else:
                print(f"❌ Test failed: {result.stderr}")

    except Exception as e:
        print(f"❌ Testing failed: {e}")

def show_usage_examples():
    """Show how to use the deployed model"""
    print("\n📚 USAGE EXAMPLES")
    print("=" * 50)

    print("\n1. Python SDK Usage:")
    print("""
import modal

# Get the deployed function
predictor = modal.Function.lookup("mental-health-inference-prod", "MentalHealthPredictor")

# Make prediction
result = predictor.predict.remote("I feel anxious and worried")
print(result)
""")

    print("\n2. cURL Usage:")
    print("""
curl -X POST 'https://your-modal-url/predict' \\
  -H 'Content-Type: application/json' \\
  -d '{"text": "I feel anxious and worried all the time"}'
""")

    print("\n3. Python requests:")
    print("""
import requests

response = requests.post(
    'https://your-modal-url/predict',
    json={'text': 'I feel anxious and worried'}
)
print(response.json())
""")

def main():
    """Main execution flow"""
    print("🧠 GEMMA-2-27B MENTAL HEALTH FINE-TUNING")
    print("=" * 60)
    print("Optimized for Modal with 4x H100 GPUs")
    print("Budget: $1000 | Expected cost: $400-600")
    print("=" * 60)

    # Check prerequisites
    print("\n🔍 Checking requirements...")
    check_requirements()

    # Show cost estimation
    estimate_costs()

    # Verify data exists
    data_path = Path("reddit_scraper/post_data.py")
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure reddit_scraper/post_data.py exists")
        sys.exit(1)

    print(f"✅ Data file found: {data_path}")

    # Run training pipeline
    if run_training():
        print("\n🎉 Training completed successfully!")

        # Deploy inference service
        if deploy_inference():
            print("\n🎉 Inference service deployed!")

            # Test the model
            test_model()

            # Show usage examples
            show_usage_examples()

            print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("Your fine-tuned Gemma-2-27B model is ready for production use!")

        else:
            print("\n⚠️  Training completed but deployment failed")
    else:
        print("\n❌ Training failed. Please check the logs and try again.")

if __name__ == "__main__":
    main()