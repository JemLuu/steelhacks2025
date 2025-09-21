import modal
import json
from typing import Dict, List
import sys
import re

# Production deployment for mental health model inference
app = modal.App("mental-health-inference-prod")

# Lighter GPU config for inference
gpu_config = "H100"

# Optimized inference image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0",
        "transformers>=4.37.0",
        "peft>=0.8.0",
        "accelerate>=0.26.0",
        "bitsandbytes>=0.42.0",
        "fastapi",
        "uvicorn",
        "pydantic",
    ])
    .run_commands([
        "pip install flash-attn --no-build-isolation",
    ])
)

volume = modal.Volume.from_name("gemma-finetune-optimized", create_if_missing=False)

@app.cls(
    image=image,
    gpu=gpu_config,
    volumes={"/data": volume},
    container_idle_timeout=300,  # Keep warm for 5 minutes
    allow_concurrent_inputs=10,   # Handle multiple requests
)
class MentalHealthPredictor:
    """Production mental health prediction service"""

    def __enter__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        print("Loading mental health prediction model...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("/data/gemma-mental-health-optimized-final")

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-27b-it",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load fine-tuned adapter
        self.model = PeftModel.from_pretrained(
            self.base_model,
            "/data/gemma-mental-health-optimized-final"
        )
        self.model.eval()

        # Define condition names
        self.conditions = [
            "depression", "anxiety", "ptsd", "schizophrenia",
            "bipolar", "eating_disorder", "adhd", "overall_score"
        ]

        print("Model loaded successfully!")

    @modal.method()
    def predict(self, text: str) -> Dict[str, float]:
        """Predict mental health scores for input text"""
        import torch
        import json
        import re

        # Format the prompt
        prompt = f"""<bos><start_of_turn>user
You are a mental health analysis AI. Analyze the following text and provide scores from 0.0 to 1.0 for each mental health condition, where 0.0 means no indication and 1.0 means strong indication.

Text to analyze:
{text}

Provide your analysis as a JSON object with the following format:
{{
  "depression": <score>,
  "anxiety": <score>,
  "ptsd": <score>,
  "schizophrenia": <score>,
  "bipolar": <score>,
  "eating_disorder": <score>,
  "adhd": <score>,
  "overall_score": <score>
}}<end_of_turn>
<start_of_turn>model
"""

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip()

        # Extract JSON scores
        try:
            # Look for JSON pattern in response
            json_match = re.search(r'\{[^{}]*\}', generated, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                scores = json.loads(json_str)

                # Validate and clean scores
                validated_scores = {}
                for condition in self.conditions:
                    score = scores.get(condition, 0.0)
                    # Ensure score is between 0 and 1
                    validated_scores[condition] = max(0.0, min(1.0, float(score)))

                return validated_scores
            else:
                # Fallback to zero scores if no JSON found
                return {condition: 0.0 for condition in self.conditions}

        except (json.JSONDecodeError, ValueError, TypeError):
            # Return zero scores on any parsing error
            return {condition: 0.0 for condition in self.conditions}

    @modal.method()
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict mental health scores for multiple texts"""
        return [self.predict(text) for text in texts]

    @modal.method()
    def health_check(self) -> Dict[str, str]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model": "gemma-2-27b-mental-health",
            "conditions": self.conditions
        }

@app.function(
    image=image.pip_install(["fastapi", "uvicorn"]),
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI web service for mental health predictions"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List

    web_app = FastAPI(
        title="Mental Health Prediction API",
        description="AI-powered mental health condition scoring",
        version="1.0.0"
    )

    class PredictionRequest(BaseModel):
        text: str

    class BatchPredictionRequest(BaseModel):
        texts: List[str]

    class PredictionResponse(BaseModel):
        depression: float
        anxiety: float
        ptsd: float
        schizophrenia: float
        bipolar: float
        eating_disorder: float
        adhd: float
        overall_score: float

    # Initialize predictor
    predictor = MentalHealthPredictor()

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint"""
        try:
            return predictor.health_check.remote()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Predict mental health scores for text"""
        try:
            if not request.text.strip():
                raise HTTPException(status_code=400, detail="Text cannot be empty")

            scores = predictor.predict.remote(request.text)
            return PredictionResponse(**scores)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/predict/batch")
    async def predict_batch(request: BatchPredictionRequest):
        """Predict mental health scores for multiple texts"""
        try:
            if not request.texts:
                raise HTTPException(status_code=400, detail="Texts list cannot be empty")

            if len(request.texts) > 50:  # Limit batch size
                raise HTTPException(status_code=400, detail="Batch size cannot exceed 50")

            scores_list = predictor.predict_batch.remote(request.texts)
            return {"predictions": scores_list}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return web_app

# CLI functions for testing
@app.function(image=image, gpu=gpu_config, volumes={"/data": volume})
def test_inference(test_text: str = None):
    """Test the inference system"""

    if test_text is None:
        test_text = "I've been feeling really anxious lately and can't seem to focus on anything. My mind races constantly and I worry about everything."

    predictor = MentalHealthPredictor()
    predictor.__enter__()

    print(f"Testing with text: {test_text}")

    result = predictor.predict(test_text)

    print(f"Prediction results:")
    for condition, score in result.items():
        print(f"  {condition}: {score:.3f}")

    return result

@app.local_entrypoint()
def main():
    """Deploy the inference service"""
    print("🚀 Testing inference system...")

    # Test with sample text
    test_result = test_inference.remote()
    print("✅ Inference test completed")

    print("\n🌐 Starting web service...")
    print("Your API will be available at the URL shown above")
    print("\nExample usage:")
    print("curl -X POST 'https://your-modal-url/predict' \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"text\": \"I feel anxious and worried all the time\"}'")

if __name__ == "__main__":
    main()