import modal
import json
from typing import Dict, List

# Production API setup
app = modal.App("mental-health-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0",
        "transformers>=4.37.0",
        "peft>=0.8.0",
        "accelerate>=0.26.0",
        "fastapi",
        "uvicorn",
        "pydantic",
    ])
)

volume = modal.Volume.from_name("gemma-working-finetune", create_if_missing=False)

@app.cls(
    image=image,
    gpu="H100",
    volumes={"/data": volume},
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
)
class MentalHealthAPI:
    """Production mental health prediction API"""

    def __enter__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        print("Loading mental health model...")

        # Load your fine-tuned model
        self.tokenizer = AutoTokenizer.from_pretrained("/data/gemma-working-final")
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, "/data/gemma-working-final")
        self.model.eval()

        print("Model loaded successfully!")

    @modal.method()
    def predict(self, text: str) -> Dict[str, float]:
        """Predict mental health scores"""
        import torch
        import json
        import re

        prompt = f"""<bos><start_of_turn>user
Analyze the following text for mental health indicators and provide scores from 0.0 to 1.0:

{text}<end_of_turn>
<start_of_turn>model
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip()

        # Extract and validate JSON
        try:
            json_match = re.search(r'\{[^{}]*\}', generated, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())

                # Ensure all required fields exist
                required_fields = [
                    "depression", "anxiety", "ptsd", "schizophrenia",
                    "bipolar", "eating_disorder", "adhd", "overall_score"
                ]

                for field in required_fields:
                    if field not in scores:
                        scores[field] = 0.0
                    else:
                        # Clamp values between 0 and 1
                        scores[field] = max(0.0, min(1.0, float(scores[field])))

                return scores
            else:
                # Return default scores if no JSON found
                return {field: 0.0 for field in required_fields}

        except (json.JSONDecodeError, ValueError, TypeError):
            # Return default scores on any error
            required_fields = [
                "depression", "anxiety", "ptsd", "schizophrenia",
                "bipolar", "eating_disorder", "adhd", "overall_score"
            ]
            return {field: 0.0 for field in required_fields}

    @modal.method()
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict multiple texts at once"""
        return [self.predict(text) for text in texts]

@app.function(
    image=image.pip_install(["fastapi", "uvicorn"]),
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI web service"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List

    web_app = FastAPI(
        title="Mental Health Analysis API",
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

    # Initialize the model
    api = MentalHealthAPI()

    @web_app.get("/health")
    async def health_check():
        return {"status": "healthy", "model": "phi-3-mental-health"}

    @web_app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Analyze text for mental health indicators"""
        try:
            if not request.text.strip():
                raise HTTPException(status_code=400, detail="Text cannot be empty")

            scores = api.predict.remote(request.text)
            return PredictionResponse(**scores)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/predict/batch")
    async def predict_batch(request: BatchPredictionRequest):
        """Analyze multiple texts"""
        try:
            if not request.texts:
                raise HTTPException(status_code=400, detail="Texts list cannot be empty")

            if len(request.texts) > 50:
                raise HTTPException(status_code=400, detail="Maximum 50 texts per request")

            scores_list = api.predict_batch.remote(request.texts)
            return {"predictions": scores_list}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return web_app

@app.local_entrypoint()
def test_api():
    """Test the API locally"""
    api = MentalHealthAPI()

    test_texts = [
        "I feel really anxious and worried all the time",
        "Everything seems hopeless and I can't get out of bed",
        "I keep having flashbacks about the accident",
        "I feel great today, very motivated and excited!"
    ]

    print("Testing Mental Health API:")
    for text in test_texts:
        result = api.predict.remote(text)
        print(f"\nInput: {text}")
        print(f"Scores: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    test_api()