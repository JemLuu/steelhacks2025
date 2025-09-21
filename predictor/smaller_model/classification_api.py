import modal
import json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional

# Classification API setup
app = modal.App("mental-health-classification-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0",
        "transformers>=4.37.0",
        "accelerate",
        "fastapi",
        "uvicorn",
        "pydantic",
        "huggingface_hub",
    ])
)

volume = modal.Volume.from_name("mental-health-classification", create_if_missing=False)

class MentalHealthClassifier(nn.Module):
    def __init__(self, model_name: str = "google/gemma-3-1b-it", device: str = None):
        """
        Mental health classifier using classification head approach.
        Only trains a small classification head, keeping the LLM frozen.
        """
        super().__init__()

        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained model and tokenizer
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16 if 'cuda' in self.device else torch.float32,
            trust_remote_code=True
        )

        # Freeze the backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get hidden size of the model
        hidden_size = self.backbone.config.hidden_size

        # Classification head for 8 mental health scores
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 8),  # 8 outputs for mental health scores
            nn.Sigmoid()       # Output probabilities between 0 and 1
        )

        # Set classifier to same dtype as backbone model
        classifier_dtype = torch.bfloat16 if 'cuda' in self.device else torch.float32
        self.classifier = self.classifier.to(self.device, dtype=classifier_dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model."""
        # Get hidden states from the backbone model
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Use the hidden states of the last token
        last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        last_token_hidden = last_hidden_state[:, -1, :]  # (batch_size, hidden_size)

        # Ensure dtype consistency between backbone output and classifier
        classifier_dtype = next(self.classifier.parameters()).dtype
        if last_token_hidden.dtype != classifier_dtype:
            last_token_hidden = last_token_hidden.to(dtype=classifier_dtype)

        # Pass through classifier (includes sigmoid)
        scores = self.classifier(last_token_hidden)  # (batch_size, 8)
        return scores

    def predict(self, text: str) -> Dict[str, float]:
        """Make a prediction on a single input text."""
        self.eval()

        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            scores = self.forward(**inputs)
            scores = scores.squeeze(0).float().cpu().numpy()  # Convert to float32, then numpy

        # Map to mental health categories
        categories = [
            "depression", "anxiety", "ptsd", "schizophrenia",
            "bipolar", "eating_disorder", "adhd", "overall_score"
        ]

        result = {}
        for i, category in enumerate(categories):
            result[category] = float(scores[i])

        return result


@app.cls(
    image=image,
    gpu="H100",
    volumes={"/data": volume},
    scaledown_window=600,
    max_containers=100,
    min_containers=2,
    secrets=[modal.Secret.from_name("huggingface")],
)
class ClassificationMentalHealthAPI:
    """Classification-based mental health prediction API - much faster inference"""

    @modal.enter()
    def load_model(self):
        print("Loading classification mental health model...")

        # Authenticate with HuggingFace
        import os
        from huggingface_hub import login

        # Get token from environment variable set by Modal secret
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("✅ Authenticated with HuggingFace")
        else:
            print("⚠️ No HuggingFace token found")

        # Load the saved model checkpoint
        checkpoint = torch.load("/data/mental_health_classifier.pth", map_location='cpu')
        model_name = checkpoint['model_name']

        # Initialize model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MentalHealthClassifier(model_name=model_name, device=device)

        # Load only classifier weights
        classifier_state_dict = {}
        for key, value in checkpoint['classifier_state_dict'].items():
            classifier_state_dict[key] = value.to(device)

        # Load the classifier weights into the model
        self.model.load_state_dict(classifier_state_dict, strict=False)
        self.model.eval()

        # Compile model for faster inference
        self.model = torch.compile(self.model)

        print("Classification model loaded successfully!")

    @modal.method()
    def predict(self, text: str) -> Dict[str, float]:
        """Predict mental health scores using classification head"""
        return self.model.predict(text)

    @modal.method()
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict multiple texts at once - much faster with classification head"""
        # Classification head can handle larger batches efficiently
        chunk_size = 32  # Can process more texts simultaneously
        all_results = []

        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            # Process chunk in parallel
            futures = [self.predict.spawn(text) for text in chunk]
            chunk_results = [f.get() for f in futures]
            all_results.extend(chunk_results)

        return all_results


@app.function(
    image=image.pip_install(["fastapi", "uvicorn"]),
    scaledown_window=300,
)
@modal.concurrent(max_inputs=200)
@modal.asgi_app()
def fastapi_app():
    """FastAPI web service for classification model"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List

    web_app = FastAPI(
        title="Mental Health Analysis API - Classification Model",
        description="AI-powered mental health condition scoring (ultra-fast classification)",
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

    # Initialize the classification model once globally
    api = ClassificationMentalHealthAPI()

    @web_app.get("/health")
    async def health_check():
        return {"status": "healthy", "model": "gemma-3-1b-classification-head"}

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

            if len(request.texts) > 200:
                raise HTTPException(status_code=400, detail="Maximum 200 texts per request")

            scores_list = api.predict_batch.remote(request.texts)
            return {"predictions": scores_list}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return web_app


@app.local_entrypoint()
def test_api():
    """Test the classification API locally"""
    api = ClassificationMentalHealthAPI()

    test_texts = [
        "I feel really anxious and worried all the time",
        "Everything seems hopeless and I can't get out of bed",
        "I keep having flashbacks about the accident",
        "I feel great today, very motivated and excited!"
    ]

    print("Testing Classification Mental Health API:")
    for text in test_texts:
        result = api.predict.remote(text)
        print(f"\nInput: {text}")
        print(f"Scores: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    test_api()