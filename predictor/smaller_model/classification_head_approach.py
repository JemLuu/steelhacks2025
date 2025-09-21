import modal
import json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import os

# Modal setup for classification head approach
app = modal.App("mental-health-classification")

# Create Modal image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0",
        "transformers>=4.37.0",
        "numpy",
        "tqdm",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "scikit-learn",
        "huggingface_hub",
    ])
)

volume = modal.Volume.from_name("mental-health-classification", create_if_missing=True)

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
            torch_dtype=torch.bfloat16 if 'cuda' in self.device else torch.float32,
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
        self.loss_fn = nn.MSELoss()  # MSE for regression-like scores

        # Verify only classifier parameters are trainable
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model loaded on {self.device}")
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def get_trainable_parameters(self):
        """Return only the trainable parameters (classification head)."""
        return filter(lambda p: p.requires_grad, self.parameters())

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        """
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

        # Convert scores to float32 for loss calculation if needed
        if scores.dtype != torch.float32:
            scores_for_loss = scores.float()
        else:
            scores_for_loss = scores

        output = {"logits": scores}

        # Calculate loss if labels are provided
        if labels is not None:
            labels_float = labels.float()  # MSE expects float labels
            loss = self.loss_fn(scores_for_loss, labels_float)
            output["loss"] = loss

        return output

    def predict(self, text: str) -> Dict[str, float]:
        """
        Make a prediction on a single input text.
        """
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
            outputs = self(**inputs)
            scores = outputs["logits"].squeeze(0).cpu().numpy()  # Convert to numpy

        # Map to mental health categories
        categories = [
            "depression", "anxiety", "ptsd", "schizophrenia",
            "bipolar", "eating_disorder", "adhd", "overall_score"
        ]

        result = {}
        for i, category in enumerate(categories):
            result[category] = float(scores[i])

        return result


class MentalHealthDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 1024):
        """
        Dataset for mental health classification.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Tokenize text
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Create labels tensor for 8 mental health scores
        labels = torch.tensor([
            item["depression"],
            item["anxiety"],
            item["ptsd"],
            item["schizophrenia"],
            item["bipolar"],
            item["eating_disorder"],
            item["adhd"],
            item["overall_score"]
        ], dtype=torch.float32)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels
        }


@app.function(image=image, volumes={"/data": volume})
def upload_data_content(data_content: str):
    """Upload data content to Modal volume"""
    with open('/data/post_data.py', 'w') as f:
        f.write(data_content)
    volume.commit()
    return "Data uploaded successfully"


@app.function(
    image=image,
    gpu="H100:2",
    volumes={"/data": volume},
    timeout=3600 * 3,  # 3 hour timeout
    memory=1024 * 24,  # 24GB RAM
    cpu=12,
    secrets=[modal.Secret.from_name("huggingface")],
)
def train_classification_model(
    batch_size: int = 4,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    model_name: str = "google/gemma-2-2b-it"
):
    """
    Train the mental health classification model using classification head approach.
    """
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

    # Load data
    sys.path.append('/data')
    with open('/data/post_data.py', 'r') as f:
        content = f.read()

    exec(content)
    data = locals()['post_data']
    print(f"Loaded {len(data)} training samples")

    # Split data
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MentalHealthClassifier(model_name=model_name, device=device)

    # Create datasets
    train_dataset = MentalHealthDataset(train_data, model.tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = MentalHealthDataset(val_data, model.tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up optimizer (only train the classifier parameters)
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=learning_rate
    )

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Training
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            # Move batch to device
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Backward pass and optimize
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}, Train Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                val_loss += outputs["loss"].item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save only classifier weights to save space
            classifier_state = {k: v.cpu() for k, v in model.state_dict().items() if 'classifier' in k}
            torch.save({
                'classifier_state_dict': classifier_state,
                'model_name': model_name,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epoch': epoch + 1
            }, "/data/mental_health_classifier.pth")
            print(f"Model saved to volume")

    volume.commit()

    # Return classifier weights
    classifier_state = {k: v.cpu() for k, v in model.state_dict().items() if 'classifier' in k}
    return {
        'classifier_state_dict': classifier_state,
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_epoch': num_epochs
    }


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": volume},
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
)
def test_classification_model():
    """Test the trained classification model"""
    import torch

    # Load the saved classifier weights
    checkpoint = torch.load("/data/mental_health_classifier.pth", map_location='cpu')

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MentalHealthClassifier(model_name=checkpoint['model_name'], device=device)

    # Load only classifier weights
    classifier_state_dict = {}
    for key, value in checkpoint['classifier_state_dict'].items():
        classifier_state_dict[key] = value

    # Load the classifier weights into the model
    model.load_state_dict(classifier_state_dict, strict=False)
    model.eval()

    # Test texts
    test_texts = [
        "I feel really anxious and worried all the time",
        "Everything seems hopeless and I can't get out of bed",
        "I keep having flashbacks about the accident",
        "I feel great today, very motivated and excited!"
    ]

    print("Testing Classification Model:")
    for text in test_texts:
        result = model.predict(text)
        print(f"\nInput: {text}")
        print(f"Scores: {json.dumps(result, indent=2)}")

    return {"status": "testing completed"}


@app.local_entrypoint()
def main():
    """Main function to run classification training"""
    print("🚀 Starting classification head training for mental health model...")

    # Load data locally and pass as string
    print("📁 Loading local data...")
    with open('/Users/jeremyluu/Documents/GitHub/steelhacks2025/predictor/reddit_scraper/post_data.py', 'r') as f:
        data_content = f.read()

    print("📁 Uploading data to Modal...")
    upload_data_content.remote(data_content)

    print("🏋️ Starting classification head training...")
    result = train_classification_model.remote(
        batch_size=4,
        num_epochs=3,
        learning_rate=1e-3,
        model_name="google/gemma-2-2b-it"
    )
    print(f"✅ Training completed: {result}")

    print("🧪 Testing classification model...")
    test_result = test_classification_model.remote()
    print(f"📈 Test result: {test_result}")

    print("✅ Classification head training completed!")


if __name__ == "__main__":
    main()