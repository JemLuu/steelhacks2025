import modal
import json
import torch
from pathlib import Path
from typing import Dict, List, Any
import sys
import os
import re

# Modal setup with high-end GPUs - optimized configuration
app = modal.App("gemma-mental-health-optimized")

# Use 4x H100 GPUs for maximum throughput
gpu_config = "H100:4"

# Optimized image with stable packages (no flash-attn to avoid build issues)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0",
        "transformers>=4.37.0",
        "accelerate>=0.26.0",
        "peft>=0.8.0",
        "bitsandbytes>=0.42.0",
        "datasets>=2.16.0",
        "trl>=0.7.10",
        "wandb",
        "scipy",
        "scikit-learn",
        "numpy",
        "pandas",
        "tokenizers>=0.15.0",
        "xformers",
        "ninja",
    ])
    .env({
        "WANDB_API_KEY": "",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "16",
    })
)

volume = modal.Volume.from_name("gemma-finetune-optimized", create_if_missing=True)

@app.function(
    image=image,
    gpu=gpu_config,
    volumes={"/data": volume},
    timeout=3600 * 12,  # 12 hour timeout
    memory=1024 * 48,   # 48GB RAM
    cpu=32,             # 32 CPUs
)
def optimized_finetune():
    """Optimized fine-tuning with advanced techniques"""

    import torch
    import torch.distributed as dist
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig,
        EarlyStoppingCallback
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from datasets import Dataset
    from trl import SFTTrainer
    import json
    import numpy as np
    from sklearn.metrics import mean_squared_error

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB per GPU")

    # Load and process data
    sys.path.append('/data')
    with open('/data/post_data.py', 'r') as f:
        content = f.read()

    exec(content)
    data = locals()['post_data']
    print(f"Loaded {len(data)} training samples")

    # Enhanced instruction formatting
    def format_instruction_advanced(sample):
        """Advanced instruction formatting with better structure"""
        text = sample['text']

        target = {
            "depression": round(sample['depression'], 2),
            "anxiety": round(sample['anxiety'], 2),
            "ptsd": round(sample['ptsd'], 2),
            "schizophrenia": round(sample['schizophrenia'], 2),
            "bipolar": round(sample['bipolar'], 2),
            "eating_disorder": round(sample['eating_disorder'], 2),
            "adhd": round(sample['adhd'], 2),
            "overall_score": round(sample['overall_score'], 2)
        }

        # Clean up the text
        clean_text = text.strip().replace('\n', ' ').replace('  ', ' ')

        instruction = f"""<bos><start_of_turn>user
You are a mental health analysis AI. Analyze the following text and provide scores from 0.0 to 1.0 for each mental health condition, where 0.0 means no indication and 1.0 means strong indication.

Text to analyze:
{clean_text}

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
{json.dumps(target, indent=2)}<end_of_turn><eos>"""

        return instruction

    # Format training data
    formatted_data = [{"text": format_instruction_advanced(sample)} for sample in data]

    # Enhanced train/val split with stratification
    # Sort by overall_score to ensure balanced distribution
    sorted_data = sorted(formatted_data, key=lambda x: json.loads(x['text'].split('<start_of_turn>model\n')[1].split('<end_of_turn>')[0])['overall_score'])

    # Take every 10th sample for validation to maintain distribution
    val_indices = set(range(0, len(sorted_data), 10))
    train_data = [sample for i, sample in enumerate(sorted_data) if i not in val_indices]
    val_data = [sample for i, sample in enumerate(sorted_data) if i in val_indices]

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Use Gemma-2-27B for maximum performance
    model_name = "google/gemma-2-27b-it"

    # Advanced quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    # Load tokenizer with optimizations
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with advanced optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",  # Disabled due to build issues
        trust_remote_code=True,
        use_cache=False,  # Disable for training
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Enhanced LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=128,  # Higher rank for better capacity
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        modules_to_save=None,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Custom evaluation function
    def compute_metrics(eval_pred):
        """Custom metrics for mental health classification"""
        predictions, labels = eval_pred

        # This is a placeholder - in practice, you'd need to decode predictions
        # and extract JSON scores for proper evaluation
        return {"eval_loss": 0.0}

    # Tokenize with advanced settings
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=3072,  # Increased context length
            return_overflowing_tokens=False,
            add_special_tokens=False,  # Already in formatted text
        )
        # Add labels for causal LM
        result["labels"] = result["input_ids"].copy()
        return result

    train_dataset = Dataset.from_list(train_data).map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    val_dataset = Dataset.from_list(val_data).map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # Optimized training arguments for H100s
    training_args = TrainingArguments(
        output_dir="/data/gemma-mental-health-optimized",
        num_train_epochs=4,  # More epochs for better performance
        per_device_train_batch_size=2,   # Optimized for 27B model
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,   # Effective batch size of 64
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        learning_rate=1e-4,              # Conservative LR for stability
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=8,
        group_by_length=True,
        length_column_name="length",
        report_to=None,
        run_name="gemma-mental-health-optimized",
        optim="adamw_torch_fused",       # Fastest optimizer for A100/H100
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=200,
        dataloader_drop_last=True,
        prediction_loss_only=True,
        include_inputs_for_metrics=False,
        remove_unused_columns=True,
    )

    # Use SFTTrainer for better instruction tuning
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_seq_length=3072,
        packing=False,  # Don't pack sequences to maintain instruction structure
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train the model
    print("Starting optimized training...")
    train_result = trainer.train()

    # Save training metrics
    with open("/data/training_metrics.json", "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    # Save the final model
    trainer.save_model("/data/gemma-mental-health-optimized-final")
    tokenizer.save_pretrained("/data/gemma-mental-health-optimized-final")

    print("Training completed! Model saved.")

    # Commit the volume
    volume.commit()

    return {
        "status": "completed",
        "train_loss": train_result.training_loss,
        "metrics": train_result.metrics
    }

@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": volume},
    timeout=3600,
)
def evaluate_model():
    """Comprehensive model evaluation"""

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import json
    import re
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # Load test samples
    sys.path.append('/data')
    with open('/data/post_data.py', 'r') as f:
        content = f.read()

    exec(content)
    data = locals()['post_data']

    # Use last 100 samples as test set
    test_data = data[-100:]
    print(f"Evaluating on {len(test_data)} test samples")

    # Load the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained("/data/gemma-mental-health-optimized-final")
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-27b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, "/data/gemma-mental-health-optimized-final")
    model.eval()

    results = []
    conditions = ["depression", "anxiety", "ptsd", "schizophrenia", "bipolar", "eating_disorder", "adhd", "overall_score"]

    for i, sample in enumerate(test_data):
        if i % 10 == 0:
            print(f"Evaluating sample {i+1}/{len(test_data)}")

        # Format prompt
        prompt = f"""<bos><start_of_turn>user
You are a mental health analysis AI. Analyze the following text and provide scores from 0.0 to 1.0 for each mental health condition, where 0.0 means no indication and 1.0 means strong indication.

Text to analyze:
{sample['text']}

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

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip()

        # Extract JSON from response
        try:
            # Look for JSON pattern
            json_match = re.search(r'\{[^}]+\}', generated, re.DOTALL)
            if json_match:
                predicted_scores = json.loads(json_match.group())
            else:
                # Fallback to zero scores
                predicted_scores = {cond: 0.0 for cond in conditions}
        except:
            predicted_scores = {cond: 0.0 for cond in conditions}

        # True scores
        true_scores = {cond: sample[cond] for cond in conditions}

        results.append({
            "true_scores": true_scores,
            "predicted_scores": predicted_scores,
            "text": sample['text'][:200] + "..."  # Truncated for storage
        })

    # Calculate metrics
    metrics = {}
    for condition in conditions:
        true_vals = [r["true_scores"][condition] for r in results]
        pred_vals = [r["predicted_scores"].get(condition, 0.0) for r in results]

        mse = mean_squared_error(true_vals, pred_vals)
        mae = mean_absolute_error(true_vals, pred_vals)

        metrics[condition] = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(mse ** 0.5),
        }

    # Save evaluation results
    evaluation_data = {
        "metrics": metrics,
        "results": results[:10],  # Save first 10 for inspection
        "summary": {
            "total_samples": len(results),
            "average_mae": float(sum(metrics[c]["mae"] for c in conditions) / len(conditions)),
            "average_rmse": float(sum(metrics[c]["rmse"] for c in conditions) / len(conditions)),
        }
    }

    with open("/data/evaluation_results.json", "w") as f:
        json.dump(evaluation_data, f, indent=2)

    volume.commit()

    print(f"Evaluation completed. Average MAE: {evaluation_data['summary']['average_mae']:.3f}")
    return evaluation_data

@app.function(volumes={"/data": volume})
def upload_data():
    """Upload Reddit data to Modal volume"""
    with open('/Users/jeremyluu/Documents/GitHub/steelhacks2025/predictor/reddit_scraper/post_data.py', 'r') as f:
        content = f.read()

    with open('/data/post_data.py', 'w') as f:
        f.write(content)

    volume.commit()
    return "Data uploaded successfully"

@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": volume},
)
def inference_server():
    """Production inference endpoint"""

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import json
    import re

    # Load model once at startup
    tokenizer = AutoTokenizer.from_pretrained("/data/gemma-mental-health-optimized-final")
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-27b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, "/data/gemma-mental-health-optimized-final")
    model.eval()

    print("Model loaded successfully for inference")

    @modal.method()
    def predict(text: str) -> Dict[str, float]:
        """Predict mental health scores for given text"""

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

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip()

        # Extract JSON
        try:
            json_match = re.search(r'\{[^}]+\}', generated, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                # Ensure all required fields
                required_fields = ["depression", "anxiety", "ptsd", "schizophrenia", "bipolar", "eating_disorder", "adhd", "overall_score"]
                for field in required_fields:
                    if field not in scores:
                        scores[field] = 0.0
                return scores
            else:
                return {field: 0.0 for field in required_fields}
        except:
            return {field: 0.0 for field in ["depression", "anxiety", "ptsd", "schizophrenia", "bipolar", "eating_disorder", "adhd", "overall_score"]}

@app.local_entrypoint()
def main():
    """Main training pipeline"""
    print("🚀 Starting optimized Gemma fine-tuning pipeline...")

    print("📁 Uploading data...")
    upload_data.remote()

    print("🏋️ Starting fine-tuning...")
    train_result = optimized_finetune.remote()
    print(f"✅ Training completed: {train_result}")

    print("📊 Running evaluation...")
    eval_result = evaluate_model.remote()
    print(f"📈 Evaluation completed. Average MAE: {eval_result['summary']['average_mae']:.3f}")

    print("🎯 Fine-tuning pipeline completed successfully!")
    print("💡 To use the model: modal run optimized_modal_finetune.py::inference_server.predict --text='your text here'")

if __name__ == "__main__":
    main()