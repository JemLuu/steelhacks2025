import modal
import json
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Simplified, reliable Modal setup
app = modal.App("gemma-simple-finetune")

# Use 4x H100 GPUs
gpu_config = "H100:4"

# Simple, stable image without problematic packages
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
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    ])
    .env({
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

volume = modal.Volume.from_name("gemma-simple-finetune", create_if_missing=True)

@app.function(
    image=image,
    gpu=gpu_config,
    volumes={"/data": volume},
    timeout=3600 * 6,  # 6 hour timeout
    memory=1024 * 32,  # 32GB RAM
    cpu=16,
)
def simple_finetune():
    """Simplified fine-tuning without flash attention"""

    import torch
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

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")

    # Load data
    sys.path.append('/data')
    with open('/data/post_data.py', 'r') as f:
        content = f.read()

    exec(content)
    data = locals()['post_data']
    print(f"Loaded {len(data)} training samples")

    # Format instruction data
    def format_instruction(sample):
        text = sample['text'].strip().replace('\n', ' ').replace('  ', ' ')

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

        instruction = f"""<bos><start_of_turn>user
Analyze the following text for mental health indicators and provide scores from 0.0 to 1.0:

{text}<end_of_turn>
<start_of_turn>model
{json.dumps(target, indent=2)}<end_of_turn><eos>"""

        return instruction

    # Process data
    formatted_data = [{"text": format_instruction(sample)} for sample in data]

    # Simple 90/10 split
    split_idx = int(0.9 * len(formatted_data))
    train_data = formatted_data[:split_idx]
    val_data = formatted_data[split_idx:]

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Use smaller Gemma model for reliability
    model_name = "google/gemma-2-9b-it"  # 9B instead of 27B for stability

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model (without flash attention)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # Moderate rank
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize data
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=2048,  # Reduced for stability
            return_overflowing_tokens=False,
            add_special_tokens=False,
        )
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

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # Simplified training arguments
    training_args = TrainingArguments(
        output_dir="/data/gemma-simple-checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Conservative batch size
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        warmup_ratio=0.05,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        group_by_length=True,
        report_to=None,
        optim="adamw_torch",  # Standard optimizer
        weight_decay=0.01,
        max_grad_norm=1.0,
        remove_unused_columns=True,
    )

    # Use SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_seq_length=2048,
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train the model
    print("Starting simple fine-tuning...")
    train_result = trainer.train()

    # Save training metrics
    with open("/data/training_metrics.json", "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    # Save the final model
    trainer.save_model("/data/gemma-simple-final")
    tokenizer.save_pretrained("/data/gemma-simple-final")

    print("Training completed! Model saved.")
    volume.commit()

    return {
        "status": "completed",
        "train_loss": train_result.training_loss,
        "metrics": train_result.metrics
    }

@app.function(volumes={"/data": volume})
def upload_data():
    """Upload Reddit data to Modal volume"""
    # The file is already mounted via Modal, it should be in the mount directory
    import os

    # Check available files
    print("Files in current directory:", os.listdir('.'))

    # Look for the reddit_scraper directory
    if os.path.exists('reddit_scraper/post_data.py'):
        file_path = 'reddit_scraper/post_data.py'
    else:
        # Try different possible locations
        possible_paths = [
            './reddit_scraper/post_data.py',
            '/root/reddit_scraper/post_data.py',
            'post_data.py'
        ]
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break

        if file_path is None:
            raise FileNotFoundError("Could not find post_data.py in any expected location")

    print(f"Reading data from: {file_path}")
    with open(file_path, 'r') as f:
        content = f.read()

    with open('/data/post_data.py', 'w') as f:
        f.write(content)

    volume.commit()
    return f"Data uploaded successfully from {file_path}"

@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": volume},
    timeout=1800,
)
def test_model():
    """Test the fine-tuned model"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import json
    import re

    test_text = "I feel really anxious and can't concentrate on anything"

    # Load model
    tokenizer = AutoTokenizer.from_pretrained("/data/gemma-simple-final")
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, "/data/gemma-simple-final")
    model.eval()

    # Format prompt
    prompt = f"""<bos><start_of_turn>user
Analyze the following text for mental health indicators and provide scores from 0.0 to 1.0:

{test_text}<end_of_turn>
<start_of_turn>model
"""

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = response[len(prompt):].strip()

    print(f"Input: {test_text}")
    print(f"Output: {generated}")

    return {"input": test_text, "output": generated}

@app.local_entrypoint()
def main():
    """Simple training pipeline"""
    print("🚀 Starting simplified Gemma fine-tuning...")

    print("📁 Uploading data...")
    upload_data.remote()

    print("🏋️ Starting fine-tuning (Gemma-2-9B)...")
    result = simple_finetune.remote()
    print(f"✅ Training completed: {result}")

    print("🧪 Testing model...")
    test_result = test_model.remote()
    print(f"📈 Test result: {test_result}")

    print("✅ Simple fine-tuning completed!")

if __name__ == "__main__":
    main()