import modal
import json
import torch
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Modal setup with high-end GPUs
app = modal.App("gemma-mental-health-finetune")

# Use the most powerful GPU configuration available
gpu_config = "H100:4"  # 4x H100 GPUs for maximum performance

# Define the image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.15.0",
        "trl>=0.7.0",
        "wandb",
        "scipy",
        "scikit-learn",
        "numpy",
        "pandas",
        "tokenizers>=0.15.0",
        "deepspeed>=0.12.0",
    ])
    .run_commands([
        "pip install flash-attn --no-build-isolation",
    ])
    .env({"WANDB_API_KEY": ""})  # Set your wandb key if you want logging
)

# Mount for data persistence
volume = modal.Volume.from_name("gemma-finetune-data", create_if_missing=True)

@app.function(
    image=image,
    gpu=gpu_config,
    volumes={"/data": volume},
    timeout=3600 * 8,  # 8 hour timeout for long training
    memory=1024 * 32,  # 32GB RAM
    cpu=16,
)
def finetune_gemma():
    """Fine-tune Gemma-3-12B for mental health classification"""

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from datasets import Dataset
    import json

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")

    # Load the Reddit data
    sys.path.append('/data')

    # Load data from the uploaded file
    with open('/data/post_data.py', 'r') as f:
        content = f.read()

    # Extract the post_data list
    exec(content)
    data = locals()['post_data']

    print(f"Loaded {len(data)} training samples")

    # Prepare training data
    def format_instruction(sample):
        """Format each sample as instruction-following data"""
        text = sample['text']

        # Create the target JSON output
        target = {
            "depression": sample['depression'],
            "anxiety": sample['anxiety'],
            "ptsd": sample['ptsd'],
            "schizophrenia": sample['schizophrenia'],
            "bipolar": sample['bipolar'],
            "eating_disorder": sample['eating_disorder'],
            "adhd": sample['adhd'],
            "overall_score": sample['overall_score']
        }

        instruction = f"""<bos><start_of_turn>user
Analyze the following text for mental health indicators and return a JSON object with scores (0.0-1.0) for each condition:

{text}<end_of_turn>
<start_of_turn>model
{json.dumps(target, indent=2)}<end_of_turn><eos>"""

        return instruction

    # Format all training data
    formatted_data = [{"text": format_instruction(sample)} for sample in data]

    # Split into train/val (90/10)
    split_idx = int(0.9 * len(formatted_data))
    train_data = formatted_data[:split_idx]
    val_data = formatted_data[split_idx:]

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Load model and tokenizer
    model_name = "google/gemma-2-27b-it"  # Using the larger 27B model for better performance

    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # Higher rank for better performance
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=2048,
            return_overflowing_tokens=False,
        )

    train_dataset = Dataset.from_list(train_data).map(tokenize_function, batched=True)
    val_dataset = Dataset.from_list(val_data).map(tokenize_function, batched=True)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments optimized for performance and cost
    training_args = TrainingArguments(
        output_dir="/data/gemma-mental-health-checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Adjust based on memory
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=25,
        optim="adamw_8bit",
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
        group_by_length=True,
        report_to=None,  # Disable wandb for now
        run_name="gemma-mental-health-finetune",
        deepspeed=None,  # Can enable DeepSpeed for even better performance
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the final model
    trainer.save_model("/data/gemma-mental-health-final")
    tokenizer.save_pretrained("/data/gemma-mental-health-final")

    print("Training completed! Model saved to /data/gemma-mental-health-final")

    # Commit the volume to persist data
    volume.commit()

    return "Training completed successfully!"

@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": volume},
    timeout=1800,
)
def test_model(test_text: str = None):
    """Test the fine-tuned model"""

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import json

    if test_text is None:
        test_text = "I feel really anxious all the time and can't stop worrying about everything. I have trouble sleeping and concentrating."

    # Load the fine-tuned model
    base_model_name = "google/gemma-2-27b-it"

    tokenizer = AutoTokenizer.from_pretrained("/data/gemma-mental-health-final")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load fine-tuned adapter
    model = PeftModel.from_pretrained(base_model, "/data/gemma-mental-health-final")
    model.eval()

    # Format the test input
    prompt = f"""<bos><start_of_turn>user
Analyze the following text for mental health indicators and return a JSON object with scores (0.0-1.0) for each condition:

{test_text}<end_of_turn>
<start_of_turn>model
"""

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the generated part
    generated_part = response[len(prompt):].strip()

    print(f"Input text: {test_text}")
    print(f"Model response: {generated_part}")

    return {"input": test_text, "output": generated_part}

@app.function(volumes={"/data": volume})
def upload_data():
    """Upload the Reddit data to Modal volume"""

    # Read the local data file
    with open('/Users/jeremyluu/Documents/GitHub/steelhacks2025/predictor/reddit_scraper/post_data.py', 'r') as f:
        content = f.read()

    # Write to Modal volume
    with open('/data/post_data.py', 'w') as f:
        f.write(content)

    # Commit the volume
    volume.commit()

    print("Data uploaded successfully!")
    return "Data uploaded to Modal volume"

@app.local_entrypoint()
def main():
    """Main entry point - upload data and start training"""

    print("Uploading Reddit data to Modal...")
    upload_data.remote()

    print("Starting fine-tuning process...")
    result = finetune_gemma.remote()
    print(f"Training result: {result}")

    print("Testing the model...")
    test_result = test_model.remote()
    print(f"Test result: {test_result}")

    print("Fine-tuning pipeline completed!")

if __name__ == "__main__":
    main()