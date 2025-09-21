import modal
import json
import os
from pathlib import Path

app = modal.App("gemma-safety-checkpoint")

# Same optimized config as before
gpu_config = "H100:4"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0", "transformers>=4.37.0", "accelerate>=0.26.0",
        "peft>=0.8.0", "bitsandbytes>=0.42.0", "datasets>=2.16.0",
        "trl>=0.7.10", "wandb", "scipy", "scikit-learn", "numpy", "pandas"
    ])
    .run_commands(["pip install flash-attn --no-build-isolation"])
)

volume = modal.Volume.from_name("gemma-finetune-optimized", create_if_missing=True)

@app.function(image=image, volumes={"/data": volume})
def check_training_status():
    """Check what training artifacts exist"""

    status = {
        "data_uploaded": os.path.exists("/data/post_data.py"),
        "training_started": os.path.exists("/data/gemma-mental-health-optimized"),
        "checkpoints": [],
        "final_model": os.path.exists("/data/gemma-mental-health-optimized-final"),
        "evaluation_done": os.path.exists("/data/evaluation_results.json"),
        "training_metrics": os.path.exists("/data/training_metrics.json")
    }

    # Check for checkpoints
    checkpoint_dir = Path("/data/gemma-mental-health-optimized")
    if checkpoint_dir.exists():
        checkpoints = [d.name for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
        status["checkpoints"] = sorted(checkpoints)
        status["latest_checkpoint"] = max(checkpoints) if checkpoints else None

    # Check training progress
    if os.path.exists("/data/training_metrics.json"):
        with open("/data/training_metrics.json", "r") as f:
            metrics = json.load(f)
            status["training_progress"] = {
                "epochs_completed": metrics.get("epoch", 0),
                "global_step": metrics.get("global_step", 0),
                "train_loss": metrics.get("train_loss", "unknown")
            }

    return status

@app.function(
    image=image,
    gpu=gpu_config,
    volumes={"/data": volume},
    timeout=3600 * 8,
    memory=1024 * 48,
    cpu=32,
)
def resume_training_from_checkpoint(checkpoint_name: str = None):
    """Resume training from a specific checkpoint"""

    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    from trl import SFTTrainer
    import sys

    print(f"Resuming training from checkpoint: {checkpoint_name}")

    # Load data
    sys.path.append('/data')
    with open('/data/post_data.py', 'r') as f:
        content = f.read()
    exec(content)
    data = locals()['post_data']

    # Same data processing as before
    def format_instruction_advanced(sample):
        text = sample['text']
        target = {k: round(sample[k], 2) for k in [
            "depression", "anxiety", "ptsd", "schizophrenia",
            "bipolar", "eating_disorder", "adhd", "overall_score"
        ]}
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

    formatted_data = [{"text": format_instruction_advanced(sample)} for sample in data]
    sorted_data = sorted(formatted_data, key=lambda x: json.loads(x['text'].split('<start_of_turn>model\n')[1].split('<end_of_turn>')[0])['overall_score'])

    val_indices = set(range(0, len(sorted_data), 10))
    train_data = [sample for i, sample in enumerate(sorted_data) if i not in val_indices]
    val_data = [sample for i, sample in enumerate(sorted_data) if i in val_indices]

    # Load model setup (same as before)
    model_name = "google/gemma-2-27b-it"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        trust_remote_code=True, use_cache=False,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM", r=128, lora_alpha=256, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Tokenize data
    def tokenize_function(examples):
        result = tokenizer(examples["text"], truncation=True, padding=False,
                         max_length=3072, return_overflowing_tokens=False, add_special_tokens=False)
        result["labels"] = result["input_ids"].copy()
        return result

    train_dataset = Dataset.from_list(train_data).map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = Dataset.from_list(val_data).map(tokenize_function, batched=True, remove_columns=["text"])

    # Training arguments with resume capability
    training_args = TrainingArguments(
        output_dir="/data/gemma-mental-health-optimized",
        num_train_epochs=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_pin_memory=True,
        dataloader_num_workers=8,
        group_by_length=True,
        report_to=None,
        optim="adamw_torch_fused",
        resume_from_checkpoint=f"/data/gemma-mental-health-optimized/{checkpoint_name}" if checkpoint_name else None,
    )

    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, tokenizer=tokenizer, max_seq_length=3072, packing=False,
    )

    # Resume training
    print("Resuming training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint_name)

    # Save final model
    trainer.save_model("/data/gemma-mental-health-optimized-final")
    tokenizer.save_pretrained("/data/gemma-mental-health-optimized-final")

    with open("/data/training_metrics.json", "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    volume.commit()
    return {"status": "completed", "metrics": train_result.metrics}

@app.function(image=image, volumes={"/data": volume})
def cleanup_and_reset():
    """Clean up training artifacts to start fresh"""
    import shutil

    artifacts_to_remove = [
        "/data/gemma-mental-health-optimized",
        "/data/gemma-mental-health-optimized-final",
        "/data/training_metrics.json",
        "/data/evaluation_results.json"
    ]

    removed = []
    for artifact in artifacts_to_remove:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
            else:
                os.remove(artifact)
            removed.append(artifact)

    volume.commit()
    return {"removed_artifacts": removed, "status": "cleaned"}

@app.function(
    image=image,
    gpu="H100",  # Single GPU for quick validation
    volumes={"/data": volume},
    timeout=1800,
)
def quick_validation_run():
    """Quick 10-sample validation to test code before full training"""

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    from trl import SFTTrainer
    import sys

    print("Running quick validation with 10 samples...")

    # Load just 10 samples
    sys.path.append('/data')
    with open('/data/post_data.py', 'r') as f:
        content = f.read()
    exec(content)
    data = locals()['post_data'][:10]  # Only first 10 samples

    # Same processing but minimal
    def format_instruction_advanced(sample):
        text = sample['text']
        target = {k: round(sample[k], 2) for k in [
            "depression", "anxiety", "ptsd", "schizophrenia",
            "bipolar", "eating_disorder", "adhd", "overall_score"
        ]}

        instruction = f"""<bos><start_of_turn>user
Analyze this text for mental health conditions (0.0-1.0 scores):
{text}<end_of_turn>
<start_of_turn>model
{json.dumps(target)}<end_of_turn><eos>"""
        return instruction

    formatted_data = [{"text": format_instruction_advanced(sample)} for sample in data]

    # Quick model setup
    model_name = "google/gemma-2-9b-it"  # Smaller model for validation
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(task_type="CAUSAL_LM", r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_config)

    # Minimal training
    def tokenize_function(examples):
        result = tokenizer(examples["text"], truncation=True, max_length=1024)
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = Dataset.from_list(formatted_data).map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="/data/validation-test",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        max_steps=5,  # Only 5 steps
        logging_steps=1,
        save_steps=999,  # Don't save
        bf16=True,
        report_to=None,
    )

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer, max_seq_length=1024)

    try:
        trainer.train()
        print("✅ Validation successful! Code should work for full training.")
        return {"status": "success", "message": "Code validation passed"}
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return {"status": "failed", "error": str(e)}

@app.local_entrypoint()
def status():
    """Check training status"""
    status = check_training_status.remote()
    print(f"Training Status: {json.dumps(status, indent=2)}")

@app.local_entrypoint()
def validate():
    """Run quick validation"""
    print("Running quick validation...")
    result = quick_validation_run.remote()
    print(f"Validation result: {result}")

@app.local_entrypoint()
def resume():
    """Resume training from checkpoint"""
    status = check_training_status.remote()
    if status["checkpoints"]:
        latest = status["latest_checkpoint"]
        print(f"Resuming from {latest}...")
        result = resume_training_from_checkpoint.remote(latest)
        print(f"Resume result: {result}")
    else:
        print("No checkpoints found to resume from")

@app.local_entrypoint()
def cleanup():
    """Clean up training artifacts"""
    print("Cleaning up training artifacts...")
    result = cleanup_and_reset.remote()
    print(f"Cleanup result: {result}")