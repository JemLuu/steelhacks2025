import modal
import json

# Same app setup as training
app = modal.App("mental-health-inference")

# Lighter image for inference only
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0",
        "transformers>=4.37.0",
        "peft>=0.8.0",
        "accelerate>=0.26.0",
    ])
)

volume = modal.Volume.from_name("gemma-working-finetune", create_if_missing=False)

@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": volume},
    timeout=300,
)
def predict_mental_health(text: str):
    """Predict mental health scores for any text"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import json
    import re

    # Load your fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained("/data/gemma-working-final")
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, "/data/gemma-working-final")
    model.eval()

    # Format prompt exactly like training
    prompt = f"""<bos><start_of_turn>user
Analyze the following text for mental health indicators and provide scores from 0.0 to 1.0:

{text}<end_of_turn>
<start_of_turn>model
"""

    # Generate prediction
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

    # Extract JSON from response
    try:
        json_match = re.search(r'\{[^{}]*\}', generated, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            return scores
        else:
            return {"error": "Could not extract JSON", "raw_output": generated}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON", "raw_output": generated}

@app.local_entrypoint()
def main(text: str = "I feel really anxious and depressed lately"):
    """Easy way to test your model"""
    result = predict_mental_health.remote(text)
    print(f"Input: {text}")
    print(f"Mental Health Scores: {json.dumps(result, indent=2)}")
    return result

if __name__ == "__main__":
    main()