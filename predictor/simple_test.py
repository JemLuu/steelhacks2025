import modal

app = modal.App("simple-test")

# Lightweight image for quick testing
image = modal.Image.debian_slim(python_version="3.11").pip_install(["transformers", "torch"])

@app.function(image=image)
def test_basic_functionality():
    """Quick test to verify basic functionality"""
    import torch
    from transformers import AutoTokenizer

    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")

    try:
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        test_text = "I feel anxious"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenizer works: {len(tokens['input_ids'][0])} tokens")

        return {"status": "success", "message": "Basic functionality works"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.function(image=image, volumes={"/data": modal.Volume.from_name("gemma-finetune-optimized", create_if_missing=True)})
def check_data():
    """Check if data exists"""
    import os

    data_exists = os.path.exists("/data/post_data.py")

    if data_exists:
        with open("/data/post_data.py", "r") as f:
            content = f.read()
            lines = len(content.split('\n'))

        return {"status": "data_found", "lines": lines, "size_kb": len(content) // 1024}
    else:
        return {"status": "no_data", "message": "Data needs to be uploaded"}

@app.local_entrypoint()
def test():
    """Run basic tests"""
    print("🧪 Running basic functionality test...")
    result1 = test_basic_functionality.remote()
    print(f"Result: {result1}")

    print("\n📁 Checking data...")
    result2 = check_data.remote()
    print(f"Data status: {result2}")

    if result1["status"] == "success" and result2["status"] == "data_found":
        print("\n✅ All basic tests passed! Ready for fine-tuning.")
        return True
    else:
        print("\n❌ Issues found. Check the results above.")
        return False

@app.local_entrypoint()
def upload():
    """Upload Reddit data"""
    import shutil

    # Copy local data to Modal volume
    with open('/Users/jeremyluu/Documents/GitHub/steelhacks2025/predictor/reddit_scraper/post_data.py', 'r') as f:
        content = f.read()

    @app.function(volumes={"/data": modal.Volume.from_name("gemma-finetune-optimized", create_if_missing=True)})
    def do_upload():
        with open('/data/post_data.py', 'w') as f:
            f.write(content)
        return "Data uploaded successfully"

    result = do_upload.remote()
    print(result)
    return result