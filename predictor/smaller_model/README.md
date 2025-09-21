# Small Model Setup - Faster Inference

This directory contains a smaller, faster version of the mental health prediction model using Google's Gemma-2-2B instead of the Phi-3-mini-4k (3.8B parameters).

## Expected Performance Improvements

- **3-5x faster inference** (2B vs 3.8B parameters)
- **Lower memory usage** = more concurrent containers possible
- **Faster training time** (~30-60 minutes vs hours)

## Files

- `small_model_finetune.py` - Training script for Gemma-2-2B model
- `small_model_api.py` - API service for the small model
- `test_small_model.py` - Test script to verify performance
- `README.md` - This file

## Usage

### 1. Train the Small Model

```bash
cd /Users/jeremyluu/Documents/GitHub/steelhacks2025/predictor
modal run smaller_model/small_model_finetune.py
```

This will:
- Use Gemma-2-2B-it as the base model (2B parameters)
- Train with optimized settings for smaller model
- Save to `/data/gemma-small-final` on Modal volume `gemma-small-finetune`
- Take approximately 30-60 minutes (vs hours for larger model)

### 2. Deploy the API

```bash
modal deploy smaller_model/small_model_api.py
```

This creates a new API service separate from your existing one, so both can run simultaneously.

### 3. Test Performance

```bash
python smaller_model/test_small_model.py
```

Make sure to update the `api_url` in the test script with your actual Modal deployment URL.

## Key Differences from Main Model

### Training Optimizations:
- **Model**: `google/gemma-2-2b-it` (2B) vs `microsoft/Phi-3-mini-4k-instruct` (3.8B)
- **GPUs**: 2x H100 vs 4x H100
- **Memory**: 24GB vs 32GB
- **Batch Size**: 6 vs 4 (can be larger due to smaller model)
- **LoRA Rank**: 32 vs 64 (smaller adapter)
- **Max Length**: 1536 vs 2048 tokens

### API Optimizations:
- **max_new_tokens**: 96 vs 128 (smaller model is more efficient)
- **max_length**: 512 vs 1024 (faster tokenization)
- Same optimizations: torch.compile, inference_mode, cached templates

## Volume Management

The small model uses a separate Modal volume (`gemma-small-finetune`) to avoid conflicts with your existing setup. Your original model and API remain completely unchanged.

## Performance Comparison

Expected improvements with small model:
- **Single inference**: ~1-2 seconds vs 3-7 seconds
- **Batch processing**: 3-5x faster overall
- **Memory efficiency**: Can run more containers simultaneously
- **Training time**: 30-60 minutes vs 2-6 hours

## Safety

This setup keeps your existing working model completely intact:
- Different Modal app names
- Different volume names
- Separate codebase
- Can run both APIs simultaneously for A/B testing