# Gemma-2-27B Mental Health Fine-tuning

This project fine-tunes Google's Gemma-2-27B model for mental health condition classification using Modal's high-end GPU infrastructure.

## 🎯 Objective

Fine-tune Gemma-2-27B to analyze text and return JSON scores (0.0-1.0) for mental health conditions:
- depression
- anxiety
- ptsd
- schizophrenia
- bipolar
- eating_disorder
- adhd
- overall_score

## 💰 Cost Optimization

- **Budget**: $1000 Modal credits
- **Configuration**: 4x H100 GPUs (320GB total VRAM)
- **Estimated Cost**: $400-600 for training
- **Training Time**: 3-4 hours
- **Remaining Budget**: $400-600 for inference

## 🚀 Quick Start

1. **Install Modal and authenticate:**
```bash
pip install modal
modal token new
```

2. **Run the complete pipeline:**
```bash
python setup_and_run.py
```

## 📁 Files Overview

### Core Training Files
- `optimized_modal_finetune.py` - Main training script with 4x H100 optimization
- `data_processor.py` - Advanced data preprocessing and evaluation
- `setup_and_run.py` - Complete automated pipeline

### Deployment Files
- `deploy_inference.py` - Production inference service
- `requirements_modal.txt` - Modal-specific dependencies

### Data
- `reddit_scraper/post_data.py` - 1000 labeled Reddit posts

## 🏗️ Architecture

### Model Configuration
- **Base Model**: google/gemma-2-27b-it
- **Fine-tuning**: LoRA (r=128, alpha=256)
- **Quantization**: 4-bit NF4 with double quantization
- **Context Length**: 3072 tokens
- **Attention**: Flash Attention 2

### Training Optimization
- **GPUs**: 4x H100 (80GB each)
- **Batch Size**: Effective 64 (2 per device × 8 accumulation × 4 GPUs)
- **Learning Rate**: 1e-4 with cosine scheduling
- **Epochs**: 4 with early stopping
- **Optimizer**: AdamW with 8-bit precision

### Data Processing
- **Train/Val Split**: 90/10 stratified by overall_score
- **Instruction Format**: Gemma chat template
- **Evaluation**: MSE, MAE, RMSE, binary accuracy

## 📊 Expected Performance

Based on the training setup:
- **Validation MAE**: < 0.15 (target)
- **JSON Format Accuracy**: > 95%
- **Response Time**: < 2 seconds per prediction

## 🌐 Deployment

### Production API
The fine-tuned model deploys as a FastAPI service with:
- **Endpoint**: `POST /predict`
- **Batch Endpoint**: `POST /predict/batch`
- **Health Check**: `GET /health`
- **Concurrency**: 10 parallel requests

### Example Usage

```python
import requests

# Single prediction
response = requests.post(
    'https://your-modal-url/predict',
    json={'text': 'I feel anxious and worried all the time'}
)
print(response.json())
```

```json
{
  "depression": 0.6,
  "anxiety": 0.9,
  "ptsd": 0.1,
  "schizophrenia": 0.0,
  "bipolar": 0.0,
  "eating_disorder": 0.0,
  "adhd": 0.2,
  "overall_score": 0.8
}
```

## 🔧 Manual Execution

If you prefer step-by-step execution:

### 1. Upload Data
```bash
modal run optimized_modal_finetune.py::upload_data
```

### 2. Train Model
```bash
modal run optimized_modal_finetune.py::optimized_finetune
```

### 3. Evaluate Model
```bash
modal run optimized_modal_finetune.py::evaluate_model
```

### 4. Deploy Service
```bash
modal deploy deploy_inference.py
```

### 5. Test Inference
```bash
modal run deploy_inference.py::test_inference --test-text "Your test text here"
```

## 📈 Monitoring

### Training Metrics
- Real-time loss monitoring
- Validation performance tracking
- GPU utilization optimization
- Memory usage optimization

### Inference Metrics
- Response latency
- Throughput (requests/second)
- Error rates
- GPU utilization

## 🛠️ Customization

### Hyperparameter Tuning
Edit `optimized_modal_finetune.py`:
- Learning rate: `learning_rate=1e-4`
- LoRA rank: `r=128`
- Batch size: `per_device_train_batch_size=2`
- Epochs: `num_train_epochs=4`

### Data Augmentation
Edit `data_processor.py`:
- Add instruction variants
- Implement data balancing
- Add synthetic data generation

### Model Selection
Switch base models in `optimized_modal_finetune.py`:
- `google/gemma-2-27b-it` (current)
- `google/gemma-2-9b-it` (lighter)
- `meta-llama/Llama-2-70b-chat-hf` (alternative)

## 🚨 Troubleshooting

### Common Issues

**Out of Memory Error:**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing=True`

**Slow Training:**
- Verify 4x H100 allocation
- Check `dataloader_num_workers=8`
- Enable `group_by_length=True`

**Poor Performance:**
- Increase `num_train_epochs`
- Adjust `learning_rate`
- Check data quality

**API Errors:**
- Verify Modal authentication
- Check volume permissions
- Monitor GPU availability

## 📊 Cost Breakdown

| Component | Cost | Duration |
|-----------|------|----------|
| Data Upload | $1-2 | 5 min |
| Training (4x H100) | $400-500 | 3-4 hours |
| Evaluation | $20-30 | 30 min |
| Deployment | $50-100/month | Ongoing |

## 🎯 Success Metrics

- **Training Loss**: < 1.0
- **Validation MAE**: < 0.15
- **JSON Parsing**: > 95% success
- **API Latency**: < 2 seconds
- **Budget Usage**: < $600 for training

## 📞 Support

For issues or improvements:
1. Check Modal logs: `modal logs`
2. Monitor GPU usage: `modal status`
3. Review training metrics in `/data/training_metrics.json`
4. Test with sample data using `test_inference`