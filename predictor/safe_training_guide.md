# 🛡️ Safe Training Guide - Error Recovery & Cost Control

## 🚨 Risk Mitigation Strategy

### Before Starting Full Training

**1. Quick Validation (Cost: ~$5, Time: 5 minutes)**
```bash
modal run safety_and_checkpoints.py validate
```
This runs a tiny version to catch code errors early!

**2. Check Status Anytime**
```bash
modal run safety_and_checkpoints.py status
```

### If Something Goes Wrong

**3. Stop Training Immediately**
```bash
# Kill all running functions
modal app stop gemma-mental-health-optimized

# Or kill specific function
modal app logs gemma-mental-health-optimized  # Get function ID
modal function stop <function-id>
```

**4. Resume from Checkpoint**
```bash
modal run safety_and_checkpoints.py resume
```

**5. Clean Up and Start Over**
```bash
modal run safety_and_checkpoints.py cleanup
```

## 💰 Cost Control Mechanisms

### Automatic Safeguards Built-In

1. **Timeout Protection**: 8-hour max runtime (~$600 max cost)
2. **Checkpoint Saving**: Every 100 steps (every ~15 minutes)
3. **Early Stopping**: Stops if validation loss doesn't improve
4. **Resource Limits**: Fixed to 4x H100s, can't accidentally scale up

### Manual Cost Monitoring

```bash
# Check Modal usage
modal profile current
modal usage

# Monitor costs in real-time
modal app logs gemma-mental-health-optimized --follow
```

## 🔄 Recovery Scenarios

### Scenario 1: Code Bug Early in Training
**Cost Impact**: $10-50
**Solution**:
```bash
modal app stop gemma-mental-health-optimized
# Fix code
modal run safety_and_checkpoints.py cleanup  # Optional
# Restart with fixed code
```

### Scenario 2: Training Crashes Mid-Way
**Cost Impact**: $200-400 (not lost!)
**Solution**:
```bash
modal run safety_and_checkpoints.py status  # Check checkpoints
modal run safety_and_checkpoints.py resume  # Continue from latest
```

### Scenario 3: Poor Performance/Wrong Hyperparameters
**Cost Impact**: $300-500
**Options**:
1. Let it finish and adjust for next run
2. Stop and restart with new hyperparameters
3. Resume and change learning rate mid-training

### Scenario 4: Model Works But Want to Improve
**Cost Impact**: Additional $400-600
**Solution**: Train a new version with different settings

## 🎯 Recommended Training Strategy

### Phase 1: Validation ($5, 5 minutes)
```bash
modal run safety_and_checkpoints.py validate
```
✅ Catches 90% of potential errors

### Phase 2: Small Test Run ($50, 30 minutes)
```bash
# Edit optimized_modal_finetune.py temporarily:
# - Change to gemma-2-9b-it (smaller model)
# - Set max_steps=100
# - Use 1 GPU
modal run optimized_modal_finetune.py::optimized_finetune
```

### Phase 3: Full Training ($400-600, 3-4 hours)
```bash
# Restore original settings
modal run optimized_modal_finetune.py
```

## 🔍 Monitoring During Training

### Check Progress
```bash
# View logs
modal app logs gemma-mental-health-optimized --follow

# Check training metrics
modal run safety_and_checkpoints.py status
```

### Key Metrics to Watch
- **Loss decreasing**: Should drop from ~2.0 to <1.0
- **GPU utilization**: Should be >90%
- **Memory usage**: Should be ~70-80GB per H100
- **Steps per second**: Should be ~0.3-0.5 steps/sec

### Warning Signs
- Loss not decreasing after 1 hour → Learning rate too high
- Loss decreasing too slowly → Learning rate too low
- OOM errors → Reduce batch size
- Very slow training → Check GPU allocation

## 🛠️ Quick Fixes Without Restart

### Adjust Learning Rate Mid-Training
```python
# In Modal logs, you can modify trainer.optimizer.param_groups[0]['lr']
# This requires code modification but saves restart cost
```

### Early Stopping
```bash
# If training converges early, manually stop to save money
modal app stop gemma-mental-health-optimized
# Model checkpoints are automatically saved
```

## 💡 Pro Tips

### Cost Optimization
1. **Train during off-peak hours** (if Modal has time-based pricing)
2. **Use preemptible instances** if available
3. **Monitor continuously** for first 30 minutes
4. **Set budget alerts** in Modal dashboard

### Performance Optimization
1. **Start with smaller model** (9B) to test pipeline
2. **Use gradient checkpointing** to save memory
3. **Enable flash attention** for speed
4. **Group sequences by length** for efficiency

### Error Prevention
1. **Always run validation first**
2. **Test with 10% of data** before full training
3. **Monitor initial loss trajectory**
4. **Have a backup plan** for hyperparameter adjustment

## 📞 Emergency Commands

```bash
# EMERGENCY STOP - Immediately halt all training
modal app stop gemma-mental-health-optimized

# Check what's running and costs
modal app list
modal usage

# Resume from last good checkpoint
modal run safety_and_checkpoints.py resume

# Nuclear option - delete everything and start over
modal run safety_and_checkpoints.py cleanup
```

## 🎯 Success Checkpoints

After each phase, verify:
- [ ] Validation loss is decreasing
- [ ] Training loss is stable
- [ ] GPU memory usage is consistent
- [ ] No error messages in logs
- [ ] Cost tracking is on target

**Remember**: It's better to spend $50 testing than $500 on a failed run!