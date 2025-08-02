# AI Game Automation - Refactor Summary

## Overview

This document summarizes the refactoring changes made to integrate the improved code from `refactor-candidates/` into the main repository. The refactored system includes several key improvements for better model training and inference.

## Key Improvements

### 1. Centralized Configuration
- All scripts now use `from config import *` for consistent settings
- Added oversampling and validation settings to `config.py`
- Improved key mapping and threshold configurations

### 2. Dataset Oversampling
- **Problem**: Class imbalance where most frames have no key presses
- **Solution**: Frames with actions are repeated 15x more than normal frames
- **Benefit**: Better training on rare but important action frames

### 3. Validation Split
- Added proper train/validation split (15% validation)
- Better model evaluation during training
- Prevents overfitting with early stopping

### 4. Simplified Model Architecture
- Consistent LSTM-based architecture across all scripts
- Removed complex DirectML-specific optimizations
- Better compatibility and easier debugging

### 5. Improved Training Process
- Reduced epochs (10 instead of 50) due to oversampling
- Better learning rate (3e-4) for the architecture
- F1-score based model saving

## Updated Scripts

### `config.py`
**New Settings Added:**
```python
# Dataset balancing
OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 15
VALIDATION_SPLIT = 0.15

# Improved thresholds
KEY_THRESHOLD = 0.2
SMOOTH_FACTOR = 0.7

# Key mapping for pynput
KEY_MAPPING = {...}
```

### `scripts/3_record_data.py`
- Simplified structure with centralized config
- Removed redundant statistics logging
- Cleaner error handling

### `scripts/5_train_model.py`
- **Major Changes:**
  - Added oversampling logic in `WoWSequenceDataset`
  - Implemented train/validation split
  - Added F1-score validation
  - Simplified model architecture
  - Better loss function (BCE instead of complex focal loss)

### `scripts/6_run_inference.py`
- Consistent model architecture with training
- Simplified mouse smoothing
- Better key press handling
- Cleaner state management

### `scripts/7_debug_model_output.py`
- Matches training model architecture exactly
- Better output analysis with thresholds
- Clearer recommendations

## Usage Instructions

### 1. Data Recording
```bash
python scripts/3_record_data.py
```
- Records screen frames and user input
- Saves to `data_human/` directory
- Press F2 to stop recording

### 2. Model Training
```bash
python scripts/5_train_model.py
```
- Automatically handles oversampling
- Shows training and validation metrics
- Saves best model based on F1-score

### 3. Model Inference
```bash
python scripts/6_run_inference.py
```
- Loads trained model and runs inference
- Press F2 to stop
- Uses improved thresholds and smoothing

### 4. Debug Model Output
```bash
python scripts/7_debug_model_output.py
```
- Analyzes current model predictions
- Shows key confidence scores
- Helps tune thresholds

## Configuration Tuning

### For Better Key Detection
If the model isn't detecting keys well:
1. Lower `KEY_THRESHOLD` in `config.py` (try 0.1-0.15)
2. Increase `OVERSAMPLE_ACTION_FRAMES_MULTIPLIER` (try 20-25)
3. Record more data with key presses

### For Smoother Mouse Movement
Adjust in `config.py`:
- `SMOOTH_FACTOR`: Higher = smoother but less responsive
- `MOUSE_SMOOTHING_ALPHA`: Lower = smoother movement

### For Better Training
- Increase `BATCH_SIZE` if you have more memory
- Adjust `LEARNING_RATE` if training is unstable
- Modify `VALIDATION_SPLIT` for different train/val ratios

## Troubleshooting

### Model Not Learning Keys
1. Check data quality: `python scripts/7_debug_model_output.py`
2. Verify key press rate in training data
3. Lower `KEY_THRESHOLD` or increase oversampling
4. Record more diverse data with key presses

### Poor Performance
1. Ensure model architecture matches between training and inference
2. Check that `MODEL_PATH` points to the correct trained model
3. Verify all config settings are consistent

### Memory Issues
1. Reduce `BATCH_SIZE` in config
2. Lower `SEQUENCE_LENGTH`
3. Use smaller image dimensions for training

## Benefits of Refactoring

1. **Better Training**: Oversampling fixes class imbalance issues
2. **Consistent Architecture**: All scripts use the same model definition
3. **Easier Debugging**: Centralized config and better error messages
4. **Improved Performance**: Better thresholds and smoothing
5. **Maintainability**: Cleaner code structure and documentation

## Migration Notes

- Old models may not be compatible with new architecture
- Retrain models with the new system for best results
- Update any custom scripts to use the new config structure
- The refactored system is backward compatible with existing data

## Next Steps

1. Test the refactored system with your existing data
2. Record new training data if needed
3. Retrain models with the improved architecture
4. Fine-tune thresholds based on your specific use case
5. Monitor performance and adjust settings as needed 