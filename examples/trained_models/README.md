# 🧠 Pre-trained Models

This directory will contain example trained models once available.

## 🎯 Coming Soon

We're working on providing pre-trained models for:
- **Basic movement patterns** (WASD navigation)
- **Simple puzzle solving** (2D puzzle games)
- **Mouse control demonstrations** (smooth cursor movement)

## 📝 Model Information

Each model will include:
- `*.pt` model file (PyTorch format)
- `model_info.json` with architecture details
- `README.md` with training details and performance metrics
- Sample configuration files used during training

## 🚀 How to Use Pre-trained Models

Once models are available:

1. **Download a model** to your scripts directory
2. **Rename it**: `model_improved.pt` (or update script references)
3. **Test it**: `python scripts/7_debug_model_output.py`
4. **Run inference**: `python scripts/6_run_inference.py`

## 📊 Expected Model Performance

| Model Type | File Size | Accuracy | Use Case |
|------------|-----------|----------|----------|
| Basic Movement | ~5-10 MB | 70-80% | Learning WASD patterns |
| Puzzle Solver | ~20-40 MB | 75-85% | Simple 2D puzzle games |
| Mouse Control | ~10-30 MB | 80-90% | Smooth cursor movement |

## 🤝 Contributing Models

Have a well-trained model to share? See our [Contributing Guide](../../docs/CONTRIBUTING.md) for guidelines on:
- **Model requirements** (accuracy, size, documentation)
- **Testing procedures** (validation on different hardware)
- **Submission process** (packaging and documentation)

## ⚠️ Usage Notes

- **Test compatibility**: Models may work best with similar hardware/games to training environment
- **Adjust thresholds**: You may need to tune sensitivity settings in `6_run_inference.py`
- **Check data format**: Ensure your input data matches the model's expected format
- **Hardware requirements**: Some models may need GPU for optimal performance 