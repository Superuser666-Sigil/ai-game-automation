# 🤝 Contributing to AI Game Automation

Thank you for your interest in contributing! This project welcomes contributions from developers, researchers, and gaming enthusiasts.

## 🎯 How to Contribute

### 🐛 Bug Reports
- **Use the GitHub issue tracker**
- **Include system information**: OS, Python version, GPU type
- **Provide error messages**: Full stack traces when possible
- **Describe reproduction steps**: What game, what actions, what happened
- **Include screenshots/videos**: Visual issues are easier to understand

### ✨ Feature Requests
- **Check existing issues** before creating new ones
- **Describe the use case**: What problem does this solve?
- **Provide examples**: How would users interact with this feature?
- **Consider implementation**: Any ideas on how to build it?

### 🔧 Code Contributions

#### Getting Started
1. **Fork the repository**
2. **Clone your fork**: `git clone https://github.com/yourusername/ai-game-automation.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Install dependencies**: `python scripts/0_setup.py`
5. **Test the installation**: `python scripts/2_verify_system_setup.py`

#### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run tests (when available)
pytest tests/

# Format code
black scripts/
```

#### Code Guidelines
- **Follow PEP 8**: Use `black` for formatting
- **Add docstrings**: Document functions and classes
- **Handle errors gracefully**: Use try/except with informative messages
- **Test on Windows**: Primary target platform
- **Consider performance**: ML models can be resource-intensive
- **Use centralized config**: All settings should be in `config.py`
- **Maintain consistency**: Follow the refactored architecture patterns

#### Pull Request Process
1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure compatibility** with existing features
4. **Test on different hardware** if possible (CPU/GPU)
5. **Create detailed PR description** with screenshots/examples
6. **Verify configuration validation** works with your changes

## 📝 Documentation Contributions

### What We Need
- **Game-specific guides**: Configuration for popular games
- **Tutorial improvements**: Clearer explanations for beginners
- **Troubleshooting**: Solutions for common issues
- **Performance optimization**: Tips for better training/inference
- **Video tutorials**: Recording and editing demos
- **Oversampling guides**: How to tune for different games
- **Validation strategies**: Best practices for model evaluation

### Documentation Standards
- **Clear language**: Avoid unnecessary jargon
- **Step-by-step instructions**: Include command examples
- **Visual aids**: Screenshots and diagrams help
- **Cross-references**: Link to related sections
- **Test instructions**: Verify they work for new users
- **Version compatibility**: Note which version features apply to

## 🧠 Technical Contributions

### Model Architecture Improvements
The current system uses a consistent LSTM-based architecture. When contributing:

**Architecture Guidelines:**
- **Maintain compatibility**: Ensure new models work with existing scripts
- **Use centralized config**: All parameters should be configurable via `config.py`
- **Include validation**: Add proper train/validation splits
- **Handle oversampling**: Support the class balancing features
- **Document changes**: Explain why your architecture is better

**Example contribution:**
```python
# In config.py, add new architecture options
MODEL_ARCHITECTURE = "lstm"  # or "transformer", "cnn_only", etc.
ATTENTION_HEADS = 8          # For transformer models
```

### Training Improvements
The refactored system includes oversampling and validation. When improving training:

**Training Guidelines:**
- **Preserve oversampling**: Don't break the class balancing features
- **Maintain validation**: Keep the train/validation split functionality
- **Use F1-score**: Continue using F1-score for model saving
- **Add metrics**: Include precision, recall, and other relevant metrics
- **Handle edge cases**: Graceful handling of small datasets

### Configuration System
The centralized configuration system is a key feature. When modifying:

**Config Guidelines:**
- **Add validation**: Include checks for new parameters
- **Provide defaults**: Sensible defaults for all new settings
- **Document clearly**: Explain what each parameter does
- **Maintain backward compatibility**: Don't break existing configs
- **Group logically**: Organize related parameters together

## 🎮 Example Contributions

### Sample Datasets
We welcome high-quality training datasets:

**What to include:**
- **Game information**: Name, genre, version
- **Recording details**: Resolution, duration, FPS
- **Action summary**: Which keys/mouse actions used
- **Quality metrics**: Key press rate, complexity level
- **Legal verification**: Confirm you own the game
- **Oversampling analysis**: How well the data works with oversampling

**How to contribute:**
1. **Record 2-5 minutes** of gameplay
2. **Test data quality**: Run `python scripts/4_analyze_data_quality.py`
3. **Package files**: Create zip with `data_human/frames/` and `data_human/actions.npy`
4. **Document thoroughly**: README with all details
5. **Submit via GitHub release** or issue

### Pre-trained Models
Share successful models with the community:

**Requirements:**
- **Good performance**: >70% accuracy on key tasks
- **Documentation**: Training data, configuration used
- **Model size**: <100MB preferred for sharing
- **Game compatibility**: Specify which games it works with
- **Oversampling info**: Document the oversampling settings used

**Contribution process:**
1. **Train with good data**: Use the refactored training system
2. **Test thoroughly**: Verify performance on validation data
3. **Document configuration**: Include all `config.py` settings
4. **Package model**: Include model file and configuration
5. **Submit with examples**: Show the model in action

## 🔧 Development Workflow

### Testing Your Changes
```bash
# 1. Verify system setup
python scripts/2_verify_system_setup.py

# 2. Test data recording
python scripts/3_record_data.py  # Record small test dataset

# 3. Test data analysis
python scripts/4_analyze_data_quality.py

# 4. Test training
python scripts/5_train_model.py

# 5. Test inference
python scripts/6_run_inference.py

# 6. Test debugging
python scripts/7_debug_model_output.py
```

### Code Quality Checks
```bash
# Format code
black scripts/ config.py

# Check for issues
flake8 scripts/ config.py

# Run configuration validation
python -c "from config import validate_config; validate_config()"
```

### Performance Testing
- **Test on CPU**: Ensure CPU fallback works
- **Test on GPU**: Verify GPU acceleration (if available)
- **Memory profiling**: Check for memory leaks
- **Speed benchmarking**: Compare performance with previous versions

## 📋 Contribution Checklist

Before submitting a contribution:

### For Code Changes
- [ ] **Follows project structure**: Uses centralized config and consistent patterns
- [ ] **Includes tests**: New functionality has appropriate tests
- [ ] **Updates documentation**: README, config docs, or troubleshooting updated
- [ ] **Handles errors**: Graceful error handling with informative messages
- [ ] **Maintains compatibility**: Works with existing features
- [ ] **Validates configuration**: New settings work with config validation

### For Documentation
- [ ] **Clear and accurate**: Information is correct and easy to understand
- [ ] **Includes examples**: Code examples and command-line instructions
- [ ] **Cross-referenced**: Links to related documentation
- [ ] **Tested**: Instructions have been verified to work
- [ ] **Version appropriate**: Content matches current project version

### For Datasets/Models
- [ ] **High quality**: Good key press rate and diverse actions
- [ ] **Well documented**: Clear description of contents and usage
- [ ] **Legal**: Proper ownership and licensing
- [ ] **Tested**: Verified to work with current system
- [ ] **Optimized**: Appropriate size and format

## 🎯 Current Development Priorities

### High Priority
- **Performance optimization**: Faster training and inference
- **Better oversampling**: Improved class balancing techniques
- **Game compatibility**: Support for more game types
- **Error handling**: More robust error recovery

### Medium Priority
- **Model architectures**: Alternative neural network designs
- **Data augmentation**: Techniques to improve training data
- **GUI interface**: User-friendly graphical interface
- **Multi-game support**: Training on multiple games simultaneously

### Low Priority
- **Mobile support**: Android/iOS compatibility
- **Cloud training**: Distributed training capabilities
- **Real-time learning**: Online learning during gameplay
- **Advanced analytics**: Detailed performance metrics

## 🤝 Community Guidelines

### Communication
- **Be respectful**: Treat all contributors with respect
- **Be helpful**: Provide constructive feedback and assistance
- **Be patient**: Some issues may take time to resolve
- **Be clear**: Use clear, concise language in discussions

### Code Review
- **Be thorough**: Review code carefully and thoughtfully
- **Be constructive**: Provide helpful, actionable feedback
- **Be timely**: Respond to PRs and issues promptly
- **Be educational**: Explain why changes are suggested

## 📞 Getting Help

### For Contributors
- **Check existing issues**: Your question might already be answered
- **Use the discussion board**: For general questions and ideas
- **Join the community**: Connect with other contributors
- **Read the docs**: Check the documentation first

### For Users
- **Check troubleshooting**: Many issues have documented solutions
- **Provide details**: Include system info and error messages
- **Be specific**: Describe exactly what you're trying to do
- **Be patient**: Community members help in their free time

---

**Thank you for contributing to AI Game Automation!** Your contributions help make this project better for everyone. 🚀 