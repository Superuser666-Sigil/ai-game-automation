# 📁 Examples Directory

This directory contains example files and sample data to help you get started with AI Game Automation.

## 📦 Contents

### `sample_recordings/`
Example training data for testing and learning:
- **Small dataset**: 100 frames for quick testing
- **Medium dataset**: 1000 frames for basic training
- **Game-specific examples**: Recordings from different game types

### `trained_models/`
Pre-trained models for demonstration:
- **Puzzle game model**: Trained on simple 2D puzzle games
- **Basic movement model**: Learns WASD movement patterns
- **Mouse tracking model**: Demonstrates smooth mouse control

## 🎯 How to Use Examples

### Testing with Sample Data

1. **Download sample recordings** (when available):
   ```bash
   # Copy sample data to your main directory
   cp examples/sample_recordings/* .
   ```

2. **Analyze the sample data**:
   ```bash
   cd scripts
   python 4_analyze_data_quality.py
   ```

3. **Train on sample data**:
   ```bash
   python 4.5_choose_model_configuration.py
   python 5_train_model.py
   ```

### Using Pre-trained Models

1. **Copy a pre-trained model**:
   ```bash
   cp examples/trained_models/basic_movement.pt scripts/model_improved.pt
   ```

2. **Test the model**:
   ```bash
   cd scripts
   python 7_debug_model_output.py
   python 6_run_inference.py
   ```

## 📝 Example Datasets

### Puzzle Game Dataset
- **Game type**: 2D puzzle game
- **Resolution**: 640x360
- **Duration**: 5 minutes
- **Actions**: Arrow keys, space bar
- **Complexity**: Simple, good for beginners

### Strategy Game Dataset  
- **Game type**: Real-time strategy
- **Resolution**: 960x540
- **Duration**: 10 minutes
- **Actions**: Mouse clicks, hotkeys 1-9, WASD
- **Complexity**: Medium, includes complex mouse patterns

### FPS Game Dataset
- **Game type**: First-person shooter
- **Resolution**: 1280x720
- **Duration**: 3 minutes
- **Actions**: WASD, mouse movement, mouse clicks, R, G, B
- **Complexity**: High, fast-paced gameplay

## 🎮 Game-Specific Examples

### For Puzzle Games
```bash
# Use the puzzle game configuration
cp examples/configs/puzzle_game_config.py scripts/
python scripts/5_train_model.py
```

### For Strategy Games
```bash
# Use the strategy game configuration
cp examples/configs/strategy_game_config.py scripts/
python scripts/5_train_model.py
```

## 📊 Expected Results

### Training Metrics from Examples

| Dataset | Training Time | Final Loss | Key Accuracy | Mouse Accuracy |
|---------|---------------|------------|--------------|----------------|
| Puzzle  | 5-15 min      | 0.12       | 85%          | 78%            |
| Strategy| 15-45 min     | 0.18       | 72%          | 82%            |
| FPS     | 20-60 min     | 0.25       | 68%          | 88%            |

## 🔄 Creating Your Own Examples

### Recording Sample Data

1. **Record short sessions** (2-5 minutes):
   ```bash
   cd scripts
   python 3_record_data.py
   ```

2. **Package for sharing**:
   ```bash
   # Create a zip file with your recording
   7z a my_game_dataset.zip frames/ actions.npy
   ```

3. **Document your dataset**:
   - Game name and type
   - Recording duration
   - Key actions used
   - Screen resolution
   - Any special notes

### Contributing Examples

We welcome contributions of:
- **Sample datasets** from different games
- **Pre-trained models** with good performance  
- **Configuration files** for specific games
- **Documentation** of your results

See [CONTRIBUTING.md](../docs/CONTRIBUTING.md) for guidelines.

## ⚠️ Legal Notice

- Only share recordings from games you own
- Respect game developers' terms of service
- Don't include copyrighted content in recordings
- Focus on gameplay mechanics, not story content

---

**Need help?** Check the main [README.md](../README.md) or [Troubleshooting Guide](../docs/TROUBLESHOOTING.md). 