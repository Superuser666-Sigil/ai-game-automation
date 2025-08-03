#!/bin/bash

# AI Game Automation Installation Script
# This script sets up the environment for AI Game Automation

set -e  # Exit on any error

echo "🚀 AI Game Automation v2.0 - Installation Script"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION detected. Python 3.8 or higher is required."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "📥 Installing AI Game Automation package..."
pip install -e .

# Detect GPU and install appropriate PyTorch
echo "🔍 Detecting GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected. Installing CUDA PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "✅ Windows detected. Installing DirectML PyTorch..."
    pip install torch-directml
else
    echo "ℹ️  No GPU detected or unsupported platform. Installing CPU PyTorch..."
    pip install torch torchvision
fi

# Install development tools (optional)
read -p "🤔 Install development tools (black, ruff, flake8)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🛠️  Installing development tools..."
    pip install -e .[dev]
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data_human/frames
mkdir -p game_model
mkdir -p runs

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Record training data: python 1_record_data.py"
echo "3. Train the model: python 2_train_model.py"
echo "4. Run inference: python 3_run_inference.py"
echo ""
echo "📚 For more information, see README.md"
echo ""
echo "Happy gaming! 🎮" 