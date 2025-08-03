@echo off
REM AI Game Automation Installation Script for Windows
REM This script sets up the environment for AI Game Automation

echo 🚀 AI Game Automation v2.0 - Installation Script
echo ==================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% detected

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install the package in development mode
echo 📥 Installing AI Game Automation package...
pip install -e .

REM Detect GPU and install appropriate PyTorch
echo 🔍 Detecting GPU...
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo ✅ NVIDIA GPU detected. Installing CUDA PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo ✅ Installing DirectML PyTorch for Windows...
    pip install torch-directml
)

REM Install development tools (optional)
set /p DEV_TOOLS="🤔 Install development tools (black, ruff, flake8)? [y/N]: "
if /i "%DEV_TOOLS%"=="y" (
    echo 🛠️  Installing development tools...
    pip install -e .[dev]
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "data_human\frames" mkdir "data_human\frames"
if not exist "game_model" mkdir "game_model"
if not exist "runs" mkdir "runs"

echo.
echo ✅ Installation complete!
echo.
echo 🎯 Next steps:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Record training data: python 1_record_data.py
echo 3. Train the model: python 2_train_model.py
echo 4. Run inference: python 3_run_inference.py
echo.
echo 📚 For more information, see README.md
echo.
echo Happy gaming! 🎮
pause 