#!/usr/bin/env python3
"""
Setup script for AI Game Automation
A behavior cloning system for game automation
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "AI Game Automation - Behavior cloning for game automation"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="ai-game-automation",
    version="2.0.0",
    author="AI Game Automation Team",
    description="Behavior cloning system for game automation using CNN-LSTM architecture",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-game-automation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "black>=23.0.0",
            "ruff>=0.1.0",
            "flake8>=6.0.0",
            "pytest>=7.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "record-data=ai_game_automation.record_data:main",
            "train-model=ai_game_automation.train_model:main",
            "run-inference=ai_game_automation.run_inference:main",
            "debug-model=ai_game_automation.debug_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ai_game_automation": ["*.py", "*.md", "*.txt"],
    },
    keywords="ai automation behavior-cloning gaming neural-networks pytorch",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ai-game-automation/issues",
        "Source": "https://github.com/yourusername/ai-game-automation",
        "Documentation": "https://github.com/yourusername/ai-game-automation#readme",
    },
) 