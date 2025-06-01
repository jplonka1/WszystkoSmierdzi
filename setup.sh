#!/bin/bash

# Enhanced Audio Event Detection Setup Script
# ==========================================

set -e

echo "🔊 Setting up Enhanced Audio Event Detection System..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Python 3.8+ required. Found: Python $python_version"
    exit 1
fi

echo "✅ Python version check passed"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models/audio_albert_classifier
mkdir -p cache
mkdir -p logs
mkdir -p plots

# Check GPU availability
echo "🔍 Checking GPU availability..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('Running on CPU')
"

# Validate dataset directories
echo "📊 Checking dataset directories..."
if [ -d "dataset_current/training" ]; then
    echo "✅ Found event dataset: dataset_current/training"
    event_count=$(find dataset_current/training -name "*.wav" | wc -l)
    echo "   Event files: $event_count"
else
    echo "⚠️  Event directory not found: dataset_current/training"
    echo "   Please ensure your event audio files are in this directory"
fi

if [ -d "Background" ]; then
    echo "✅ Found noise dataset: Background"
    noise_count=$(find Background -name "*.wav" | wc -l)
    echo "   Noise files: $noise_count"
else
    echo "⚠️  Noise directory not found: Background"
    echo "   Please ensure your background audio files are in this directory"
fi

# Test import of main modules
echo "🧪 Testing module imports..."
python3 -c "
try:
    from config import load_config
    from model import create_audio_albert_model
    from dataset import load_dataset_from_directories
    from utils import compute_metrics
    print('✅ All modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Ensure your audio files are in the correct directories:"
echo "   - Event files: dataset_current/training/"
echo "   - Background files: Background/"
echo ""
echo "2. Start training with:"
echo "   python3 training_enhanced.py --event-dir dataset_current/training --noise-dir Background"
echo ""
echo "3. Or use the original training script:"
echo "   python3 training.py"
echo ""
echo "4. For evaluation:"
echo "   python3 evaluate.py --model-path models/audio_albert_classifier --data-dir your_test_data/"
echo ""
echo "5. For quick prediction:"
echo "   python3 training_enhanced.py --mode predict --model-path models/audio_albert_classifier --audio-path your_audio.wav"
echo ""
echo "📖 See README_ENHANCED.md for detailed documentation"
