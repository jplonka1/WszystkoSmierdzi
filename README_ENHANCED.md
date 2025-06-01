# Enhanced Audio Event Detection System

## 🎯 Overview

Production-ready audio event detection system using **ALBERT-base-v2** transformer with mel-spectrogram preprocessing. Detects audio events from background noise with comprehensive training infrastructure and evaluation tools.

## ✅ System Status: **FULLY OPERATIONAL**

- **Model**: ALBERT-base-v2 with mel-spectrogram features (128-dim)
- **Data**: 838 event samples + 7,464 background samples  
- **Validation**: 4/4 tests passing ✅
- **Ready**: Training, evaluation, and deployment ready

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
# Simple training (recommended)
python3 training_simple.py

# Advanced training with full features
python3 training_enhanced.py --event-dir dataset_current/training --noise-dir Background

# Custom configuration
python3 training_enhanced.py --config custom_config.json
```

### Validation
```bash
python3 validate_system.py  # Verify system integrity
```

## 📁 Core Components

```
📦 Enhanced Audio Event Detection
├── 🤖 AI/ML Core
│   ├── model.py              # ALBERT-based audio classifier
│   ├── dataset.py            # Audio preprocessing pipeline
│   ├── utils.py              # Metrics and evaluation tools
│   └── config.py             # Configuration management
│
├── 🎓 Training Scripts
│   ├── training_simple.py    # Quick start training
│   ├── training_enhanced.py  # Full-featured training
│   └── validate_system.py    # System validation
│
├── 📊 Data & Config
│   ├── config_default.json   # Default settings
│   ├── requirements.txt      # Dependencies
│   ├── dataset_current/      # Event samples (838 files)
│   └── Background/           # Noise samples (7,464 files)
│
└── 📖 Documentation
    ├── SYSTEM_SUMMARY.md     # Quick overview
    └── README_ENHANCED.md    # This file
```

## ⚙️ Key Features

- **ALBERT-base-v2**: Efficient transformer architecture
- **Mel-spectrograms**: Advanced audio feature extraction
- **Data augmentation**: Gain, noise, pitch shift
- **Early stopping**: Prevent overfitting
- **Comprehensive metrics**: Accuracy, precision, recall, F1, ROC-AUC
- **GPU support**: CUDA optimization when available

## 🔧 Configuration

The system uses JSON-based configuration. Key sections:

```json
{
  "audio": {
    "sample_rate": 16000,
    "max_duration": 10.0,
    "n_mels": 128
  },
  "model": {
    "model_name": "albert-base-v2",
    "num_classes": 2
  },
  "training": {
    "num_train_epochs": 20,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16
  }
}
```

## 📈 Model Architecture

```
Audio Input (16kHz WAV)
    ↓
Mel-Spectrogram Extraction (128-dim)
    ↓
Feature Projection Layer
    ↓
ALBERT Transformer (albert-base-v2)
    ↓
Classification Head
    ↓
Event/No-Event Prediction
```

## 🎛️ Advanced Features

### Data Processing
- **Mel-spectrograms**: 128 mel-frequency bins
- **Audio augmentation**: Gain, noise, pitch variations
- **Caching**: Preprocessed feature caching for speed
- **Balanced sampling**: Handle imbalanced datasets

### Training Infrastructure
- **Early stopping**: Configurable patience and criteria
- **Model checkpointing**: Save best performing models
- **Metrics tracking**: Comprehensive evaluation metrics
- **Logging**: Detailed training progress logs

### Evaluation Tools
- **Confusion matrices**: Visual performance analysis
- **Per-class metrics**: Detailed breakdown by class
- **Training visualization**: Loss and metric plots
- **Model comparison**: Compare multiple trained models

## 🔍 Usage Examples

### Custom Training
```python
from config import ExperimentConfig
from training_enhanced import train_model

config = ExperimentConfig.from_json("config_default.json")
config.training.learning_rate = 1e-5
results = train_model(config)
```

### Model Evaluation
```python
from utils import evaluate_model_comprehensive

metrics = evaluate_model_comprehensive(
    model=model,
    eval_dataloader=dataloader,
    device=device,
    output_dir="results/"
)
```

## 🐛 Troubleshooting

### Common Issues
1. **Memory errors**: Reduce batch size in config
2. **Audio format**: Ensure 16kHz WAV files
3. **CUDA issues**: Check GPU availability with `python3 validate_system.py`

### Performance Tips
- Use GPU when available for faster training
- Enable caching for repeated runs
- Adjust batch size based on available memory
- Use early stopping to prevent overfitting

## 📊 System Validation

Run validation to ensure everything works:
```bash
python3 validate_system.py
```

Expected output:
```
✅ Module Imports: PASSED
✅ Configuration System: PASSED  
✅ Model Creation: PASSED
✅ Audio Processing: PASSED
Overall: 4/4 tests passed
```

## 🔄 Migration Notes

Enhanced system improvements over original:
- **Model**: wav2vec2 → ALBERT-base-v2
- **Features**: Basic audio → Mel-spectrograms
- **Training**: Simple loop → Professional pipeline
- **Config**: Hardcoded → JSON-based system
- **Evaluation**: Basic → Comprehensive metrics

## 📚 Key Dependencies

- PyTorch + transformers (Hugging Face)
- librosa (audio processing)
- sklearn (metrics)
- matplotlib + seaborn (visualization)
- See `requirements.txt` for complete list

---

**Status**: Production ready • **Tests**: 4/4 passing • **Last updated**: June 2025
