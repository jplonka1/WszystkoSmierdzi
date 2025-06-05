#!/usr/bin/env python3
"""
System Validation Script
=======================

This script validates that the enhanced audio event detection system is properly installed
and configured, and runs basic functionality tests.
"""

import os
import sys
import logging
import tempfile
import numpy as np
import torch
import torchaudio
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)



def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        # Core modules
        from config import load_config, ExperimentConfig
        from model import create_model
        from dataset import AudioDataset, load_dataset_from_directories
        from utils import compute_metrics, setup_logging
        
        logger.info("✅ All core modules imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False


def test_config_system():
    """Test configuration loading and validation."""
    logger.info("Testing configuration system...")
    
    try:
        from config import load_config, ExperimentConfig
        
        # Test default config loading
        config = load_config()
        assert hasattr(config, 'audio')
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'data')
        
        # Test config properties access
        assert config.audio.sample_rate == 16000
        assert config.model.model_name == "albert-base-v2"
        assert config.training.num_train_epochs > 0
        assert hasattr(config.data, 'event_dir')
        assert hasattr(config.data, 'background_dir')
        assert hasattr(config.data, 'noise_dir')
        
        logger.info("✅ Configuration system working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configurationf test failed: {e}")
        return False


def test_model_creation():
    """Test model creation and basic forward pass."""
    logger.info("Testing model creation...")
    
    try:
        from config import load_config
        from model import create_model
        
        config = load_config()
        model = create_model(config)
        
        # Test model structure
        assert hasattr(model, 'audio_feature_extractor')
        assert hasattr(model, 'albert')
        assert hasattr(model, 'feature_projection')
        
        # Test forward pass with dummy data
        batch_size = 2
        audio_length = config.audio.sample_rate * 2  # 2 seconds of audio
        
        dummy_input = torch.randn(batch_size, audio_length)
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_values=dummy_input)
            logits = outputs['logits']
            
        assert logits.shape == (batch_size, config.model.num_labels)
        
        logger.info("✅ Model creation and forward pass successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False


def test_audio_processing():
    """Test audio loading and preprocessing."""
    logger.info("Testing audio processing...")
    
    try:
        from config import load_config
        from dataset import AudioDataset
        
        config = load_config()
        
        # Create dummy audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            dummy_audio_path = f.name
        
        # Generate dummy audio
        sample_rate = config.audio.sample_rate
        duration = 2.0  # seconds
        frequency = 440  # Hz (A note)
        
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * np.pi * frequency * t).unsqueeze(0)
        
        torchaudio.save(dummy_audio_path, audio, sample_rate)
        
        # Test dataset creation
        dataset = AudioDataset(
            audio_paths=[dummy_audio_path],
            config=config,
            is_training=False
        )
        
        # Test data loading
        sample = dataset[0]
        assert 'input_values' in sample
        assert 'labels' in sample
        assert len(sample['input_values'].shape) == 1  # Raw audio should be 1D
        
        # Cleanup
        os.unlink(dummy_audio_path)
        
        logger.info("✅ Audio processing working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Audio processing test failed: {e}")
        return False


def test_system_info():
    """Display system information."""
    logger.info("System Information:")
    
    # Python version
    logger.info(f"  Python: {sys.version}")
    
    # PyTorch info
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # CPU info
    logger.info(f"  CPU threads: {torch.get_num_threads()}")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"  RAM: {memory.total / 1e9:.1f} GB total, {memory.available / 1e9:.1f} GB available")
    except ImportError:
        logger.info("  RAM: psutil not available for memory info")


def check_data_directories():
    """Check if data directories exist and contain files."""
    logger.info("Checking data directories...")
    
    directories = {
        "Event data": "dataset_preliminary",
        "Background data": "Background",
        "Noise data": "noise"
    }
    
    for name, path in directories.items():
        if os.path.exists(path):
            wav_files = list(Path(path).rglob("*.wav"))
            logger.info(f"  ✅ {name}: {path} ({len(wav_files)} .wav files)")
        else:
            logger.info(f"  ⚠️  {name}: {path} (not found)")


def main():
    """Run all validation tests."""
    logger.info("🔍 Running Enhanced Audio Event Detection System Validation")
    logger.info("=" * 60)
    
    # Display system info
    test_system_info()
    logger.info("-" * 60)
    
    # Check data directories
    check_data_directories()
    logger.info("-" * 60)
    
    # Run tests
    tests = [
        ("Module Imports", test_imports),
        ("Configuration System", test_config_system),
        ("Model Creation", test_model_creation),
        ("Audio Processing", test_audio_processing),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
        logger.info("-" * 60)
    
    # Summary
    logger.info("📋 Validation Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready for use.")
        logger.info("\n📋 Next steps:")
        logger.info("1. Run: python3 training_enhanced.py --help")
        logger.info("2. Check the README_ENHANCED.md for detailed usage instructions")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
