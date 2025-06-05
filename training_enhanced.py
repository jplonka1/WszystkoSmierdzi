#!/usr/bin/env python3
"""
Enhanced Audio Event Detection Training Script with ALBERT-based Architecture
===========================================================================

This script implements a comprehensive audio event detection system using an ALBERT-based
transformer architecture adapted for audio classification tasks.

Key Features:
- ALBERT transformer backbone with audio feature extraction
- Advanced data augmentation pipeline
- Comprehensive evaluation metrics
- Early stopping and model checkpointing
- Detailed logging and visualization

Author: Enhanced by AI Assistant
Date: 2025
"""

import os
import sys
import logging
import random
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments, 
    Trainer, 
    TrainerCallback,
    set_seed
)

# Local imports
from config import load_config, ExperimentConfig
from model import create_model
from dataset import load_dataset_from_directories, collate_fn, AudioDataset
from utils import (
    compute_metrics, 
    evaluate_model_comprehensive,
    EarlyStopping,
    setup_logging,
    log_system_info,
    plot_training_history
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class MetricsCallback(TrainerCallback):# idk whats that fro yet, yet to do even chatgpt of it
    """Custom callback to track training metrics."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = {
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'roc_auc': []
        }
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):# idk whats that fro yet, yet to do even chatgpt of it
        if logs:
            if 'train_loss' in logs:
                self.train_losses.append(logs['train_loss'])
            
            if 'eval_loss' in logs:
                self.val_losses.append(logs['eval_loss'])
            
            for metric_name in self.val_metrics:
                if f'eval_{metric_name}' in logs:
                    self.val_metrics[metric_name].append(logs[f'eval_{metric_name}'])


def setup_directories(config: ExperimentConfig) -> Dict[str, str]:#should be ok
    """Setup output directories."""
    
    output_dir = config.training.output_dir
    dirs = {
        'output': output_dir,
        'models': os.path.join(output_dir, 'models'),
        'logs': os.path.join(output_dir, 'logs'),
        'plots': os.path.join(output_dir, 'plots'),
        'cache': config.data.cache_dir
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def predict_audio_event(
    audio_path: str,
    model_path: str,
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Predict audio event for a single file."""
    
    # Load config
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config)
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
    model.to(device)
    model.eval()
    
    # Create temporary dataset for single file #TODO: completely redo the next lines
    temp_dataset = AudioDataset([audio_path], [0], config, is_training=False)
    
    # Get features
    sample = temp_dataset[0]
    input_values = sample['input_values'].unsqueeze(0).to(device)
    
    # Predict #up to here, and TODO evaluate whats below cause I didnt yet
    with torch.no_grad():
        outputs = model(input_values=input_values)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    result = {
        'predicted_class': predicted_class,
        'predicted_label': config.data.id2label[predicted_class],
        'probabilities': {
            config.data.id2label[i]: float(prob) 
            for i, prob in enumerate(probabilities[0])
        },
        'confidence': float(probabilities[0].max())
    }
    
    return result


def create_trainer(#looks good for now
    model,
    train_dataset,
    eval_dataset,
    config: ExperimentConfig,
    output_dirs: Dict[str, str]
) -> Trainer:
    """Create and configure the Trainer."""
    
    # Training arguments
    print("YO")
    training_args = TrainingArguments(
        output_dir=output_dirs['models'],
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        
        # Evaluation
        eval_strategy=config.training.evaluation_strategy,
        eval_steps=None,  # Evaluate every epoch
        
        # Saving
        save_strategy=config.training.save_strategy,
        save_steps=None,  # Save every epoch
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        
        # Logging
        logging_dir=output_dirs['logs'],
        logging_steps=config.training.logging_steps,
        logging_strategy=config.training.logging_strategy,
        report_to=config.training.report_to or [],
        
        # Performance
        fp16=config.training.fp16,
        dataloader_num_workers=config.training.dataloader_num_workers,
        
        # Other
        seed=config.training.seed,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    print("YoO")
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    
    return trainer


def train_model(config: ExperimentConfig) -> Dict[str, Any]:
    """Main training function."""
    
    # Setup
    setup_logging("INFO", log_file=os.path.join(config.training.output_dir, "logs", "training.log"))
    set_seed(config.training.seed)
    
    logger.info("Starting audio event detection training")
    #logger.info(f"Configuration: {config}")
    log_system_info()
    
    # Setup directories
    output_dirs = setup_directories(config)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_dataset_from_directories(config)

    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())#numel?xd
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)#smart i guess
    logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Create trainer
    print("yo")
    trainer = create_trainer(model, train_dataset, val_dataset, config, output_dirs)
    print("YOOOOOOOOOO")
    # Add custom callback
    metrics_callback = MetricsCallback()
    trainer.add_callback(metrics_callback)
    
    # Train
    logger.info("Starting training...")
    try:
        train_result = trainer.train()
        
        # Save final model
        trainer.save_model()
        logger.info(f"Model saved to {output_dirs['models']}")
        
        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.training.per_device_eval_batch_size,
            collate_fn=collate_fn,
            num_workers=config.training.dataloader_num_workers
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        test_metrics = evaluate_model_comprehensive(
            model=model,
            eval_dataloader=test_dataloader,
            device=device,
            class_names=list(config.data.id2label.values()),
            output_dir=output_dirs['plots']
        )
        
        # Plot training history
        if metrics_callback.train_losses:
            plot_path = os.path.join(output_dirs['plots'], "training_history.png")
            plot_training_history(
                train_losses=metrics_callback.train_losses,
                val_losses=metrics_callback.val_losses,
                val_metrics=metrics_callback.val_metrics,
                save_path=plot_path
            )
        
        results = {
            'train_result': train_result,
            'test_metrics': test_metrics,
            'model_path': output_dirs['models'],
            'output_dir': output_dirs['output']
        }
        
        logger.info("Training completed successfully!")
        logger.info(f"Final test metrics: {test_metrics}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Audio Event Detection Training")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--mode", choices=["train", "predict"], default="train", help="Mode: train or predict")
    parser.add_argument("--model-path", type=str, help="Path to trained model (for prediction)")
    parser.add_argument("--audio-path", type=str, help="Path to audio file (for prediction)")
    parser.add_argument("--event-dir", type=str, help="Directory containing event audio files")
    parser.add_argument("--noise-dir", type=str, help="Directory containing noise audio files")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.event_dir:
        config.data.event_dir = args.event_dir
    if args.noise_dir:
        config.data.noise_dir = args.noise_dir
    
    if args.mode == "train":
        # Validate required directories
        if not os.path.exists(config.data.event_dir):
            logger.error(f"Event directory not found: {config.data.event_dir}")
            sys.exit(1)
        if not os.path.exists(config.data.noise_dir):
            logger.error(f"Noise directory not found: {config.data.noise_dir}")
            sys.exit(1)
        if not os.path.exists(config.data.background_dir):
            logger.error(f"Event directory not found: {config.data.background_dir}")
            sys.exit(1)
        
        # Train model
        results = train_model(config)
        print(f"Training completed. Results saved to: {results['output_dir']}")
        
    elif args.mode == "predict":
        if not args.model_path or not args.audio_path:
            logger.error("Model path and audio path required for prediction")
            sys.exit(1)
        
        # Make prediction
        result = predict_audio_event(args.audio_path, args.model_path, args.config)
        print(f"Prediction for {args.audio_path}:")
        print(f"  Class: {result['predicted_label']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Probabilities: {result['probabilities']}")


if __name__ == "__main__":
    main()
