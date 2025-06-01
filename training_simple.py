#!/usr/bin/env python3
"""
Simple ALBERT-based Audio Event Detection Training Script
=========================================================

A simplified training script using the standard ALBERT model for audio classification.
This script provides a straightforward approach to training without complex configurations.
"""

import os
import glob
import random
import numpy as np
import torch
import torchaudio
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from typing import Dict, List

from model import AudioALBERTClassifier
from config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_audio_files(event_dir: str, noise_dir: str) -> tuple:
    """Load audio files and create labels."""
    event_files = glob.glob(os.path.join(event_dir, "*.wav"))
    noise_files = glob.glob(os.path.join(noise_dir, "*.wav"))
    
    all_files = event_files + noise_files
    all_labels = [1] * len(event_files) + [0] * len(noise_files)
    
    logger.info(f"Loaded {len(event_files)} event files and {len(noise_files)} noise files")
    return all_files, all_labels


def preprocess_audio(audio_path: str, sample_rate: int = 16000, max_duration: float = 10.0) -> torch.Tensor:
    """Load and preprocess audio file."""
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    
    # Fix duration
    target_length = int(max_duration * sample_rate)
    current_length = audio.shape[1]
    
    if current_length > target_length:
        # Random crop during training
        start = random.randint(0, current_length - target_length)
        audio = audio[:, start:start + target_length]
    elif current_length < target_length:
        # Pad if shorter
        pad_length = target_length - current_length
        audio = torch.nn.functional.pad(audio, (0, pad_length), "constant", 0)
    
    return audio.squeeze(0)  # Return 1D tensor


class SimpleAudioDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset for audio classification."""
    
    def __init__(self, audio_paths: List[str], labels: List[int], is_training: bool = True):
        self.audio_paths = audio_paths
        self.labels = labels
        self.is_training = is_training
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        try:
            audio = preprocess_audio(audio_path)
            
            # Simple augmentation during training
            if self.is_training and random.random() < 0.5:
                # Random gain
                gain = random.uniform(0.7, 1.3)
                audio = audio * gain
            
            return {
                'input_values': audio,
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            # Return dummy data
            dummy_audio = torch.zeros(160000)  # 10 seconds at 16kHz
            return {
                'input_values': dummy_audio,
                'labels': torch.tensor(0, dtype=torch.long)
            }


def split_dataset(audio_paths: List[str], labels: List[int], train_ratio: float = 0.8):
    """Split dataset into train and validation."""
    # Shuffle data
    combined = list(zip(audio_paths, labels))
    random.shuffle(combined)
    audio_paths, labels = zip(*combined)
    audio_paths, labels = list(audio_paths), list(labels)
    
    # Split
    n_train = int(len(audio_paths) * train_ratio)
    
    train_paths = audio_paths[:n_train]
    train_labels = labels[:n_train]
    
    val_paths = audio_paths[n_train:]
    val_labels = labels[n_train:]
    
    return train_paths, train_labels, val_paths, val_labels


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average='weighted', zero_division=0),
        "recall": recall_score(labels, predictions, average='weighted', zero_division=0),
        "f1": f1_score(labels, predictions, average='weighted', zero_division=0)
    }


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    input_values = torch.stack([item['input_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_values': input_values,
        'labels': labels
    }


def main():
    """Main training function."""
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration - you can modify these paths
    event_dir = "dataset_current/training"  # Directory with event audio files
    noise_dir = "Background"                # Directory with background/noise files
    output_dir = "./albert_audio_model"     # Where to save the trained model
    
    # Check if directories exist
    if not os.path.exists(event_dir):
        logger.error(f"Event directory not found: {event_dir}")
        return
    if not os.path.exists(noise_dir):
        logger.error(f"Noise directory not found: {noise_dir}")
        return
    
    # Load data
    logger.info("Loading audio files...")
    audio_paths, labels = load_audio_files(event_dir, noise_dir)
    
    if len(audio_paths) == 0:
        logger.error("No audio files found!")
        return
    
    # Split dataset
    train_paths, train_labels, val_paths, val_labels = split_dataset(audio_paths, labels)
    logger.info(f"Dataset split - Train: {len(train_paths)}, Validation: {len(val_paths)}")
    
    # Create datasets
    train_dataset = SimpleAudioDataset(train_paths, train_labels, is_training=True)
    val_dataset = SimpleAudioDataset(val_paths, val_labels, is_training=False)
    
    # Create model
    logger.info("Creating ALBERT model...")
    model = AudioALBERTClassifier(
        model_name="albert-base-v2",
        num_labels=2,
        sample_rate=16000,
        n_mels=128,
        max_length=512,
        freeze_albert=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        logging_steps=50,
        logging_strategy="steps",
        
        fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    
    # Train
    logger.info("Starting training...")
    try:
        trainer.train()
        
        # Save final model
        trainer.save_model()
        logger.info(f"Model saved to {output_dir}")
        
        # Final evaluation
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def predict_single_file(model_path: str, audio_path: str) -> Dict:
    """Predict on a single audio file."""
    # Load model
    model = AudioALBERTClassifier.from_pretrained(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Preprocess audio
    audio = preprocess_audio(audio_path)
    audio = audio.unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        outputs = model(input_values=audio)
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1).item()
    
    # Map to labels
    id2label = {0: "no_event", 1: "event"}
    
    return {
        'prediction': prediction,
        'label': id2label[prediction],
        'confidence': float(probabilities[0, prediction]),
        'probabilities': {
            'no_event': float(probabilities[0, 0]),
            'event': float(probabilities[0, 1])
        }
    }


if __name__ == "__main__":
    main()
