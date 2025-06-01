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
import wandb

# Local imports
from config import load_config, ExperimentConfig
from model import create_audio_albert_model
from dataset import load_dataset_from_directories, collate_fn, BalancedAudioDataset
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

def load_custom_dataset(event_dir, noise_dir):
    """Load audio event and noise files into a dataset."""
    event_files = glob.glob(os.path.join(event_dir, "*.wav"))
    noise_files = glob.glob(os.path.join(noise_dir, "*.wav"))
    audio_paths = event_files + noise_files
    labels = [1] * len(event_files) + [0] * len(noise_files)
    dataset = Dataset.from_dict({"audio": audio_paths, "label": labels})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset.train_test_split(test_size=0.2)

def get_noise_files(noise_dir):
    return glob.glob(os.path.join(noise_dir, "*.wav"))

def apply_augmentations(audio, noise_files, sample_rate=16000):
    """Apply random gain, additive noise, and optional pitch shift."""
    gain = random.uniform(0.5, 1.5)
    audio = audio * gain

    if random.random() < 0.5 and noise_files:
        noise_path = random.choice(noise_files)
        noise, sr = torchaudio.load(noise_path)
        if sr != sample_rate:
            noise = torchaudio.functional.resample(noise, sr, sample_rate)
        if noise.shape[1] > sample_rate:
            start = random.randint(0, noise.shape[1] - sample_rate)
            noise = noise[:, start:start+sample_rate]
        else:
            pad_length = sample_rate - noise.shape[1]
            noise = torch.nn.functional.pad(noise, (0, pad_length), "constant", 0)
        noise_level = random.uniform(0.1, 0.5)
        audio = audio + (noise * noise_level)

    if random.random() < 0.2:
        n_steps = random.randint(-2, 2)
        audio = torchaudio.functional.pitch_shift(audio, sample_rate, n_steps)

    return audio

def preprocess_with_augmentations(examples, feature_extractor, noise_files):
    """Preprocess audio with augmentations and feature extraction."""
    audio = examples["audio"]["array"]
    sample_rate = examples["audio"]["sampling_rate"]

    if len(audio) > sample_rate:
        start = random.randint(0, len(audio) - sample_rate)
        audio = audio[start:start+sample_rate]
    else:
        pad_length = sample_rate - len(audio)
        audio = np.pad(audio, (0, pad_length), 'constant')

    audio = torch.tensor(audio, dtype=torch.float32)
    augmented_audio = apply_augmentations(audio, noise_files)

    inputs = feature_extractor(
        augmented_audio.numpy(),
        sampling_rate=sample_rate,
        max_length=16000,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )

    examples["input_values"] = inputs["input_values"][0]
    return examples

def compute_metrics(eval_pred):
    """Compute evaluation metrics for binary classification."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load("accuracy")
    precision = load("precision")
    recall = load("recall")
    f1 = load("f1")
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels)["precision"],
        "recall": recall.compute(predictions=predictions, references=labels)["recall"],
        "f1": f1.compute(predictions=predictions, references=labels)["f1"]
    }

def predict_audio_event(audio_path, model, feature_extractor):
    """Predict if an audio clip contains an event."""
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if audio.shape[1] > 16000:
        audio = audio[:, :16000]
    else:
        pad_length = 16000 - audio.shape[1]
        audio = torch.nn.functional.pad(audio, (0, pad_length), "constant", 0)

    inputs = feature_extractor(
        audio[0].numpy(),
        sampling_rate=16000,
        max_length=16000,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    return model.config.id2label[predicted_id]

def main():
    # Set paths
    event_dir = "path/to/events"  # Replace with actual path
    noise_dir = "path/to/noise"   # Replace with actual path

    # Prepare dataset and noise files
    dataset = load_custom_dataset(event_dir, noise_dir)
    noise_files = get_noise_files(noise_dir)

    # Feature extractor and model
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=2,
        label2id={"no_event": 0, "event": 1},
        id2label={0: "no_event", 1: "event"}
    )

    # Preprocessing functions
    def train_transform(examples):
        return preprocess_with_augmentations(examples, feature_extractor, noise_files)

    train_dataset = dataset["train"].with_transform(train_transform)
    eval_dataset = dataset["test"].map(
        lambda x: preprocess_with_augmentations(x, feature_extractor, noise_files),
        remove_columns=["audio"]
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./audio_event_detector",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Example inference
    # result = predict_audio_event("path/to/new_audio.wav", model, feature_extractor)
    # print(f"Prediction: {result}")

if __name__ == "__main__":
    main()