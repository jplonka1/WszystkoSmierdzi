import os
import glob
import random
import logging
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, Audio
from transformers import BatchFeature
import pandas as pd

from config import ExperimentConfig

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """Custom PyTorch dataset for audio classification."""
    
    def __init__(
        self,
        audio_paths: List[str],
        labels: List[int],
        config: ExperimentConfig,
        is_training: bool = True,
        cache_features: bool = True
    ):
        self.audio_paths = audio_paths
        self.labels = labels
        self.config = config
        self.is_training = is_training
        self.cache_features = cache_features
        
        # Cache for processed features
        self._feature_cache = {} if cache_features else None
        
        # Load noise files for augmentation
        self.noise_files = self._load_noise_files() if is_training else []
        
        logger.info(f"Created dataset with {len(self.audio_paths)} samples")
        logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    def _load_noise_files(self) -> List[str]:
        """Load noise files for augmentation."""
        if not os.path.exists(self.config.data.noise_dir):
            logger.warning(f"Noise directory not found: {self.config.data.noise_dir}")
            return []
        
        noise_files = glob.glob(os.path.join(self.config.data.noise_dir, "*.wav"))
        logger.info(f"Loaded {len(noise_files)} noise files for augmentation")
        return noise_files
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # Check cache first
        if self._feature_cache is not None and audio_path in self._feature_cache:
            audio_tensor = self._feature_cache[audio_path]
        else:
            # Load and process audio
            audio_tensor = self._load_and_process_audio(audio_path)
            
            # Cache if enabled
            if self._feature_cache is not None:
                self._feature_cache[audio_path] = audio_tensor
        
        # Apply augmentations if training
        if self.is_training and self.config.augmentation.enable:
            audio_tensor = self._apply_augmentations(audio_tensor)
        
        return {
            'input_values': audio_tensor,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def _load_and_process_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and return raw audio tensor."""
        try:
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            
            # Convert to mono if necessary
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.config.audio.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, sr, self.config.audio.sample_rate
                )
            
            # Normalize duration and return as 1D tensor
            audio = self._normalize_duration(audio)
            return audio.squeeze(0)  # Remove batch dimension, return 1D audio
            
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            # Return zeros as fallback
            duration = int(self.config.audio.max_duration * self.config.audio.sample_rate)
            return torch.zeros(duration)
    
    def _normalize_duration(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio duration to fixed length."""
        target_length = int(self.config.audio.max_duration * self.config.audio.sample_rate)
        current_length = audio.shape[1]
        
        if current_length > target_length:
            # Random crop if longer
            if self.is_training:
                start = random.randint(0, current_length - target_length)
            else:
                start = (current_length - target_length) // 2
            audio = audio[:, start:start + target_length]
        elif current_length < target_length:
            # Pad if shorter
            pad_length = target_length - current_length
            audio = torch.nn.functional.pad(audio, (0, pad_length), "constant", 0)
        
        return audio
    
    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel-spectrogram."""
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.audio.sample_rate,
            n_mels=self.config.audio.n_mels,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length,
        )
        
        mel_spec = mel_transform(audio)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-8)
        
        # Normalize if specified
        if self.config.audio.normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return mel_spec
    
    def _apply_augmentations(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply data augmentations to raw audio."""
        aug_config = self.config.augmentation
        
        # Volume/Gain augmentation
        if random.random() < 0.5:
            gain = random.uniform(*aug_config.gain_range)
            audio = audio * gain
        
        # Add background noise
        if random.random() < aug_config.noise_prob and self.noise_files:
            audio = self._add_background_noise(audio)
        
        # Pitch shift (if available)
        if random.random() < aug_config.pitch_shift_prob:
            audio = self._pitch_shift(audio)
        
        return audio
    
    def _add_background_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Add background noise to audio."""
        if not self.noise_files:
            return audio
        
        try:
            # Load random noise file
            noise_path = random.choice(self.noise_files)
            noise, sr = torchaudio.load(noise_path)
            
            # Convert to mono and resample if needed
            if noise.shape[0] > 1:
                noise = noise.mean(dim=0, keepdim=True)
            
            if sr != self.config.audio.sample_rate:
                noise = torchaudio.functional.resample(
                    noise, sr, self.config.audio.sample_rate
                )
            
            noise = noise.squeeze(0)  # Remove batch dimension
            
            # Match lengths
            if len(noise) > len(audio):
                start = random.randint(0, len(noise) - len(audio))
                noise = noise[start:start + len(audio)]
            elif len(noise) < len(audio):
                # Repeat noise if too short
                repeats = (len(audio) // len(noise)) + 1
                noise = noise.repeat(repeats)[:len(audio)]
            
            # Random mixing level
            noise_level = random.uniform(*self.config.augmentation.noise_level_range)
            
            # Mix audio with noise
            mixed_audio = (1 - noise_level) * audio + noise_level * noise
            return mixed_audio
            
        except Exception as e:
            logger.warning(f"Failed to add background noise: {e}")
            return audio
    
    def _pitch_shift(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply pitch shift to audio."""
        try:
            n_steps = random.randint(*self.config.augmentation.pitch_shift_range)
            if n_steps != 0:
                audio = torchaudio.functional.pitch_shift(
                    audio.unsqueeze(0), 
                    self.config.audio.sample_rate, 
                    n_steps
                ).squeeze(0)
            return audio
        except Exception as e:
            logger.warning(f"Failed to apply pitch shift: {e}")
            return audio
    
    def _time_mask(self, mel_spec: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """Apply time masking to mel-spectrogram."""
        n_mels, time_steps = mel_spec.shape
        mask_length = int(time_steps * mask_ratio)
        
        if mask_length > 0:
            start = random.randint(0, max(1, time_steps - mask_length))
            mel_spec = mel_spec.clone()
            mel_spec[:, start:start + mask_length] = mel_spec.mean()
        
        return mel_spec
    
    def _freq_mask(self, mel_spec: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """Apply frequency masking to mel-spectrogram."""
        n_mels, time_steps = mel_spec.shape
        mask_length = int(n_mels * mask_ratio)
        
        if mask_length > 0:
            start = random.randint(0, max(1, n_mels - mask_length))
            mel_spec = mel_spec.clone()
            mel_spec[start:start + mask_length, :] = mel_spec.mean()
        
        return mel_spec
    
    def _add_noise_mel(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Add noise by mixing with noise mel-spectrogram."""
        if not self.noise_files:
            return mel_spec
        
        try:
            # Load random noise file
            noise_path = random.choice(self.noise_files)
            noise_mel = self._load_and_process_audio(noise_path)
            
            # Random mixing ratio
            noise_level = random.uniform(*self.config.augmentation.noise_level_range)
            
            # Mix
            mixed_mel = (1 - noise_level) * mel_spec + noise_level * noise_mel
            return mixed_mel
            
        except Exception as e:
            logger.warning(f"Failed to add noise: {e}")
            return mel_spec


def load_dataset_from_directories(config: ExperimentConfig) -> Tuple[AudioDataset, AudioDataset, AudioDataset]:
    """Load dataset from event and noise directories."""
    
    # Load event files
    event_files = []
    if os.path.exists(config.data.event_dir):
        event_files = glob.glob(os.path.join(config.data.event_dir, "*.wav"))
        logger.info(f"Found {len(event_files)} event files")
    else:
        logger.warning(f"Event directory not found: {config.data.event_dir}")
    
    # Load noise files (as negative examples)
    noise_files = []
    if os.path.exists(config.data.noise_dir):
        noise_files = glob.glob(os.path.join(config.data.noise_dir, "*.wav"))
        logger.info(f"Found {len(noise_files)} noise files")
    else:
        logger.warning(f"Noise directory not found: {config.data.noise_dir}")
    
    # Combine files and labels
    all_files = event_files + noise_files
    all_labels = [1] * len(event_files) + [0] * len(noise_files)
    
    # Limit samples per class if specified
    if config.data.max_samples_per_class:
        event_indices = random.sample(range(len(event_files)), 
                                    min(len(event_files), config.data.max_samples_per_class))
        noise_indices = random.sample(range(len(event_files), len(all_files)), 
                                    min(len(noise_files), config.data.max_samples_per_class))
        
        selected_indices = event_indices + noise_indices
        all_files = [all_files[i] for i in selected_indices]
        all_labels = [all_labels[i] for i in selected_indices]
    
    # Shuffle data
    combined = list(zip(all_files, all_labels))
    random.shuffle(combined)
    all_files, all_labels = zip(*combined)
    all_files, all_labels = list(all_files), list(all_labels)
    
    # Split into train/val/test
    n_total = len(all_files)
    n_test = int(n_total * config.data.test_split)
    n_val = int(n_total * config.data.validation_split)
    n_train = n_total - n_test - n_val
    
    train_files = all_files[:n_train]
    train_labels = all_labels[:n_train]
    
    val_files = all_files[n_train:n_train + n_val]
    val_labels = all_labels[n_train:n_train + n_val]
    
    test_files = all_files[n_train + n_val:]
    test_labels = all_labels[n_train + n_val:]
    
    logger.info(f"Dataset split - Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Create datasets
    train_dataset = AudioDataset(train_files, train_labels, config, is_training=True)
    val_dataset = AudioDataset(val_files, val_labels, config, is_training=False)
    test_dataset = AudioDataset(test_files, test_labels, config, is_training=False)
    
    return train_dataset, val_dataset, test_dataset


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> BatchFeature:
    """Custom collate function for batching audio data."""
    
    # Stack input values and labels
    input_values = torch.stack([item['input_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return BatchFeature({
        'input_values': input_values,
        'labels': labels
    })


class BalancedAudioDataset(AudioDataset):
    """Audio dataset with balanced sampling per epoch."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Group indices by class
        self.class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        # Calculate samples per epoch
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.samples_per_epoch = min_class_size * len(self.class_indices)
        
        logger.info(f"Balanced dataset: {self.samples_per_epoch} samples per epoch")
        logger.info(f"Class sizes: {[(label, len(indices)) for label, indices in self.class_indices.items()]}")
    
    def __len__(self) -> int:
        return self.samples_per_epoch
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Use balanced sampling
        epoch_idx = idx % self.samples_per_epoch
        class_id = epoch_idx % len(self.class_indices)
        within_class_idx = epoch_idx // len(self.class_indices)
        
        # Get random sample from the selected class
        class_indices = list(self.class_indices.values())[class_id]
        actual_idx = random.choice(class_indices)
        
        return super().__getitem__(actual_idx)
