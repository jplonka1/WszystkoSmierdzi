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
import random

from config import ExperimentConfig

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """Custom PyTorch dataset for audio classification."""
    
    def __init__(
        self,
        audio_paths: List[str],
        config: ExperimentConfig,
        is_training: bool = True,#so far were not really using this flag and have to think if theres going to be difference between f.e. inference/evaluation and training here on the side of this class, ignore for now I guess
        shared_noise_files: List[str] = None,
        shared_background_noise = None,
    ):
        print("soefneojnfwoenf")
        self.audio_paths = audio_paths
        self.config = config
        self.is_training = is_training

    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path = self.audio_paths[idx]#TODO: investigate if idx should be randomized here or why its actually giving idx at all        
        # TODO: completely rebuild this funciton, it now is setup to not throw errors
        print(f"something asked for {idx} item from audio dataset")
        audio_tensor = self._load_and_process_audio(audio_path)
        # TODO: augmentations
        print(f"Loaded audio tensor of shape {audio_tensor.shape}")
        label = 1 if "drone" in audio_path else 0  # placeholder
        return {
            'input_values': audio_tensor,
            'labels': torch.tensor(label, dtype=torch.long)#TODO: ensure this is correct notation/syntax to prep this flag (drone or no, label 1 or 0) for model
        }
    
    def _load_and_process_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and return raw audio tensor."""
        try:
            # Load audio
            audio, sr = torchaudio.load(audio_path)#TODO: Load the file as .wav and apply seom augmentations that cannot be done on tensors (not supported by tensor audio), do them then convert to tensor and apply the rest of augmentations on GPU. TODO2: Ensure there is no context switching after final tensor conversion
            
            # Convert to mono if necessary
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.config.audio.sample_rate:
                print("SHOULD NEVER HAPPEN DEBUG 4")
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
        
        if current_length > target_length:#TODO: Id like different logic of, not symmetric (if i understand the code correctly this is symmetric)
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

def load_dataset_from_directories(config: ExperimentConfig) -> Tuple[AudioDataset, AudioDataset, AudioDataset]:
    """Load dataset from event and noise directories."""
    
    # Load event files
    event_files = []
    if os.path.exists(config.data.event_dir):
        event_files = glob.glob(os.path.join(config.data.event_dir, "*.wav"))#chatgpt se globa (cntrl+I gupaku jesli dalej nie pamietasz, najlepiej przed cntr+I zaznaczyc kontekst myszka)
        logger.info(f"Found {len(event_files)} event files")
    else:
        logger.warning(f"Event directory not found: {config.data.event_dir}")

    
    random.shuffle(event_files)
    n_total = len(event_files)
    n_test = int(n_total * config.data.test_split)
    n_val = int(n_total * config.data.validation_split)
    n_train = n_total - n_test - n_val
    
    train_files = event_files[:n_train]
    val_files = event_files[n_train:n_train + n_val]
    test_files = event_files[n_train + n_val:]
    
    logger.info(f"Dataset split - Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Load noise files once and share them across all datasets to save memory
    #logger.info("Loading shared noise files...")
    shared_noise_files = []
    if os.path.exists(config.data.noise_dir):
        shared_noise_files = glob.glob(os.path.join(config.data.noise_dir, "*.wav"))
        logger.info(f"Found {len(shared_noise_files)} shared noise files")
    
    shared_background_noise = []
    if os.path.exists(config.data.background_dir):
        shared_background_noise = glob.glob(os.path.join(config.data.background_dir, "*.wav"))
        logger.info(f"Found {len(shared_background_noise)} shared background noise files")

            # Load .wav files from shared_background_noise list of strings
        # Load .wav files from shared_background_noise list of strings into memory
    background_files = []
    for f in shared_background_noise:
        if f.lower().endswith('.wav'):
            try:
                audio, sr = torchaudio.load(f)
                background_files.append((audio, sr))
            except Exception as e:
                logger.warning(f"Failed to load background noise file {f}: {e}")
        
    noise_files = []
    for f in shared_noise_files:
        if f.lower().endswith('.wav'):
            try:
                audio, sr = torchaudio.load(f)
                noise_files.append((audio, sr))
            except Exception as e:
                logger.warning(f"Failed to load noise file {f}: {e}")
    
    # Create datasets with shared noise files
    train_dataset = AudioDataset(
        train_files, config, is_training=True,
        shared_noise_files=noise_files,
        shared_background_noise=background_files
    )
    val_dataset = AudioDataset(
        val_files, config, is_training=False,
        shared_noise_files=noise_files,
        shared_background_noise=background_files
    )
    test_dataset = AudioDataset(
        test_files, config, is_training=False,
        shared_noise_files=noise_files,
        shared_background_noise=background_files
    )
    
    return train_dataset, val_dataset, test_dataset


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> BatchFeature:#read more about it
    """Custom collate function for batching audio data."""
    # Input: List of individual items from dataset.__getitem__()
    # Each item: {'input_values': tensor, 'labels': tensor}
    
    # Stack into batches
    input_values = torch.stack([item['input_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # Return Hugging Face compatible BatchFeature
    return BatchFeature({
        'input_values': input_values,  # Shape: [batch_size, audio_length]
        'labels': labels               # Shape: [batch_size]
    })
#ABOUT collate_fn output from agent Claude SOnnet 4:
# Reasons why it's essential:

# Tensor Stacking: Individual audio tensors need to be stacked into batch dimensions for GPU processing

# Input: [tensor1, tensor2, ...] → Output: tensor[batch_size, audio_length]
# Hugging Face Compatibility: Returns BatchFeature object expected by ALBERT model

# Standard PyTorch default_collate would return regular dict, not BatchFeature
# Model expects specific key names (input_values, labels)
# Memory Layout: Ensures proper tensor contiguity for efficient GPU operations

# Type Safety: Maintains consistent tensor dtypes across batch
# Hugging Face Design Pattern: The AlbertForSequenceClassification expects labels during training for automatic loss computation

# When labels=None: Returns only logits (inference mode)
# When labels provided: Computes cross-entropy loss automatically and returns both loss and logits