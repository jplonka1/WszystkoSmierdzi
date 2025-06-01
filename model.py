import torch
import torch.nn as nn
import torchaudio.transforms as T
from transformers import AlbertForSequenceClassification, AlbertTokenizer, AlbertConfig
from typing import Optional, Dict, Any
import librosa
import numpy as np


class AudioFeatureExtractor(nn.Module):
    """Extract mel-spectrogram features from audio for ALBERT processing."""
    
    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        
        # Convert to dB scale
        self.amplitude_to_db = T.AmplitudeToDB()
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio waveform to mel-spectrogram features.
        
        Args:
            audio: Raw audio tensor (batch_size, audio_length)
            
        Returns:
            mel_spec: Mel-spectrogram (batch_size, n_mels, time_frames)
        """
        # Ensure audio is 2D (batch_size, audio_length)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3:
            audio = audio.squeeze(1)  # Remove channel dimension if present
            
        # Extract mel-spectrogram
        mel_spec = self.mel_transform(audio)
        mel_spec = self.amplitude_to_db(mel_spec)
        
        return mel_spec


class AudioALBERTClassifier(nn.Module):
    """
    ALBERT-based audio event detection model.
    Uses mel-spectrogram features as input to ALBERT for sequence classification.
    """
    
    def __init__(
        self, 
        model_name: str = "albert-base-v2",
        num_labels: int = 2,
        sample_rate: int = 16000,
        n_mels: int = 128,
        max_length: int = 512,
        freeze_albert: bool = False
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.max_length = max_length
        self.n_mels = n_mels
        
        # Audio feature extraction
        self.audio_feature_extractor = AudioFeatureExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels
        )
        
        # ALBERT configuration and model
        self.albert_config = AlbertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        self.albert = AlbertForSequenceClassification.from_pretrained(
            model_name,
            config=self.albert_config
        )
        
        # Freeze ALBERT if specified
        if freeze_albert:
            for param in self.albert.albert.parameters():
                param.requires_grad = False
        
        # Projection layer to map mel features to ALBERT input dimension
        self.feature_projection = nn.Linear(n_mels, self.albert_config.embedding_size)
        self.dropout = nn.Dropout(0.1)
        
        # Temporal pooling for variable-length sequences
        self.temporal_pooling = nn.AdaptiveAvgPool1d(max_length)
        
    def forward(
        self, 
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_values: Raw audio tensor (batch_size, audio_length)
            attention_mask: Attention mask (batch_size, sequence_length)
            labels: Target labels (batch_size,)
            
        Returns:
            Dictionary containing loss, logits, and other outputs
        """
        batch_size = input_values.size(0)
        device = input_values.device
        
        # Extract mel-spectrogram features
        mel_features = self.audio_feature_extractor(input_values)  # (batch, n_mels, time)
        
        # Transpose to (batch, time, n_mels) for sequence processing
        mel_features = mel_features.transpose(1, 2)  # (batch, time, n_mels)
        
        # Apply temporal pooling to ensure consistent sequence length
        if mel_features.size(1) > self.max_length:
            # Pool to max_length if sequence is too long
            mel_features = mel_features.transpose(1, 2)  # (batch, n_mels, time)
            mel_features = self.temporal_pooling(mel_features)  # (batch, n_mels, max_length)
            mel_features = mel_features.transpose(1, 2)  # (batch, max_length, n_mels)
        elif mel_features.size(1) < self.max_length:
            # Pad if sequence is too short
            pad_length = self.max_length - mel_features.size(1)
            padding = torch.zeros(batch_size, pad_length, self.n_mels, device=device)
            mel_features = torch.cat([mel_features, padding], dim=1)
        
        # Project mel features to ALBERT input dimension
        inputs_embeds = self.feature_projection(mel_features)  # (batch, max_length, hidden_size)
        inputs_embeds = self.dropout(inputs_embeds)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, self.max_length, device=device)
        
        # Pass through ALBERT
        outputs = self.albert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }
    
    def predict(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Make predictions on audio input.
        
        Args:
            audio: Raw audio tensor
            
        Returns:
            Dictionary with predictions and probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(audio)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
        return {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'logits': logits.cpu().numpy()
        }


def create_model(config) -> AudioALBERTClassifier:
    """
    Create AudioALBERTClassifier model with configuration.
    
    Args:
        config: Configuration object with model parameters
        
    Returns:
        AudioALBERTClassifier model instance
    """
    return AudioALBERTClassifier(
        model_name=getattr(config.model, 'model_name', 'albert-base-v2'),
        num_labels=getattr(config.model, 'num_labels', 2),
        sample_rate=getattr(config.audio, 'sample_rate', 16000),
        n_mels=getattr(config.audio, 'n_mels', 128),
        max_length=getattr(config.model, 'max_position_embeddings', 512),
        freeze_albert=getattr(config.model, 'freeze_base_model', False)
    )