import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    max_duration: float = 10.0  # seconds
    min_duration: float = 0.5   # seconds
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    normalize: bool = True


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    enable: bool = True
    noise_prob: float = 0.5
    noise_level_range: tuple = (0.1, 0.5)
    gain_range: tuple = (0.5, 1.5)
    pitch_shift_prob: float = 0.2
    pitch_shift_range: tuple = (-2, 2)
    time_mask_prob: float = 0.3
    time_mask_ratio: float = 0.1
    freq_mask_prob: float = 0.3
    freq_mask_ratio: float = 0.1


@dataclass
class ModelConfig:
    """Simplified ALBERT model configuration."""
    model_name: str = "albert-base-v2"
    num_labels: int = 2
    max_position_embeddings: int = 512
    freeze_base_model: bool = False
    classifier_dropout_prob: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./models/audio_albert_classifier"
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # Evaluation and saving
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    
    # Logging
    logging_steps: int = 50
    logging_strategy: str = "steps"
    report_to: List[str] = None
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # Mixed precision
    fp16: bool = True
    dataloader_num_workers: int = 4
    seed: int = 42


@dataclass
class DataConfig:
    """Dataset configuration."""
    event_dir: str = "dataset_current/training"
    noise_dir: str = "Background"
    validation_split: float = 0.2
    test_split: float = 0.1
    cache_dir: str = "./cache"
    max_samples_per_class: Optional[int] = None
    
    # Class labels
    label2id: Dict[str, int] = None
    id2label: Dict[int, str] = None
    
    def __post_init__(self):
        if self.label2id is None:
            self.label2id = {"no_event": 0, "event": 1}
        if self.id2label is None:
            self.id2label = {0: "no_event", 1: "event"}


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    audio: AudioConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    
    def __init__(self, **kwargs):
        self.audio = AudioConfig(**kwargs.get('audio', {}))
        self.augmentation = AugmentationConfig(**kwargs.get('augmentation', {}))
        self.model = ModelConfig(**kwargs.get('model', {}))
        self.training = TrainingConfig(**kwargs.get('training', {}))
        self.data = DataConfig(**kwargs.get('data', {}))
    
    @classmethod
    def from_json(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def update(self, **kwargs) -> 'ExperimentConfig':
        """Update configuration with new values."""
        config_dict = asdict(self)
        
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        deep_update(config_dict, kwargs)
        return ExperimentConfig(**config_dict)


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


def load_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """Load configuration from file or return default."""
    if config_path and os.path.exists(config_path):
        return ExperimentConfig.from_json(config_path)
    return get_default_config()
