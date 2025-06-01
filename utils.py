import os
import logging
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from transformers import EvalPrediction
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    predictions, labels = eval_pred
    
    # Get predicted classes
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # ROC AUC if we have probabilities
    try:
        if eval_pred.predictions.ndim > 1 and eval_pred.predictions.shape[1] == 2:
            probs = torch.softmax(torch.tensor(eval_pred.predictions), dim=1)[:, 1].numpy()
            roc_auc = roc_auc_score(labels, probs)
        else:
            roc_auc = 0.0
    except:
        roc_auc = 0.0
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }
    
    # Add per-class metrics
    class_names = ["no_event", "event"]
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics[f"precision_{class_name}"] = float(precision_per_class[i])
            metrics[f"recall_{class_name}"] = float(recall_per_class[i])
            metrics[f"f1_{class_name}"] = float(f1_per_class[i])
            metrics[f"support_{class_name}"] = int(support[i])
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True
) -> None:
    """Plot confusion matrix."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    val_metrics: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """Plot training history."""
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    if 'accuracy' in val_metrics:
        axes[0, 1].plot(epochs, val_metrics['accuracy'], 'g-', label='Validation Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # F1 Score plot
    if 'f1' in val_metrics:
        axes[1, 0].plot(epochs, val_metrics['f1'], 'purple', label='Validation F1')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # ROC AUC plot
    if 'roc_auc' in val_metrics:
        axes[1, 1].plot(epochs, val_metrics['roc_auc'], 'orange', label='Validation ROC AUC')
        axes[1, 1].set_title('Validation ROC AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('ROC AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to {save_path}")
    
    plt.show()


def evaluate_model_comprehensive(
    model,
    eval_dataloader,
    device: torch.device,
    class_names: List[str] = None,
    output_dir: str = None
) -> Dict[str, float]:
    """Comprehensive model evaluation."""
    
    if class_names is None:
        class_names = ["no_event", "event"]
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move batch to device
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_values=input_values)
            logits = outputs.logits
            
            # Get predictions and probabilities
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Compute metrics
    eval_pred = EvalPrediction(predictions=all_probabilities, label_ids=all_labels)
    metrics = compute_metrics(eval_pred)
    
    # Print detailed results
    logger.info("=== Detailed Evaluation Results ===")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Plot confusion matrix
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            all_labels, all_predictions, class_names, 
            save_path=cm_path, normalize=True
        )
        
        # Save detailed metrics
        metrics_path = os.path.join(output_dir, "detailed_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("=== Detailed Evaluation Results ===\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
        
        logger.info(f"Evaluation results saved to {output_dir}")
    
    return metrics


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "max",
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        self.compare = self._get_compare_fn()
    
    def _get_compare_fn(self):
        if self.mode == "max":
            return lambda current, best: current > best + self.min_delta
        else:
            return lambda current, best: current < best - self.min_delta
    
    def __call__(self, score: float, model=None) -> bool:
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if model is not None and self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best model weights")
        
        return self.early_stop


def log_system_info():
    """Log system and environment information."""
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    logger.info(f"CPU count: {torch.get_num_threads()}")


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from transformers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
