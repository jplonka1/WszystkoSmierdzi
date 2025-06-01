#!/usr/bin/env python3
"""
Audio Event Detection Model Evaluation Script
============================================

This script provides comprehensive evaluation capabilities for trained audio event detection models.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

from config import load_config
from model import create_model
from dataset import AudioDataset, collate_fn
from utils import evaluate_model_comprehensive, setup_logging

logger = logging.getLogger(__name__)


def evaluate_on_directory(
    model_path: str,
    data_dir: str,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 32
) -> Dict:
    """Evaluate model on all audio files in a directory."""
    
    # Load config
    config = load_config(config_path)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_audio_albert_model(config)
    
    # Load model weights
    model_file = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    
    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(Path(data_dir).rglob(ext))
    
    audio_files = [str(f) for f in audio_files]
    logger.info(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        raise ValueError(f"No audio files found in {data_dir}")
    
    # Create dataset (dummy labels since we're just evaluating)
    dataset = AudioDataset(
        audio_paths=audio_files,
        labels=[0] * len(audio_files),  # Dummy labels
        config=config,
        is_training=False
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Run inference
    all_predictions = []
    all_probabilities = []
    all_files = []
    
    logger.info("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(audio_files))
            batch_files = audio_files[start_idx:end_idx]
            
            input_values = batch['input_values'].to(device)
            
            outputs = model(input_values=input_values)
            logits = outputs.logits
            
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_files.extend(batch_files)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'file_path': all_files,
        'prediction': all_predictions,
        'predicted_label': [config.data.id2label[str(pred)] for pred in all_predictions],
        'confidence': [probs.max() for probs in all_probabilities],
        'prob_no_event': [probs[0] for probs in all_probabilities],
        'prob_event': [probs[1] for probs in all_probabilities]
    })
    
    # Add file metadata
    results_df['filename'] = results_df['file_path'].apply(lambda x: os.path.basename(x))
    results_df['directory'] = results_df['file_path'].apply(lambda x: os.path.dirname(x))
    
    # Summary statistics
    summary = {
        'total_files': len(results_df),
        'event_predictions': (results_df['prediction'] == 1).sum(),
        'no_event_predictions': (results_df['prediction'] == 0).sum(),
        'mean_confidence': results_df['confidence'].mean(),
        'low_confidence_files': (results_df['confidence'] < 0.7).sum()
    }
    
    logger.info(f"Evaluation Summary:")
    logger.info(f"  Total files: {summary['total_files']}")
    logger.info(f"  Event predictions: {summary['event_predictions']}")
    logger.info(f"  No-event predictions: {summary['no_event_predictions']}")
    logger.info(f"  Mean confidence: {summary['mean_confidence']:.3f}")
    logger.info(f"  Low confidence files: {summary['low_confidence_files']}")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_path = os.path.join(output_dir, "evaluation_results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Save summary
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Evaluation Summary\n")
            f.write("================\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        # Create visualizations
        create_evaluation_plots(results_df, output_dir)
    
    return {
        'results_df': results_df,
        'summary': summary
    }


def create_evaluation_plots(results_df: pd.DataFrame, output_dir: str):
    """Create evaluation visualization plots."""
    
    # Set style safely
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            pass  # Use default style
    
    # 1. Prediction distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prediction counts
    pred_counts = results_df['predicted_label'].value_counts()
    axes[0, 0].bar(pred_counts.index, pred_counts.values)
    axes[0, 0].set_title('Prediction Distribution')
    axes[0, 0].set_ylabel('Count')
    
    # Confidence distribution
    axes[0, 1].hist(results_df['confidence'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Confidence Distribution')
    axes[0, 1].set_xlabel('Confidence')
    axes[0, 1].set_ylabel('Count')
    
    # Confidence by prediction
    for label in results_df['predicted_label'].unique():
        subset = results_df[results_df['predicted_label'] == label]
        axes[1, 0].hist(subset['confidence'], alpha=0.6, label=label, bins=20)
    axes[1, 0].set_title('Confidence by Prediction')
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # Probability scatter
    axes[1, 1].scatter(results_df['prob_no_event'], results_df['prob_event'], 
                      c=results_df['prediction'], alpha=0.6, cmap='viridis')
    axes[1, 1].set_xlabel('P(No Event)')
    axes[1, 1].set_ylabel('P(Event)')
    axes[1, 1].set_title('Probability Space')
    axes[1, 1].plot([0, 1], [1, 0], 'r--', alpha=0.5, label='Decision Boundary')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Low confidence analysis
    low_conf_threshold = 0.7
    low_conf_files = results_df[results_df['confidence'] < low_conf_threshold]
    
    if len(low_conf_files) > 0:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        low_conf_files['predicted_label'].value_counts().plot(kind='bar')
        plt.title(f'Low Confidence Predictions (< {low_conf_threshold})')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.hist(low_conf_files['confidence'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Low Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'low_confidence_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def compare_models(
    model_paths: List[str],
    data_dir: str,
    model_names: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """Compare multiple models on the same dataset."""
    
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(model_paths))]
    
    all_results = []
    
    for model_path, model_name in zip(model_paths, model_names):
        logger.info(f"Evaluating {model_name}...")
        
        try:
            result = evaluate_on_directory(
                model_path=model_path,
                data_dir=data_dir,
                config_path=config_path,
                output_dir=None,  # Don't save individual results
                batch_size=32
            )
            
            summary = result['summary']
            summary['model_name'] = model_name
            summary['model_path'] = model_path
            all_results.append(summary)
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            continue
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(all_results)
    
    if output_dir and len(comparison_df) > 0:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison results
        comparison_path = os.path.join(output_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Event detection rate
        axes[0, 0].bar(comparison_df['model_name'], 
                      comparison_df['event_predictions'] / comparison_df['total_files'])
        axes[0, 0].set_title('Event Detection Rate')
        axes[0, 0].set_ylabel('Proportion')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Mean confidence
        axes[0, 1].bar(comparison_df['model_name'], comparison_df['mean_confidence'])
        axes[0, 1].set_title('Mean Confidence')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Low confidence rate
        axes[1, 0].bar(comparison_df['model_name'], 
                      comparison_df['low_confidence_files'] / comparison_df['total_files'])
        axes[1, 0].set_title('Low Confidence Rate')
        axes[1, 0].set_ylabel('Proportion')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Summary table (text)
        axes[1, 1].axis('off')
        table_data = comparison_df[['model_name', 'event_predictions', 'mean_confidence']].round(3)
        table = axes[1, 1].table(cellText=table_data.values, 
                                colLabels=table_data.columns,
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        axes[1, 1].set_title('Summary Table')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison results saved to {output_dir}")
    
    return comparison_df


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Audio Event Detection Model Evaluation")
    parser.add_argument("--model-path", type=str, required=True, 
                       help="Path to trained model directory")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing audio files to evaluate")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--compare-models", nargs="+", help="Multiple model paths for comparison")
    parser.add_argument("--model-names", nargs="+", help="Names for comparison models")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO")
    
    if args.compare_models:
        # Compare multiple models
        logger.info("Comparing multiple models...")
        comparison_df = compare_models(
            model_paths=args.compare_models,
            data_dir=args.data_dir,
            model_names=args.model_names,
            config_path=args.config,
            output_dir=args.output_dir
        )
        
        print("\nModel Comparison Results:")
        print(comparison_df.to_string(index=False))
        
    else:
        # Evaluate single model
        logger.info("Evaluating single model...")
        result = evaluate_on_directory(
            model_path=args.model_path,
            data_dir=args.data_dir,
            config_path=args.config,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
        
        print("\nEvaluation Results:")
        print(f"Total files: {result['summary']['total_files']}")
        print(f"Event predictions: {result['summary']['event_predictions']}")
        print(f"No-event predictions: {result['summary']['no_event_predictions']}")
        print(f"Mean confidence: {result['summary']['mean_confidence']:.3f}")
        print(f"Low confidence files: {result['summary']['low_confidence_files']}")


if __name__ == "__main__":
    main()
