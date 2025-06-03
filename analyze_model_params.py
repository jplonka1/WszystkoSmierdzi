#!/usr/bin/env python3
"""
Model Parameter Analysis Script
==============================

Analyzes how hidden_size, num_attention_heads, and num_hidden_layers
affect the total parameter count in the AudioALBERTClassifier model.
"""

import torch
from transformers import AlbertConfig, AlbertForSequenceClassification
from model import AudioALBERTClassifier, create_model
from config import load_config


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def analyze_albert_params(hidden_size, num_attention_heads, num_hidden_layers, num_labels=2):
    """Analyze ALBERT parameter count with different configurations."""
    
    # Create ALBERT config
    config = AlbertConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=hidden_size * 4,  # Standard transformer ratio
        num_labels=num_labels,
        max_position_embeddings=512
    )
    
    # Create ALBERT model
    albert = AlbertForSequenceClassification(config)
    
    total, trainable = count_parameters(albert)
    
    return {
        'total_params': total,
        'trainable_params': trainable,
        'config': {
            'hidden_size': hidden_size,
            'num_attention_heads': num_attention_heads,
            'num_hidden_layers': num_hidden_layers,
            'intermediate_size': config.intermediate_size
        }
    }


def analyze_audio_albert_params(hidden_size, num_attention_heads, num_hidden_layers, n_mels=128):
    """Analyze AudioALBERTClassifier parameter count."""
    
    # Create AudioALBERTClassifier
    model = AudioALBERTClassifier(
        model_name="albert-base-v2",  # This will be overridden by custom config
        num_labels=2,
        n_mels=n_mels,
        max_length=512,
        freeze_albert=False
    )
    
    # Manually override the ALBERT config for analysis
    albert_config = AlbertConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=hidden_size * 4,
        num_labels=2,
        max_position_embeddings=512
    )
    
    # Create new ALBERT with custom config
    model.albert = AlbertForSequenceClassification(albert_config)
    
    # Recreate feature projection to match hidden_size
    model.feature_projection = torch.nn.Linear(n_mels, hidden_size)
    
    total, trainable = count_parameters(model)
    
    # Analyze component breakdown
    albert_total, albert_trainable = count_parameters(model.albert)
    projection_params = sum(p.numel() for p in model.feature_projection.parameters())
    
    return {
        'total_params': total,
        'trainable_params': trainable,
        'albert_params': albert_total,
        'projection_params': projection_params,
        'other_params': total - albert_total - projection_params,
        'config': {
            'hidden_size': hidden_size,
            'num_attention_heads': num_attention_heads,
            'num_hidden_layers': num_hidden_layers,
            'intermediate_size': hidden_size * 4,
            'n_mels': n_mels
        }
    }


def parameter_impact_analysis():
    """Comprehensive analysis of parameter impacts."""
    
    print("🧠 ALBERT Model Parameter Impact Analysis")
    print("=" * 60)
    
    # Base configuration from config_default.json
    base_config = {
        'hidden_size': 768,
        'num_attention_heads': 16,
        'num_hidden_layers': 4,
        'n_mels': 128
    }
    
    print(f"\n📊 Base Configuration Analysis:")
    print(f"hidden_size: {base_config['hidden_size']}")
    print(f"num_attention_heads: {base_config['num_attention_heads']}")
    print(f"num_hidden_layers: {base_config['num_hidden_layers']}")
    print(f"n_mels: {base_config['n_mels']}")
    
    base_result = analyze_audio_albert_params(**base_config)
    print(f"\n🎯 Base Model Parameters:")
    print(f"Total Parameters: {base_result['total_params']:,}")
    print(f"Trainable Parameters: {base_result['trainable_params']:,}")
    print(f"ALBERT Component: {base_result['albert_params']:,}")
    print(f"Feature Projection: {base_result['projection_params']:,}")
    print(f"Other Components: {base_result['other_params']:,}")
    
    # Impact of num_hidden_layers
    print(f"\n🔍 Impact of num_hidden_layers:")
    print("Layers | Total Params | ALBERT Params | Difference")
    print("-" * 50)
    
    base_layers = base_config['num_hidden_layers']
    base_albert_params = base_result['albert_params']
    
    for layers in [1, 2, 4, 6, 8, 12]:
        config = base_config.copy()
        config['num_hidden_layers'] = layers
        result = analyze_audio_albert_params(**config)
        
        diff = result['albert_params'] - base_albert_params
        diff_str = f"{diff:+,}" if layers != base_layers else "baseline"
        
        print(f"  {layers:2d}   | {result['total_params']:10,} | {result['albert_params']:11,} | {diff_str}")
    
    # Impact of hidden_size
    print(f"\n🔍 Impact of hidden_size:")
    print("Hidden | Total Params | ALBERT Params | Projection | Difference")
    print("-" * 65)
    
    base_hidden = base_config['hidden_size']
    
    for hidden_size in [256, 384, 512, 768, 1024]:
        config = base_config.copy()
        config['hidden_size'] = hidden_size
        config['num_attention_heads'] = min(16, hidden_size // 48)  # Adjust heads for valid division
        
        result = analyze_audio_albert_params(**config)
        
        diff = result['total_params'] - base_result['total_params']
        diff_str = f"{diff:+,}" if hidden_size != base_hidden else "baseline"
        
        print(f"  {hidden_size:3d}  | {result['total_params']:10,} | {result['albert_params']:11,} | {result['projection_params']:8,} | {diff_str}")
    
    # Impact of num_attention_heads
    print(f"\n🔍 Impact of num_attention_heads:")
    print("Heads | Total Params | ALBERT Params | Difference")
    print("-" * 45)
    
    base_heads = base_config['num_attention_heads']
    
    for heads in [4, 8, 12, 16, 24]:
        if base_config['hidden_size'] % heads != 0:
            continue  # Skip invalid configurations
            
        config = base_config.copy()
        config['num_attention_heads'] = heads
        result = analyze_audio_albert_params(**config)
        
        diff = result['albert_params'] - base_albert_params
        diff_str = f"{diff:+,}" if heads != base_heads else "baseline"
        
        print(f"  {heads:2d}  | {result['total_params']:10,} | {result['albert_params']:11,} | {diff_str}")
    
    # Memory impact analysis
    print(f"\n💾 Memory Impact Analysis:")
    print(f"Model Size (FP32): ~{base_result['total_params'] * 4 / 1024 / 1024:.1f} MB")
    print(f"Model Size (FP16): ~{base_result['total_params'] * 2 / 1024 / 1024:.1f} MB")
    print(f"Gradient Memory (FP32): ~{base_result['trainable_params'] * 4 / 1024 / 1024:.1f} MB")
    
    # Comparison with standard ALBERT-base-v2
    print(f"\n🔬 Comparison with Standard ALBERT-base-v2:")
    standard_albert = analyze_albert_params(768, 16, 12, 2)
    print(f"Standard ALBERT-base-v2: {standard_albert['total_params']:,} parameters")
    print(f"Your Configuration: {base_result['albert_params']:,} parameters")
    print(f"Reduction: {standard_albert['total_params'] - base_result['albert_params']:,} parameters")
    print(f"Ratio: {base_result['albert_params'] / standard_albert['total_params']:.2%} of standard size")


if __name__ == "__main__":
    parameter_impact_analysis()
