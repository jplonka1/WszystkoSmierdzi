#!/usr/bin/env python3
"""
Detailed ALBERT Parameter Analysis
=================================

ALBERT uses parameter sharing which makes layer count impact different from other transformers.
This script provides a detailed breakdown of how each parameter affects model size.
"""

""" Key Insights:
1. hidden_size has QUADRATIC impact (H² in attention)
2. num_hidden_layers has ZERO impact (parameter sharing)
3. num_attention_heads has ZERO impact on param count
4. intermediate_size has LINEAR impact (H × I)
"""

import torch
import torch.nn as nn
from transformers import AlbertConfig, AlbertModel, AlbertForSequenceClassification


def detailed_albert_analysis():
    """Detailed analysis of ALBERT parameter structure."""
    
    print("🔬 ALBERT Parameter Structure Analysis")
    print("=" * 60)
    
    # Your configuration
    config = AlbertConfig(
        hidden_size=768,
        num_attention_heads=16,
        num_hidden_layers=4,
        intermediate_size=3072,
        max_position_embeddings=512,
        num_labels=2
    )
    
    model = AlbertForSequenceClassification(config)
    
    print(f"\n📊 Your Configuration:")
    print(f"hidden_size: {config.hidden_size}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"intermediate_size: {config.intermediate_size}")
    
    print(f"\n🧱 ALBERT Layer Structure Breakdown:")
    
    total_params = 0
    
    # Analyze each component
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"{name:40s}: {params:8,} params")
                total_params += params
    
    print(f"{'':40s}  {'-'*10}")
    print(f"{'Total':40s}: {total_params:8,} params")
    
    return total_params


def parameter_formula_analysis():
    """Mathematical breakdown of ALBERT parameter count."""
    
    print(f"\n🧮 ALBERT Parameter Count Formula Analysis")
    print("=" * 60)
    
    # Configuration
    H = 768    # hidden_size
    A = 16     # num_attention_heads
    L = 4      # num_hidden_layers (NOTE: shared in ALBERT!)
    I = 3072   # intermediate_size
    V = 30000  # vocab_size (approximately)
    P = 512    # max_position_embeddings
    
    print(f"Configuration:")
    print(f"H (hidden_size): {H}")
    print(f"A (num_attention_heads): {A}")
    print(f"L (num_hidden_layers): {L} (SHARED in ALBERT!)")
    print(f"I (intermediate_size): {I}")
    print(f"V (vocab_size): ~{V}")
    print(f"P (max_position_embeddings): {P}")
    
    print(f"\n📏 Parameter Breakdown:")
    
    # Embedding layers
    word_embeddings = V * H
    position_embeddings = P * H
    token_type_embeddings = 2 * H  # Usually 2 token types
    embedding_layernorm = 2 * H    # weight + bias
    
    print(f"Word Embeddings (V × H): {word_embeddings:,}")
    print(f"Position Embeddings (P × H): {position_embeddings:,}")
    print(f"Token Type Embeddings: {token_type_embeddings:,}")
    print(f"Embedding LayerNorm: {embedding_layernorm:,}")
    
    embeddings_total = word_embeddings + position_embeddings + token_type_embeddings + embedding_layernorm
    print(f"Total Embeddings: {embeddings_total:,}")
    
    # ALBERT Transformer Layers (SHARED!)
    print(f"\n🔄 ALBERT Transformer (Parameter Sharing):")
    
    # Self-attention
    attention_qkv = 3 * H * H  # Q, K, V projections
    attention_out = H * H      # Output projection
    attention_layernorm = 2 * H
    
    # Feed-forward
    ff_intermediate = H * I    # Hidden to intermediate
    ff_output = I * H          # Intermediate to hidden
    ff_layernorm = 2 * H
    
    # CRITICAL: In ALBERT, these weights are SHARED across all layers!
    single_layer_params = attention_qkv + attention_out + attention_layernorm + ff_intermediate + ff_output + ff_layernorm
    
    print(f"Self-Attention (Q,K,V): {attention_qkv:,}")
    print(f"Attention Output: {attention_out:,}")
    print(f"Attention LayerNorm: {attention_layernorm:,}")
    print(f"Feed-Forward In: {ff_intermediate:,}")
    print(f"Feed-Forward Out: {ff_output:,}")
    print(f"FF LayerNorm: {ff_layernorm:,}")
    print(f"Single Layer Total: {single_layer_params:,}")
    print(f"⚠️  SHARED across {L} layers (not multiplied!)")
    
    # Classification head
    classifier_params = H * 2  # num_labels = 2
    
    print(f"\n🎯 Classification Head: {classifier_params:,}")
    
    # Total calculation
    total_albert = embeddings_total + single_layer_params + classifier_params
    
    print(f"\n📊 Total Parameter Count:")
    print(f"Embeddings: {embeddings_total:,}")
    print(f"Transformer (shared): {single_layer_params:,}")
    print(f"Classifier: {classifier_params:,}")
    print(f"Total: {total_albert:,}")
    
    return {
        'embeddings': embeddings_total,
        'transformer': single_layer_params,
        'classifier': classifier_params,
        'total': total_albert
    }


def impact_analysis():
    """Analyze impact of changing each parameter."""
    
    print(f"\n🎯 Parameter Impact Analysis")
    print("=" * 60)
    
    base_config = {
        'H': 768,
        'A': 16,
        'L': 4,
        'I': 3072
    }
    
    def calculate_params(H, A, L, I):
        V, P = 30000, 512
        
        # Embeddings (scale with H)
        embeddings = V * H + P * H + 2 * H + 2 * H
        
        # Transformer (scale with H² and I, NOT with L due to sharing)
        transformer = 3 * H * H + H * H + 2 * H + H * I + I * H + 2 * H
        
        # Classifier
        classifier = H * 2
        
        return embeddings + transformer + classifier
    
    base_params = calculate_params(**base_config)
    
    print(f"Base Configuration: {base_params:,} parameters")
    
    print(f"\n🔍 Impact of hidden_size (H):")
    print("Size | Parameters | Change    | Change %")
    print("-" * 45)
    
    for H in [256, 384, 512, 768, 1024]:
        config = base_config.copy()
        config['H'] = H
        config['I'] = H * 4  # Maintain 4:1 ratio
        
        params = calculate_params(**config)
        change = params - base_params
        change_pct = (change / base_params) * 100
        
        marker = " (base)" if H == 768 else ""
        print(f"{H:4d} | {params:9,} | {change:+8,} | {change_pct:+6.1f}%{marker}")
    
    print(f"\n🔍 Impact of num_hidden_layers (L):")
    print("⚠️  MINIMAL IMPACT due to parameter sharing!")
    print("Layers | Parameters | Change")
    print("-" * 30)
    
    for L in [1, 4, 8, 12]:
        config = base_config.copy()
        config['L'] = L
        
        params = calculate_params(**config)
        change = params - base_params
        
        marker = " (base)" if L == 4 else ""
        print(f"  {L:2d}   | {params:9,} | {change:+6,}{marker}")
    
    print(f"\n🔍 Impact of num_attention_heads (A):")
    print("⚠️  NO IMPACT on parameter count!")
    print("Heads | Parameters | Note")
    print("-" * 35)
    
    for A in [4, 8, 12, 16, 24]:
        if 768 % A != 0:
            note = "(invalid: H not divisible)"
        else:
            note = "(same params)" if A != 16 else "(base)"
        
        print(f"  {A:2d}  | {base_params:9,} | {note}")
    
    print(f"\n💡 Key Insights:")
    print(f"1. hidden_size has QUADRATIC impact (H² in attention)")
    print(f"2. num_hidden_layers has ZERO impact (parameter sharing)")
    print(f"3. num_attention_heads has ZERO impact on param count")
    print(f"4. intermediate_size has LINEAR impact (H × I)")


def audio_projection_analysis():
    """Analyze the audio-specific components."""
    
    print(f"\n🎵 Audio-Specific Components Analysis")
    print("=" * 60)
    
    n_mels = 128
    hidden_size = 768
    
    # Feature projection layer
    projection_params = n_mels * hidden_size + hidden_size  # weight + bias
    
    # Dropout (no parameters)
    dropout_params = 0
    
    # Temporal pooling (no parameters in AdaptiveAvgPool1d)
    pooling_params = 0
    
    print(f"Feature Projection ({n_mels} → {hidden_size}): {projection_params:,} params")
    print(f"Dropout: {dropout_params:,} params")
    print(f"Temporal Pooling: {pooling_params:,} params")
    print(f"Total Audio Components: {projection_params:,} params")
    
    print(f"\n🔍 Impact of n_mels on projection layer:")
    print("n_mels | Projection Params | Total Added")
    print("-" * 40)
    
    for mels in [64, 80, 128, 256]:
        proj_params = mels * hidden_size + hidden_size
        marker = " (current)" if mels == 128 else ""
        
        print(f"  {mels:3d}  | {proj_params:12,} | {proj_params:10,}{marker}")


if __name__ == "__main__":
    detailed_albert_analysis()
    parameter_formula_analysis()
    impact_analysis()
    audio_projection_analysis()
