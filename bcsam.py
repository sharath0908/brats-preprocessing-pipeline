"""
Bidirectional Cross-Scale Attention Module (BCSAM)
Multi-directional attention fusion across fine, medium, and coarse patch tokens
Optimized for medical image segmentation with tumor boundary awareness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention
    Standard transformer attention mechanism
    
    Args:
        embed_dim (int): Total embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        bias (bool): Whether to use bias in projections
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query (torch.Tensor): Query tokens [B, Nq, D]
            key (torch.Tensor): Key tokens [B, Nk, D]
            value (torch.Tensor): Value tokens [B, Nk, D]
            attn_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            tuple: (output, attention_weights)
                - output: [B, Nq, D]
                - attention_weights: [B, num_heads, Nq, Nk]
        """
        B, Nq, D = query.shape
        _, Nk, _ = key.shape
        
        # Project to (B, N, num_heads, head_dim)
        Q = self.q_proj(query).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Nq, D)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class CrossScaleAttentionBlock(nn.Module):
    """
    Single cross-scale attention block
    Allows one scale to attend to another scale
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        ffn_dim (int): Hidden dimension of feedforward network
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): Query tokens [B, Nq, D]
            key (torch.Tensor): Key tokens [B, Nk, D]
            value (torch.Tensor): Value tokens [B, Nk, D]
            
        Returns:
            torch.Tensor: Output tokens [B, Nq, D]
        """
        # Cross-attention with residual
        attn_out, _ = self.attention(query, key, value)
        query = query + attn_out
        query = self.norm1(query)
        
        # FFN with residual
        ffn_out = self.ffn(query)
        query = query + ffn_out
        query = self.norm2(query)
        
        return query


class BidirectionalCrossScaleAttention(nn.Module):
    """
    Bidirectional Cross-Scale Attention Module (BCSAM)
    
    Enables bidirectional information flow across multiple scales:
    - Fine ↔ Medium (bidirectional)
    - Medium ↔ Coarse (bidirectional)
    - Fine ↔ Coarse (bidirectional)
    
    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │         Input: Fine, Medium, Coarse Tokens              │
    └────────┬────────────────┬────────────────┬──────────────┘
             │                │                │
    ┌────────▼────┐   ┌───────▼────┐   ┌──────▼──────┐
    │   Fine      │   │   Medium   │   │   Coarse    │
    │ [B,784,D]   │   │ [B,196,D]  │   │  [B,49,D]   │
    └────────┬────┘   └───────┬────┘   └──────┬──────┘
             │                │                │
    ┌────────┼────────────────┼────────────────┼────────────┐
    │        │ Bidirectional  │ Bidirectional │            │
    │        │ Cross-Scale    │ Cross-Scale   │            │
    │        │ Attention      │ Attention     │            │
    └────────┼────────────────┼────────────────┼────────────┘
             │                │                │
    ┌────────▼────┐   ┌───────▼────┐   ┌──────▼──────┐
    │   Fine      │   │   Medium   │   │   Coarse    │
    │ Enhanced    │   │ Enhanced   │   │  Enhanced   │
    │ [B,784,D]   │   │ [B,196,D]  │   │  [B,49,D]   │
    └─────────────┘   └────────────┘   └─────────────┘
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        ffn_dim (int): Hidden dimension of FFN
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Fine ↔ Medium attention blocks
        self.fine_to_medium = CrossScaleAttentionBlock(embed_dim, num_heads, ffn_dim, dropout)
        self.medium_to_fine = CrossScaleAttentionBlock(embed_dim, num_heads, ffn_dim, dropout)
        
        # Medium ↔ Coarse attention blocks
        self.medium_to_coarse = CrossScaleAttentionBlock(embed_dim, num_heads, ffn_dim, dropout)
        self.coarse_to_medium = CrossScaleAttentionBlock(embed_dim, num_heads, ffn_dim, dropout)
        
        # Fine ↔ Coarse attention blocks (long-range)
        self.fine_to_coarse = CrossScaleAttentionBlock(embed_dim, num_heads, ffn_dim, dropout)
        self.coarse_to_fine = CrossScaleAttentionBlock(embed_dim, num_heads, ffn_dim, dropout)
        
        # Fusion layer for multi-scale fusion
        self.fusion_weight_fine = nn.Parameter(torch.ones(3) / 3)  # Fuse info from 3 sources
        self.fusion_weight_medium = nn.Parameter(torch.ones(3) / 3)
        self.fusion_weight_coarse = nn.Parameter(torch.ones(3) / 3)
        
        # Learnable temperature for fusion
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self,
        fine_tokens: torch.Tensor,
        medium_tokens: torch.Tensor,
        coarse_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply bidirectional cross-scale attention
        
        Args:
            fine_tokens (torch.Tensor): Fine-scale tokens [B, 784, embed_dim]
            medium_tokens (torch.Tensor): Medium-scale tokens [B, 196, embed_dim]
            coarse_tokens (torch.Tensor): Coarse-scale tokens [B, 49, embed_dim]
        
        Returns:
            dict: Enhanced tokens at each scale
                {
                    'fine_tokens': [B, 784, D],
                    'medium_tokens': [B, 196, D],
                    'coarse_tokens': [B, 49, D],
                }
        
        Attention Flow:
        ───────────────────────────────────────────────────
        1. Fine ↔ Medium Bidirectional Attention
           - Fine attends to Medium (captures coarse context)
           - Medium attends to Fine (refines coarse features)
        
        2. Medium ↔ Coarse Bidirectional Attention
           - Medium attends to Coarse (captures global context)
           - Coarse attends to Medium (guides boundary refinement)
        
        3. Fine ↔ Coarse Long-range Attention
           - Fine attends to Coarse (global shape awareness)
           - Coarse attends to Fine (fine-grained detail awareness)
        
        4. Feature Fusion
           - Each scale fuses information from all three scales
           - Weighted fusion with learnable weights
        ───────────────────────────────────────────────────
        """
        
        # ===== Step 1: Fine ↔ Medium Bidirectional Attention =====
        fine_from_medium = self.fine_to_medium(fine_tokens, medium_tokens, medium_tokens)
        medium_from_fine = self.medium_to_fine(medium_tokens, fine_tokens, fine_tokens)
        
        # ===== Step 2: Medium ↔ Coarse Bidirectional Attention =====
        medium_from_coarse = self.medium_to_coarse(medium_tokens, coarse_tokens, coarse_tokens)
        coarse_from_medium = self.coarse_to_medium(coarse_tokens, medium_tokens, medium_tokens)
        
        # ===== Step 3: Fine ↔ Coarse Long-range Attention =====
        # Upscale coarse tokens to match fine dimension for direct attention
        fine_from_coarse = self.fine_to_coarse(fine_tokens, coarse_tokens, coarse_tokens)
        coarse_from_fine = self.coarse_to_fine(coarse_tokens, fine_tokens, fine_tokens)
        
        # ===== Step 4: Multi-scale Fusion =====
        # Fuse information from all three pathways for each scale
        
        # Fine tokens: fuse [original_fine, fine_from_medium, fine_from_coarse]
        fine_fusion_weights = self.softmax(self.fusion_weight_fine)
        fine_enhanced = (
            fine_fusion_weights[0] * fine_tokens +
            fine_fusion_weights[1] * fine_from_medium +
            fine_fusion_weights[2] * fine_from_coarse
        )
        
        # Medium tokens: fuse [original_medium, medium_from_fine, medium_from_coarse]
        medium_fusion_weights = self.softmax(self.fusion_weight_medium)
        medium_enhanced = (
            medium_fusion_weights[0] * medium_tokens +
            medium_fusion_weights[1] * medium_from_fine +
            medium_fusion_weights[2] * medium_from_coarse
        )
        
        # Coarse tokens: fuse [original_coarse, coarse_from_medium, coarse_from_fine]
        coarse_fusion_weights = self.softmax(self.fusion_weight_coarse)
        coarse_enhanced = (
            coarse_fusion_weights[0] * coarse_tokens +
            coarse_fusion_weights[1] * coarse_from_medium +
            coarse_fusion_weights[2] * coarse_from_fine
        )
        
        return {
            'fine_tokens': fine_enhanced,
            'medium_tokens': medium_enhanced,
            'coarse_tokens': coarse_enhanced,
        }


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    """
    Test and demonstrate BCSAM module
    """
    
    print("=" * 100)
    print("Bidirectional Cross-Scale Attention Module (BCSAM) - Testing")
    print("=" * 100)
    
    # Initialize module
    bcsam = BidirectionalCrossScaleAttention(
        embed_dim=256,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.1,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in bcsam.parameters())
    print(f"\nBCSAM Module Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Embedding dimension: 256")
    print(f"  Number of heads: 8")
    print(f"  Head dimension: 32")
    
    # Create dummy input tokens from different scales
    batch_size = 2
    embed_dim = 256
    
    # From ATPE module
    fine_tokens = torch.randn(batch_size, 784, embed_dim)      # 28×28 patches
    medium_tokens = torch.randn(batch_size, 196, embed_dim)    # 14×14 patches
    coarse_tokens = torch.randn(batch_size, 49, embed_dim)     # 7×7 patches
    
    print(f"\nInput Token Shapes:")
    print(f"  Fine tokens:   {fine_tokens.shape}   (28×28 = 784 patches)")
    print(f"  Medium tokens: {medium_tokens.shape}   (14×14 = 196 patches)")
    print(f"  Coarse tokens: {coarse_tokens.shape}    (7×7 = 49 patches)")
    
    # Forward pass
    print(f"\nForward pass...")
    output = bcsam(fine_tokens, medium_tokens, coarse_tokens)
    
    # Display results
    print(f"\nOutput Token Shapes (after cross-scale attention):")
    print(f"  Fine tokens:   {output['fine_tokens'].shape}   ✓")
    print(f"  Medium tokens: {output['medium_tokens'].shape}   ✓")
    print(f"  Coarse tokens: {output['coarse_tokens'].shape}    ✓")
    
    # Verify no dimension change
    assert output['fine_tokens'].shape == fine_tokens.shape, "Fine tokens dimension mismatch"
    assert output['medium_tokens'].shape == medium_tokens.shape, "Medium tokens dimension mismatch"
    assert output['coarse_tokens'].shape == coarse_tokens.shape, "Coarse tokens dimension mismatch"
    
    # Attention flow visualization
    print(f"\nAttention Flow Summary:")
    print(f"""
    Bidirectional paths:
    ├─ Fine ↔ Medium
    │  ├─ Fine attends to Medium (coarse context)
    │  └─ Medium attends to Fine (fine detail refinement)
    │
    ├─ Medium ↔ Coarse
    │  ├─ Medium attends to Coarse (global context)
    │  └─ Coarse attends to Medium (boundary guidance)
    │
    └─ Fine ↔ Coarse (long-range)
       ├─ Fine attends to Coarse (global shape)
       └─ Coarse attends to Fine (detail awareness)
    
    Multi-scale Fusion:
    ├─ Fine:   fuses [original, from_medium, from_coarse]
    ├─ Medium: fuses [original, from_fine, from_coarse]
    └─ Coarse: fuses [original, from_medium, from_fine]
    """)
    
    # Compute information flow metrics
    print(f"\nInformation Flow Analysis:")
    
    # Measure token similarity before and after
    fine_similarity_before = F.cosine_similarity(fine_tokens[0, :2], fine_tokens[0, 2:4]).mean().item()
    fine_similarity_after = F.cosine_similarity(output['fine_tokens'][0, :2], 
                                               output['fine_tokens'][0, 2:4]).mean().item()
    
    medium_similarity_before = F.cosine_similarity(medium_tokens[0, :2], 
                                                   medium_tokens[0, 2:4]).mean().item()
    medium_similarity_after = F.cosine_similarity(output['medium_tokens'][0, :2], 
                                                 output['medium_tokens'][0, 2:4]).mean().item()
    
    print(f"  Fine tokens - similarity change: {fine_similarity_before:.4f} → {fine_similarity_after:.4f}")
    print(f"  Medium tokens - similarity change: {medium_similarity_before:.4f} → {medium_similarity_after:.4f}")
    
    # Cross-scale information transfer
    print(f"\nCross-scale Information Transfer:")
    fine_change = (output['fine_tokens'] - fine_tokens).norm().item()
    medium_change = (output['medium_tokens'] - medium_tokens).norm().item()
    coarse_change = (output['coarse_tokens'] - coarse_tokens).norm().item()
    
    print(f"  Fine tokens changed by: {fine_change:.4f}")
    print(f"  Medium tokens changed by: {medium_change:.4f}")
    print(f"  Coarse tokens changed by: {coarse_change:.4f}")
    
    print("\n" + "=" * 100)
    print("✓ BCSAM module test successful!")
    print("=" * 100)
