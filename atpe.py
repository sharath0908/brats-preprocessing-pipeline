"""
Adaptive Tumor Patch Embedding (ATPE) Module
Multi-scale patch extraction and embedding for medical image analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
import math


class PatchEmbedding2D(nn.Module):
    """
    2D Convolutional patch embedding layer
    Extracts local features and generates tokens
    
    Architecture:
        Input: [B, C, H, W]
        Conv -> BN -> ReLU -> Flatten -> Linear projection
        Output: [B, num_patches, embed_dim]
    """
    
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int, stride: int = None):
        """
        Args:
            in_channels (int): Input channels
            patch_size (int): Size of patches (patch_size x patch_size)
            embed_dim (int): Embedding dimension
            stride (int): Stride for patch extraction (default: patch_size)
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        stride = stride or patch_size
        
        # Convolutional patch extraction
        self.conv = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=stride, 
            padding=0
        )
        self.bn = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract patches and generate embeddings
        
        Args:
            x (torch.Tensor): Input features [B, C, H, W]
            
        Returns:
            tuple: (tokens, spatial_dims)
                - tokens: [B, num_patches, embed_dim]
                - spatial_dims: (num_patches_h, num_patches_w)
        """
        B, C, H, W = x.shape
        
        # Extract patches using convolution
        # Output: [B, embed_dim, num_patches_h, num_patches_w]
        patches = self.conv(x)
        patches = self.bn(patches)
        patches = self.relu(patches)
        
        _, _, Ph, Pw = patches.shape
        
        # Flatten spatial dimensions
        # [B, embed_dim, Ph, Pw] -> [B, embed_dim, Ph*Pw] -> [B, Ph*Pw, embed_dim]
        tokens = patches.flatten(2).transpose(1, 2)
        
        return tokens, (Ph, Pw)


class PositionalEncoding(nn.Module):
    """
    2D Positional encoding for patch tokens
    Uses learnable positional embeddings
    
    Args:
        embed_dim (int): Embedding dimension
        max_h (int): Maximum height dimension
        max_w (int): Maximum width dimension
    """
    
    def __init__(self, embed_dim: int, max_h: int = 32, max_w: int = 32):
        super().__init__()
        
        # Create 2D positional encoding
        self.pos_h = nn.Embedding(max_h, embed_dim // 2)
        self.pos_w = nn.Embedding(max_w, embed_dim // 2)
        
        self.embed_dim = embed_dim
        self.max_h = max_h
        self.max_w = max_w
    
    def forward(self, tokens: torch.Tensor, spatial_dims: Tuple[int, int]) -> torch.Tensor:
        """
        Add positional encoding to tokens
        
        Args:
            tokens (torch.Tensor): Patch tokens [B, num_patches, embed_dim]
            spatial_dims (tuple): (num_patches_h, num_patches_w)
            
        Returns:
            torch.Tensor: Tokens with positional encoding [B, num_patches, embed_dim]
        """
        B, N, D = tokens.shape
        Ph, Pw = spatial_dims
        
        # Create 2D position indices
        h_indices = torch.arange(Ph, device=tokens.device)
        w_indices = torch.arange(Pw, device=tokens.device)
        
        # Get positional embeddings
        pos_h = self.pos_h(h_indices)  # [Ph, embed_dim//2]
        pos_w = self.pos_w(w_indices)  # [Pw, embed_dim//2]
        
        # Create grid
        h_pos = pos_h.unsqueeze(1).expand(-1, Pw, -1)  # [Ph, Pw, embed_dim//2]
        w_pos = pos_w.unsqueeze(0).expand(Ph, -1, -1)  # [Ph, Pw, embed_dim//2]
        
        # Concatenate and reshape
        pos_encoding = torch.cat([h_pos, w_pos], dim=-1)  # [Ph, Pw, embed_dim]
        pos_encoding = pos_encoding.reshape(N, D).unsqueeze(0)  # [1, num_patches, embed_dim]
        
        return tokens + pos_encoding


class AdaptiveTumorPatchEmbedding(nn.Module):
    """
    Adaptive Tumor Patch Embedding (ATPE) Module
    
    Extracts multi-scale patches from medical images for tumor analysis:
    - Fine patches (8×8): Capture tumor microstructure details
    - Medium patches (16×16): Capture tumor morphology
    - Coarse patches (32×32): Capture tumor context
    
    Architecture Diagram:
    ┌─────────────────────────────────────────────────┐
    │           Input Feature Map                      │
    │           [B, C, H, W]                           │
    └────────┬──────────┬──────────┬──────────────────┘
             │          │          │
        ┌────▼───┐  ┌────▼───┐  ┌─▼────────┐
        │ Fine   │  │ Medium │  │  Coarse  │
        │ Patch  │  │ Patch  │  │  Patch   │
        │ 8×8    │  │ 16×16  │  │  32×32   │
        └────┬───┘  └────┬───┘  └─┬────────┘
             │           │        │
        ┌────▼───┐   ┌────▼───┐  ┌─▼────────┐
        │ Token  │   │ Token  │  │  Token   │
        │ Gen    │   │ Gen    │  │  Gen     │
        └────┬───┘   └────┬───┘  └─┬────────┘
             │           │        │
        ┌────▼───────────▼────────▼────────┐
        │   Positional Encoding            │
        │   (2D learnable embeddings)      │
        └────┬───────────┬────────┬────────┘
             │           │        │
    ┌─Fine──▼─┐ Medium──▼┐ Coarse▼─┐
    │ Tokens  │ Tokens   │ Tokens  │
    │ [B,N₁,D]│ [B,N₂,D] │ [B,N₃,D]│
    └─────────┴──────────┴─────────┘
    
    Args:
        in_channels (int): Input feature channels (default: 4 for 4 MRI modalities)
        embed_dim (int): Embedding dimension (default: 256)
        fine_patch_size (int): Size of fine patches (default: 8)
        medium_patch_size (int): Size of medium patches (default: 16)
        coarse_patch_size (int): Size of coarse patches (default: 32)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 256,
        fine_patch_size: int = 8,
        medium_patch_size: int = 16,
        coarse_patch_size: int = 32,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.fine_patch_size = fine_patch_size
        self.medium_patch_size = medium_patch_size
        self.coarse_patch_size = coarse_patch_size
        
        # Patch extractors for each scale
        self.fine_embedder = PatchEmbedding2D(
            in_channels=in_channels,
            patch_size=fine_patch_size,
            embed_dim=embed_dim,
        )
        
        self.medium_embedder = PatchEmbedding2D(
            in_channels=in_channels,
            patch_size=medium_patch_size,
            embed_dim=embed_dim,
        )
        
        self.coarse_embedder = PatchEmbedding2D(
            in_channels=in_channels,
            patch_size=coarse_patch_size,
            embed_dim=embed_dim,
        )
        
        # Positional encoders
        self.fine_pos_encoder = PositionalEncoding(embed_dim, max_h=32, max_w=32)
        self.medium_pos_encoder = PositionalEncoding(embed_dim, max_h=16, max_w=16)
        self.coarse_pos_encoder = PositionalEncoding(embed_dim, max_h=8, max_w=8)
        
        # Token projection layers (optional, for additional feature transformation)
        self.fine_proj = nn.Linear(embed_dim, embed_dim)
        self.medium_proj = nn.Linear(embed_dim, embed_dim)
        self.coarse_proj = nn.Linear(embed_dim, embed_dim)
        
        # Activation
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale patch tokens
        
        Args:
            x (torch.Tensor): Input feature map [B, C, H, W]
                - B: Batch size
                - C: Number of channels (typically 4 for 4 MRI modalities)
                - H, W: Spatial dimensions (typically 224×224)
        
        Returns:
            dict: Dictionary containing tokens at each scale:
                {
                    'fine_tokens': [B, num_fine_patches, embed_dim],
                    'medium_tokens': [B, num_medium_patches, embed_dim],
                    'coarse_tokens': [B, num_coarse_patches, embed_dim],
                    'fine_spatial': (num_fine_patches_h, num_fine_patches_w),
                    'medium_spatial': (num_medium_patches_h, num_medium_patches_w),
                    'coarse_spatial': (num_coarse_patches_h, num_coarse_patches_w),
                }
        
        Tensor Dimension Explanation:
        ─────────────────────────────────────────────────────────────
        Input: [B, 4, 224, 224]
        ├─ B: Batch size (typically 8-32)
        ├─ 4: MRI modalities (T1, T1ce, T2, FLAIR)
        └─ 224×224: Spatial dimensions
        
        Fine Stream (8×8 patches):
        ├─ Extracted patches: [B, 256, 28, 28]
        │  (224/8 = 28 patches per dimension)
        ├─ Flattened tokens: [B, 784, 256]
        │  (28×28 = 784 tokens, embed_dim = 256)
        └─ With positional encoding: [B, 784, 256]
        
        Medium Stream (16×16 patches):
        ├─ Extracted patches: [B, 256, 14, 14]
        │  (224/16 = 14 patches per dimension)
        ├─ Flattened tokens: [B, 196, 256]
        │  (14×14 = 196 tokens)
        └─ With positional encoding: [B, 196, 256]
        
        Coarse Stream (32×32 patches):
        ├─ Extracted patches: [B, 256, 7, 7]
        │  (224/32 = 7 patches per dimension)
        ├─ Flattened tokens: [B, 49, 256]
        │  (7×7 = 49 tokens)
        └─ With positional encoding: [B, 49, 256]
        
        Total tokens: 784 + 196 + 49 = 1029 multi-scale tokens
        ─────────────────────────────────────────────────────────────
        """
        
        B, C, H, W = x.shape
        
        # Extract fine patches
        fine_tokens, fine_spatial = self.fine_embedder(x)
        fine_tokens = self.fine_pos_encoder(fine_tokens, fine_spatial)
        fine_tokens = self.gelu(self.fine_proj(fine_tokens))
        
        # Extract medium patches
        medium_tokens, medium_spatial = self.medium_embedder(x)
        medium_tokens = self.medium_pos_encoder(medium_tokens, medium_spatial)
        medium_tokens = self.gelu(self.medium_proj(medium_tokens))
        
        # Extract coarse patches
        coarse_tokens, coarse_spatial = self.coarse_embedder(x)
        coarse_tokens = self.coarse_pos_encoder(coarse_tokens, coarse_spatial)
        coarse_tokens = self.gelu(self.coarse_proj(coarse_tokens))
        
        return {
            'fine_tokens': fine_tokens,
            'medium_tokens': medium_tokens,
            'coarse_tokens': coarse_tokens,
            'fine_spatial': fine_spatial,
            'medium_spatial': medium_spatial,
            'coarse_spatial': coarse_spatial,
        }


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    """
    Test and demonstrate ATPE module
    """
    
    print("=" * 80)
    print("Adaptive Tumor Patch Embedding (ATPE) - Testing")
    print("=" * 80)
    
    # Create dummy input
    # Simulates BraTS MRI data: 4 modalities (T1, T1ce, T2, FLAIR)
    batch_size = 2
    num_channels = 4
    height, width = 224, 224
    
    x = torch.randn(batch_size, num_channels, height, width)
    print(f"\nInput shape: {x.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Channels (modalities): {num_channels}")
    print(f"  Spatial dimensions: {height}×{width}")
    
    # Initialize ATPE module
    atpe = AdaptiveTumorPatchEmbedding(
        in_channels=num_channels,
        embed_dim=256,
        fine_patch_size=8,
        medium_patch_size=16,
        coarse_patch_size=32,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in atpe.parameters())
    trainable_params = sum(p.numel() for p in atpe.parameters() if p.requires_grad)
    print(f"\nATPA Module:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Forward pass
    print(f"\nForward pass...")
    output = atpe(x)
    
    # Display output
    print(f"\nOutput tokens:")
    print(f"  Fine tokens shape: {output['fine_tokens'].shape}")
    print(f"    - Batch size: {output['fine_tokens'].shape[0]}")
    print(f"    - Number of patches: {output['fine_tokens'].shape[1]} (28×28)")
    print(f"    - Token dimension: {output['fine_tokens'].shape[2]}")
    
    print(f"\n  Medium tokens shape: {output['medium_tokens'].shape}")
    print(f"    - Batch size: {output['medium_tokens'].shape[0]}")
    print(f"    - Number of patches: {output['medium_tokens'].shape[1]} (14×14)")
    print(f"    - Token dimension: {output['medium_tokens'].shape[2]}")
    
    print(f"\n  Coarse tokens shape: {output['coarse_tokens'].shape}")
    print(f"    - Batch size: {output['coarse_tokens'].shape[0]}")
    print(f"    - Number of patches: {output['coarse_tokens'].shape[1]} (7×7)")
    print(f"    - Token dimension: {output['coarse_tokens'].shape[2]}")
    
    # Calculate total tokens
    total_tokens = (output['fine_tokens'].shape[1] + 
                   output['medium_tokens'].shape[1] + 
                   output['coarse_tokens'].shape[1])
    print(f"\n  Total multi-scale tokens: {total_tokens} (784 + 196 + 49)")
    
    # Verify memory efficiency
    input_size = batch_size * num_channels * height * width
    fine_tokens_size = output['fine_tokens'].numel()
    medium_tokens_size = output['medium_tokens'].numel()
    coarse_tokens_size = output['coarse_tokens'].numel()
    total_tokens_size = fine_tokens_size + medium_tokens_size + coarse_tokens_size
    
    print(f"\nMemory analysis:")
    print(f"  Input elements: {input_size:,}")
    print(f"  Output tokens elements: {total_tokens_size:,}")
    print(f"  Compression ratio: {input_size / total_tokens_size:.2f}×")
    
    print("\n" + "=" * 80)
    print("✓ ATPE module test successful!")
    print("=" * 80)
