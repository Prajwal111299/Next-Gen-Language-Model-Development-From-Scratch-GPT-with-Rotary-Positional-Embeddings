"""
Originally forked from Andrej Karpathy's minGPT,
Modified based on Stanford CS224N 2023-24: Homework 4

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
Soumya Chatterjee <soumyac@stanford.edu>
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def precompute_rotary_emb(dim, max_positions):
    """
    Precompute rotary embedding cache.
    Returns a tensor of shape (max_positions, dim/2, 2) where the last dimension
    contains the cosine and sine values for each position and dimension.
    """
    # Compute the inverse frequency for each dimension (for indices 0 to dim/2 - 1)
    inv_freq = 1.0 / (10000 ** ((2 * torch.arange(0, dim // 2).float()) / dim))
    # Create a positions tensor (max_positions, 1)
    positions = torch.arange(max_positions, dtype=torch.float).unsqueeze(1)
    # Compute angles (max_positions, dim/2)
    angles = positions * inv_freq.unsqueeze(0)
    # Stack cos and sin to get shape (max_positions, dim/2, 2)
    rope_cache = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    return rope_cache


def apply_rotary_emb(x, rope_cache):
    """Apply the RoPE to the input tensor x.
       x shape: (B, n_head, T, head_dim) where head_dim is even.
       Returns the rotated tensor of the same shape.
    """
    B, n_head, T, head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim must be even for RoPE."
    # Reshape x to separate the dimensions for applying RoPE.
    x = x.view(B, n_head, T, head_dim // 2, 2)
    # Truncate the precomputed cache if T is less than max_positions.
    rope = rope_cache[:T, :, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, T, head_dim//2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    cos = rope[..., 0]
    sin = rope[..., 1]
    # Apply the rotation: for each pair, compute [x1*cos - x2*sin, x1*sin + x2*cos]
    x_rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    # Reshape back to original dimensions: (B, n_head, T, head_dim)
    rotated_x = x_rotated.view(B, n_head, T, head_dim)
    return rotated_x

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.rope = getattr(config, 'rope', False)  # Default to False if not specified
        if self.rope:
            # Ensure that the head dimension is even.
            assert (config.n_embd // config.n_head) % 2 == 0
            # Precompute RoPE cache for each head (using head dimension) and for the max sequence length.
            rope_cache = precompute_rotary_emb(config.n_embd // config.n_head, config.block_size)
            self.register_buffer("rope_cache", rope_cache)

        # Regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # Create the causal mask to ensure each token only attends to its left context (including itself).
        # The mask is a lower-triangular matrix of shape (block_size, block_size).
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        if self.rope:
            # Apply RoPE to query and key.
            q = apply_rotary_emb(q, self.rope_cache)
            k = apply_rotary_emb(k, self.rope_cache)

        # Compute attention scores
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply the causal mask to the attention scores
        # We need to make sure the mask is on the same device as the attention tensor
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        att = F.softmax(att, dim=-1)
        
        # Apply dropout to the attention weights
        att = self.attn_drop(att)
        
        # Apply attention weights to the values
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = att @ v
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply output projection and residual dropout
        y = self.resid_drop(self.proj(y))
        
        return y