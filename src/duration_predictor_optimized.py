# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Optimized Duration Predictor for TTNN with on-device convolutions.

The duration predictor predicts phoneme durations from text encoder outputs:
1. 2-layer Conv1d stack (kernel_size 3)
2. Layer normalization
3. ReLU activation
4. Linear projection to duration logits

Key optimizations:
- All Conv1d layers on-device with DRAM for large inputs
- LayerNorm implemented with TTNN operations
- No CPU fallbacks
"""

from typing import Optional, Any, List

import torch

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

from conv1d_optimized import ttnn_conv1d_optimized as ttnn_conv1d


def ttnn_layernorm(
    x: Any,
    weight: Any,
    bias: Any,
    eps: float = 1e-5,
    device: Optional[Any] = None,
) -> Any:
    """
    TTNN implementation of LayerNorm.

    LayerNorm: normalized = (x - mean) / sqrt(var + eps)
                output = gamma * normalized + beta

    Args:
        x: Input tensor [B, C, T]
        weight: Gamma (scale) [C]
        bias: Beta (shift) [C]
        eps: Epsilon for numerical stability
        device: TTNN device

    Returns:
        Normalized tensor [B, C, T]
    """
    # Check if input is PyTorch tensor
    is_torch = isinstance(x, torch.Tensor)

    if not TTNN_AVAILABLE or is_torch:
        # PyTorch fallback
        import torch.nn.functional as F

        def to_torch(t):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t
            if TTNN_AVAILABLE:
                return ttnn.to_torch(t)
            return t

        x_torch = to_torch(x)
        w_torch = to_torch(weight)
        b_torch = to_torch(bias)

        # Reshape for LayerNorm: [B, C, T] -> [B, T, C]
        x_reshaped = x_torch.permute(0, 2, 1)

        # Apply LayerNorm
        normalized = F.layer_norm(
            x_reshaped,
            normalized_shape=w_torch.shape,
            weight=w_torch,
            bias=b_torch,
            eps=eps,
        )

        # Reshape back: [B, T, C] -> [B, C, T]
        out = normalized.permute(0, 2, 1)
        return out

    # TTNN implementation
    batch_size, channels, length = x.shape

    # Compute mean along channel dimension
    x_sum = ttnn.sum(x, dim=1, keepdim=True)  # [B, 1, T]
    mean = ttnn.multiply(x_sum, 1.0 / channels)  # [B, 1, T]

    # Compute variance
    x_centered = ttnn.subtract(x, mean)  # [B, C, T]
    x_centered_sq = ttnn.multiply(x_centered, x_centered)  # [B, C, T]
    var = ttnn.sum(x_centered_sq, dim=1, keepdim=True)  # [B, 1, T]
    var = ttnn.multiply(var, 1.0 / channels)  # [B, 1, T]

    # Normalize
    var_plus_eps = ttnn.add(var, eps)
    std = ttnn.sqrt(var_plus_eps)  # [B, 1, T]
    normalized = ttnn.divide(x_centered, std)  # [B, C, T]

    # Scale and shift
    # weight [C] needs to be broadcasted to [B, C, T]
    weight_expanded = ttnn.unsqueeze(weight, 0)  # [1, C]
    weight_expanded = ttnn.unsqueeze(weight_expanded, 2)  # [1, C, 1]
    scaled = ttnn.multiply(normalized, weight_expanded)

    # bias [C] needs to be broadcasted to [B, C, T]
    bias_expanded = ttnn.unsqueeze(bias, 0)  # [1, C]
    bias_expanded = ttnn.unsqueeze(bias_expanded, 2)  # [1, C, 1]
    out = ttnn.add(scaled, bias_expanded)

    return out


class DurationPredictorOptimized:
    """
    Optimized Duration Predictor for phoneme duration prediction.

    Architecture:
        1. 2x Conv1d layers with kernel_size 3, padding 1
        2. LayerNorm after each conv
        3. ReLU activation after each conv
        4. Dropout (training only)
        5. Linear projection to duration logits

    Args:
        conv1_weight: First Conv1d weight [hidden, hidden, 1, K]
        conv1_bias: First Conv1d bias [hidden]
        ln1_weight: First LayerNorm gamma [hidden]
        ln1_bias: First LayerNorm beta [hidden]
        conv2_weight: Second Conv1d weight [hidden, hidden, 1, K]
        conv2_bias: Second Conv1d bias [hidden]
        ln2_weight: Second LayerNorm gamma [hidden]
        ln2_bias: Second LayerNorm beta [hidden]
        proj_weight: Projection weight [1, hidden, 1, K]
        proj_bias: Projection bias [1]
        dropout: Dropout rate (0 for inference)
        device: TTNN device
    """

    def __init__(
        self,
        conv1_weight: Any,
        conv1_bias: Any,
        ln1_weight: Any,
        ln1_bias: Any,
        conv2_weight: Any,
        conv2_bias: Any,
        ln2_weight: Any,
        ln2_bias: Any,
        proj_weight: Any,
        proj_bias: Any,
        dropout: float = 0.1,
        device: Optional[Any] = None,
    ):
        self.conv1_weight = conv1_weight
        self.conv1_bias = conv1_bias
        self.ln1_weight = ln1_weight
        self.ln1_bias = ln1_bias

        self.conv2_weight = conv2_weight
        self.conv2_bias = conv2_bias
        self.ln2_weight = ln2_weight
        self.ln2_bias = ln2_bias

        self.proj_weight = proj_weight
        self.proj_bias = proj_bias

        self.dropout = dropout
        self.device = device

        # Infer hidden size
        if hasattr(conv1_weight.shape, 'rank'):
            self.hidden_size = conv1_weight.shape[0]
        else:
            self.hidden_size = conv1_weight.shape[0]

    def __call__(self, x: Any, x_mask: Optional[Any] = None) -> Any:
        """
        Predict durations from text encoder outputs.

        Args:
            x: Text encoder outputs [B, hidden, T]
            x_mask: Optional mask [B, 1, T]

        Returns:
            Duration logits [B, 1, T]
        """
        # Check if input is PyTorch tensor
        is_torch = isinstance(x, torch.Tensor)

        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask)
        return self._forward_ttnn(x, x_mask)

    def _forward_ttnn(self, x, x_mask):
        """Forward pass with optimized TTNN operations."""

        # Apply mask if provided
        if x_mask is not None:
            x = ttnn.multiply(x, x_mask)

        # First Conv1d block
        x = ttnn_conv1d(
            x,
            weight=self.conv1_weight,
            bias=self.conv1_bias,
            stride=1,
            padding=1,
            device=self.device,
        )
        x = ttnn_layernorm(x, self.ln1_weight, self.ln1_bias, device=self.device)
        x = ttnn.relu(x)

        # Apply mask after first block
        if x_mask is not None:
            x = ttnn.multiply(x, x_mask)

        # Second Conv1d block
        x = ttnn_conv1d(
            x,
            weight=self.conv2_weight,
            bias=self.conv2_bias,
            stride=1,
            padding=1,
            device=self.device,
        )
        x = ttnn_layernorm(x, self.ln2_weight, self.ln2_bias, device=self.device)
        x = ttnn.relu(x)

        # Apply mask after second block
        if x_mask is not None:
            x = ttnn.multiply(x, x_mask)

        # Project to duration logits
        x = ttnn_conv1d(
            x,
            weight=self.proj_weight,
            bias=self.proj_bias,
            stride=1,
            padding=0,
            device=self.device,
        )

        # Apply final mask
        if x_mask is not None:
            x = ttnn.multiply(x, x_mask)

        return x

    def _forward_pytorch(self, x, x_mask):
        """PyTorch fallback for development."""
        import torch.nn.functional as F

        def to_torch(t):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t
            if TTNN_AVAILABLE:
                return ttnn.to_torch(t)
            return t

        # Apply mask if provided
        if x_mask is not None:
            x_mask_torch = to_torch(x_mask)
            x = x * x_mask_torch

        # First Conv1d block
        w1 = to_torch(self.conv1_weight)
        if w1.dim() == 4:
            w1 = w1.squeeze(2)
        b1 = to_torch(self.conv1_bias)
        x = F.conv1d(x, w1, b1, padding=1)

        # LayerNorm
        x_reshaped = x.permute(0, 2, 1)
        x = F.layer_norm(
            x_reshaped,
            normalized_shape=to_torch(self.ln1_weight).shape,
            weight=to_torch(self.ln1_weight),
            bias=to_torch(self.ln1_bias),
        ).permute(0, 2, 1)

        x = F.relu(x)

        if x_mask is not None:
            x = x * x_mask_torch

        # Second Conv1d block
        w2 = to_torch(self.conv2_weight)
        if w2.dim() == 4:
            w2 = w2.squeeze(2)
        b2 = to_torch(self.conv2_bias)
        x = F.conv1d(x, w2, b2, padding=1)

        # LayerNorm
        x_reshaped = x.permute(0, 2, 1)
        x = F.layer_norm(
            x_reshaped,
            normalized_shape=to_torch(self.ln2_weight).shape,
            weight=to_torch(self.ln2_weight),
            bias=to_torch(self.ln2_bias),
        ).permute(0, 2, 1)

        x = F.relu(x)

        if x_mask is not None:
            x = x * x_mask_torch

        # Project to duration logits
        w_proj = to_torch(self.proj_weight)
        if w_proj.dim() == 4:
            w_proj = w_proj.squeeze(2)
        b_proj = to_torch(self.proj_bias)
        x = F.conv1d(x, w_proj, b_proj)

        if x_mask is not None:
            x = x * x_mask_torch

        return x


def get_duration_predictor_weights(state_dict: dict, prefix: str = "dur_pred") -> dict:
    """
    Extract duration predictor weights from state dict.

    Args:
        state_dict: Model state dict
        prefix: Prefix for duration predictor keys

    Returns:
        Dictionary with extracted weights
    """
    weights = {}

    # First Conv1d block
    weights["conv1_weight"] = state_dict.get(f"{prefix}.convs.0.weight")
    weights["conv1_bias"] = state_dict.get(f"{prefix}.convs.0.bias")
    weights["ln1_weight"] = state_dict.get(f"{prefix}.convs.1.weight")
    weights["ln1_bias"] = state_dict.get(f"{prefix}.convs.1.bias")

    # Second Conv1d block
    weights["conv2_weight"] = state_dict.get(f"{prefix}.convs.3.weight")
    weights["conv2_bias"] = state_dict.get(f"{prefix}.convs.3.bias")
    weights["ln2_weight"] = state_dict.get(f"{prefix}.convs.4.weight")
    weights["ln2_bias"] = state_dict.get(f"{prefix}.convs.4.bias")

    # Projection
    weights["proj_weight"] = state_dict.get(f"{prefix}.proj.weight")
    weights["proj_bias"] = state_dict.get(f"{prefix}.proj.bias")

    return weights
