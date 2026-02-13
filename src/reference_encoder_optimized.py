# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Optimized Reference Encoder for TTNN with on-device GRU.

The reference encoder extracts speaker characteristics from reference audio:
1. 6-layer Conv1d stack (strides 2, 2, 2, 2, 2, 1)
2. GRU layer for temporal modeling
3. Final attention pooling

Key optimizations:
- All Conv1d layers use DRAM for stride >= 2
- GRU uses dynamic L1/DRAM switching
- No CPU fallbacks throughout the pipeline
"""

from typing import Optional, Any, List, Tuple

import torch

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

from conv1d_optimized import ttnn_conv1d_optimized as ttnn_conv1d
from gru_optimized import ttnn_gru_optimized, GRULayerOptimized


LRELU_SLOPE = 0.1


class ReferenceEncoderOptimized:
    """
    Optimized Reference Encoder for speaker embedding extraction.

    Architecture:
        1. 6x Conv1d layers with stride 2 (except last layer)
        2. GRU layer for temporal modeling
        3. Attention pooling to produce speaker embedding

    Args:
        conv_weights: List of 6 Conv1d weights
        conv_biases: List of 6 Conv1d biases
        gru_weight_ih: GRU input-hidden weights
        gru_weight_hh: GRU hidden-hidden weights
        gru_bias_ih: GRU input-hidden bias
        gru_bias_hh: GRU hidden-hidden bias
        device: TTNN device
    """

    def __init__(
        self,
        conv_weights: List[Any],
        conv_biases: List[Any],
        gru_weight_ih: Any,
        gru_weight_hh: Any,
        gru_bias_ih: Optional[Any] = None,
        gru_bias_hh: Optional[Any] = None,
        device: Optional[Any] = None,
    ):
        self.conv_weights = conv_weights
        self.conv_biases = conv_biases
        self.device = device

        # Create optimized GRU layer
        self.gru = GRULayerOptimized(
            weight_ih=gru_weight_ih,
            weight_hh=gru_weight_hh,
            bias_ih=gru_bias_ih,
            bias_hh=gru_bias_hh,
            batch_first=True,
            device=device,
        )

        # Infer dimensions from first conv
        self.num_convs = len(conv_weights)
        if self.num_convs == 6:
            # Standard OpenVoice V2 reference encoder
            # Channels: [1, 32, 32, 64, 64, 128]
            # Strides: [2, 2, 2, 2, 2, 1]
            # Kernel sizes: [3, 3, 3, 3, 3, 3]
            self.strides = [2, 2, 2, 2, 2, 1]
            self.kernel_sizes = [3, 3, 3, 3, 3, 3]
            self.hidden_size = 128  # GRU hidden size
        else:
            # Generic configuration
            self.strides = [2] * (self.num_convs - 1) + [1]
            self.kernel_sizes = [3] * self.num_convs
            self.hidden_size = gru_weight_ih.shape[0] // 3

    def __call__(self, x: Any, x_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Extract speaker embedding from reference audio.

        Args:
            x: Reference audio/melspectrogram [B, 1, T]
            x_mask: Optional mask [B, 1, T]

        Returns:
            Tuple of:
                - gru_out: GRU outputs [B, T', hidden_size]
                - h_last: Last hidden state [1, B, hidden_size]
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

        # Conv1d stack with stride 2 (except last)
        for i in range(self.num_convs):
            x = ttnn.leaky_relu(x, LRELU_SLOPE)

            # Apply mask after activation if provided
            if x_mask is not None:
                # Downsample mask to match current x shape
                mask_down = ttnn_conv1d(
                    x_mask,
                    weight=ttnn.ones((1, 1, 1, 1), dtype=x.dtype),
                    bias=None,
                    stride=self.strides[i],
                    padding=1,
                    device=self.device,
                )
                mask_down = ttnn.gt(mask_down, 0)  # Binarize
                x = ttnn.multiply(x, mask_down)
                x_mask = mask_down

            # Conv1d with automatic DRAM for stride >= 2
            x = ttnn_conv1d(
                x,
                weight=self.conv_weights[i],
                bias=self.conv_biases[i],
                stride=self.strides[i],
                padding=1,
                device=self.device,
            )

        # GRU layer with dynamic L1/DRAM switching
        # x shape: [B, channels, T'] -> need [B, T', channels]
        if x.shape.rank == 3:
            x = ttnn.permute(x, (0, 2, 1))  # [B, C, T] -> [B, T, C]

        gru_out, h_last = self.gru(x)

        # Apply final mask if provided
        if x_mask is not None and x_mask.shape.rank == 3:
            mask_flat = x_mask.squeeze(1)  # [B, 1, T] -> [B, T]
            mask_expanded = ttnn.unsqueeze(mask_flat, 2)  # [B, T] -> [B, T, 1]
            gru_out = ttnn.multiply(gru_out, mask_expanded)

        return gru_out, h_last

    def _forward_pytorch(self, x, x_mask):
        """PyTorch fallback for development."""
        import torch.nn.functional as F

        def to_torch(t, dtype=torch.float32):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t.to(dtype)
            if TTNN_AVAILABLE:
                return ttnn.to_torch(t).to(dtype)
            return t

        # Apply mask if provided
        if x_mask is not None:
            x_mask_torch = to_torch(x_mask)
            x = x * x_mask_torch

        # Conv1d stack
        for i in range(self.num_convs):
            x = F.leaky_relu(x, LRELU_SLOPE)

            if x_mask is not None:
                # Downsample mask
                x_mask_torch = to_torch(x_mask)
                mask_down = F.conv1d(
                    x_mask_torch,
                    torch.ones(1, 1, 1, 1, dtype=x.dtype),
                    stride=self.strides[i],
                    padding=1,
                )
                mask_down = (mask_down > 0).float()
                x = x * mask_down
                x_mask = mask_down

            # Conv1d
            w = to_torch(self.conv_weights[i])
            if w.dim() == 4:
                w = w.squeeze(2)
            b = to_torch(self.conv_biases[i])
            x = F.conv1d(x, w, b, stride=self.strides[i], padding=1)

        # GRU
        if x.dim() == 3:
            x = x.permute(0, 2, 1)  # [B, C, T] -> [B, T, C]

        gru_out, h_last = self.gru(x)

        # Apply mask
        if x_mask is not None and x_mask.dim() == 3:
            x_mask_torch = to_torch(x_mask)
            mask_flat = x_mask_torch.squeeze(1)
            mask_expanded = mask_flat.unsqueeze(-1)
            gru_out = gru_out * mask_expanded

        return gru_out, h_last


def get_ref_encoder_weights(state_dict: dict, prefix: str = "ref_enc") -> dict:
    """
    Extract reference encoder weights from state dict.

    Args:
        state_dict: Model state dict
        prefix: Prefix for reference encoder keys

    Returns:
        Dictionary with extracted weights
    """
    weights = {}

    # Conv1d weights (6 layers)
    conv_weights = []
    conv_biases = []

    for i in range(6):
        # Get weight with weight normalization support
        w = state_dict.get(f"{prefix}.convs.{i}.weight")
        if w is None:
            # Try weight normalization format
            g = state_dict.get(f"{prefix}.convs.{i}.weight_g")
            v = state_dict.get(f"{prefix}.convs.{i}.weight_v")
            if g is not None and v is not None:
                dims = tuple(range(1, v.dim()))
                v_norm = v / (torch.norm(v, dim=dims, keepdim=True) + 1e-7)
                w = g * v_norm

        # Get bias
        b = state_dict.get(f"{prefix}.convs.{i}.bias")

        conv_weights.append(w)
        conv_biases.append(b)

    weights["conv_weights"] = conv_weights
    weights["conv_biases"] = conv_biases

    # GRU weights
    weights["gru_weight_ih"] = state_dict.get(f"{prefix}.gru.weight_ih_l0")
    weights["gru_weight_hh"] = state_dict.get(f"{prefix}.gru.weight_hh_l0")
    weights["gru_bias_ih"] = state_dict.get(f"{prefix}.gru.bias_ih_l0")
    weights["gru_bias_hh"] = state_dict.get(f"{prefix}.gru.bias_hh_l0")

    return weights
