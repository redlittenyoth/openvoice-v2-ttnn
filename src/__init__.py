# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
OpenVoice V2 TTNN optimized modules.

This package provides optimized implementations of TTNN operations
with dynamic L1/DRAM memory management to avoid CPU fallbacks.
"""

from .gru_optimized import (
    ttnn_gru_cell_optimized,
    ttnn_gru_optimized,
    GRULayerOptimized,
)

from .conv1d_optimized import (
    ttnn_conv1d_optimized,
    ttnn_conv_transpose1d_optimized,
    Conv1dLayerOptimized,
)

from .reference_encoder_optimized import (
    ReferenceEncoderOptimized,
    get_ref_encoder_weights,
)

from .duration_predictor_optimized import (
    DurationPredictorOptimized,
    ttnn_layernorm,
    get_duration_predictor_weights,
)

from .ops import (
    ttnn_sequence_mask,
    ttnn_generate_path,
)

__all__ = [
    # GRU
    "ttnn_gru_cell_optimized",
    "ttnn_gru_optimized",
    "GRULayerOptimized",
    # Conv1d
    "ttnn_conv1d_optimized",
    "ttnn_conv_transpose1d_optimized",
    "Conv1dLayerOptimized",
    # Reference Encoder
    "ReferenceEncoderOptimized",
    "get_ref_encoder_weights",
    # Duration Predictor
    "DurationPredictorOptimized",
    "ttnn_layernorm",
    "get_duration_predictor_weights",
    # Alignment
    "ttnn_sequence_mask",
    "ttnn_generate_path",
]
