import torch
import torch.nn.functional as F
import math

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

if not TTNN_AVAILABLE:
    # Minimal mock for local logic testing without ttnn
    class ttnn_mock:
        bfloat16 = "bfloat16"
        def from_torch(self, t, **kwargs): return t
        def to_torch(self, t): return t
        def unsqueeze(self, t, dim): return t.unsqueeze(dim)
        def lt(self, a, b): return a < b
        def ge(self, a, b): return a >= b
        def logical_and(self, a, b): return a & b
        def to_dtype(self, t, dtype): return t
        def multiply(self, a, b): return a * b
        def matmul(self, a, b): return torch.matmul(a, b)
        def zeros(self, shape, **kwargs): return torch.zeros(shape)
        def concat(self, tensors, dim): return torch.cat(tensors, dim=dim)
        def reshape(self, t, shape): return t.reshape(shape)
        def permute(self, t, dims): return t.permute(dims)
        def sum(self, t, dim): return t.sum(dim)
        def exp(self, t): return torch.exp(t)
        def ceil(self, t): return torch.ceil(t)
        def randn(self, shape, **kwargs): return torch.randn(shape)
        def add(self, a, b): return a + b
    ttnn = ttnn_mock()
else:
    import ttnn

def ttnn_sequence_mask(lengths, max_len=None, device=None):
    """TTNN version of sequence_mask."""
    if max_len is None:
        # We need max_len on host to create the arange
        lengths_host = ttnn.to_torch(lengths)
        max_len = int(lengths_host.max().item())
    
    # Create range [0, 1, 2, ..., max_len-1]
    range_tensor = torch.arange(max_len, dtype=torch.float32)
    range_tt = ttnn.from_torch(range_tensor, dtype=ttnn.bfloat16, device=device)
    range_tt = ttnn.unsqueeze(range_tt, 0) # [1, max_len]
    
    lengths_tt = ttnn.unsqueeze(lengths, 1) # [B, 1]
    
    # Mask: range < lengths
    mask = ttnn.lt(range_tt, lengths_tt)
    return mask

def ttnn_generate_path(duration, mask, device=None):
    """
    TTNN implementation of monotonic alignment path generation.
    duration: [B, 1, T_x] - number of frames for each input token
    mask: [B, 1, T_y, T_x] - attention mask
    """
    b, _, t_x = duration.shape
    t_y = mask.shape[2]
    
    # Step 1: Compute cumsum of durations via matmul (lower triangular matrix)
    tri = torch.tril(torch.ones(t_x, t_x))
    tri_tt = ttnn.from_torch(tri, dtype=ttnn.bfloat16, device=device)
    dur_cumsum = ttnn.matmul(duration, tri_tt) # [B, 1, T_x]
    
    # Step 2: dur_cumsum_shifted
    zeros = ttnn.zeros((b, 1, 1), dtype=ttnn.bfloat16, device=device)
    dur_cumsum_shifted = ttnn.concat([zeros, dur_cumsum[:, :, :-1]], dim=2)
    
    # Step 3: Compare j_indices [0...T_y-1] with [start, end]
    j_indices = torch.arange(t_y, dtype=torch.float32).view(1, t_y, 1)
    j_tt = ttnn.from_torch(j_indices, dtype=ttnn.bfloat16, device=device)
    
    # Broadcast compare:
    start = ttnn.unsqueeze(dur_cumsum_shifted, 2)
    end = ttnn.unsqueeze(dur_cumsum, 2)
    
    mask_start = ttnn.ge(j_tt, start)
    mask_end = ttnn.lt(j_tt, end)
    
    path = ttnn.logical_and(mask_start, mask_end)
    path = ttnn.to_dtype(path, ttnn.bfloat16)
    
    return ttnn.multiply(path, mask)
