import torch
import numpy as np

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    # Mocking ttnn to allow verify_pipeline to run logic tests on Torch
    class ttnn_mock:
        def open_device(self, **kwargs): return "mock_device"
        def close_device(self, device): pass
        def from_torch(self, t, **kwargs): return t
        def to_torch(self, t): return t
        bfloat16 = torch.bfloat16
    ttnn = ttnn_mock()

from ops import ttnn_generate_path, ttnn_sequence_mask

def verify_alignment_op():
    print("Testing ttnn_generate_path (On-Device Alignment)...")
    device_id = 0
    device = ttnn.open_device(device_id=device_id)
    
    # Dummy duration: [B, 1, T_x]
    # 3 tokens, durations 2, 3, 1 frames
    duration_torch = torch.tensor([[[2.0, 3.0, 1.0]]], dtype=torch.float32)
    duration_tt = ttnn.from_torch(duration_torch, dtype=ttnn.bfloat16, device=device)
    
    # Mask: [B, 1, T_y, T_x]
    # T_y = sum(durations) = 6
    t_y = 6
    t_x = 3
    mask_torch = torch.ones((1, 1, t_y, t_x))
    mask_tt = ttnn.from_torch(mask_torch, dtype=ttnn.bfloat16, device=device)
    
    # Run op
    path_tt = ttnn_generate_path(duration_tt, mask_tt, device=device)
    path_torch = ttnn.to_torch(path_tt)
    
    # Expected path:
    # Token 1 (dur 2): frames 0, 1
    # Token 2 (dur 3): frames 2, 3, 4
    # Token 3 (dur 1): frame 5
    expected = torch.zeros((1, 1, 6, 3))
    expected[0, 0, 0:2, 0] = 1
    expected[0, 0, 2:5, 1] = 1
    expected[0, 0, 5:6, 2] = 1
    
    print("Result Path:\n", path_torch[0, 0])
    print("Expected Path:\n", expected[0, 0])
    
    diff = torch.abs(path_torch - expected).max().item()
    if diff < 1e-2:
        print("✅ Alignment OP Verification PASSED!")
    else:
        print(f"❌ Alignment OP Verification FAILED! Max diff: {diff}")

    ttnn.close_device(device)

if __name__ == "__main__":
    try:
        verify_alignment_op()
    except Exception as e:
        print(f"Error during verification: {e}")
        print("Note: This requires ttnn and a compatible device (or simulator).")
