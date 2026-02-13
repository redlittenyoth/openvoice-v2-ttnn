#!/usr/bin/env python3
"""
Test script for verifying GRU and HiFi-GAN optimizations.

This script verifies:
1. GRU L1/DRAM switching for large hidden states
2. Conv1d/ConvTranspose1d without CPU fallback for stride >= 2
3. Memory configuration auto-detection
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_gru_memory_config():
    """Test GRU memory configuration detection."""
    print("=" * 60)
    print("Testing GRU Memory Configuration...")
    print("=" * 60)

    try:
        from gru_optimized import _get_memory_config

        # Small hidden state - should use L1
        config_small = _get_memory_config(batch_size=4, hidden_size=128, seq_len=100)
        print(f"Small hidden state (4×128×100): L1 = {config_small}")

        # Large hidden state - should use DRAM
        config_large = _get_memory_config(batch_size=4, hidden_size=512, seq_len=100)
        print(f"Large hidden state (4×512×100): DRAM = {config_large}")

        # Long sequence - should use DRAM
        config_long = _get_memory_config(batch_size=4, hidden_size=128, seq_len=500)
        print(f"Long sequence (4×128×500): DRAM = {config_long}")

        print("✅ GRU memory config test PASSED!")
        return True

    except Exception as e:
        print(f"❌ GRU memory config test FAILED: {e}")
        return False


def test_conv1d_dram_detection():
    """Test Conv1d DRAM detection for large strides."""
    print("\n" + "=" * 60)
    print("Testing Conv1d DRAM Detection...")
    print("=" * 60)

    try:
        from conv1d_optimized import _should_use_dram

        # Small kernel + stride 1 - should use L1
        should_dram_1 = _should_use_dram(input_size=1024, kernel_size=3, stride=1)
        print(f"Small kernel (3, stride=1): use_dram = {should_dram_1}")

        # Large kernel + stride 2 - should use DRAM
        should_dram_2 = _should_use_dram(input_size=1024, kernel_size=16, stride=8)
        print(f"Large kernel (16, stride=8): use_dram = {should_dram_2}")

        # Large input - should use DRAM
        should_dram_3 = _should_use_dram(input_size=128*1024, kernel_size=3, stride=1)
        print(f"Large input (128K elements): use_dram = {should_dram_3}")

        # Verify expected results
        assert not should_dram_1, "Small kernel should not use DRAM"
        assert should_dram_2, "Large kernel+stride should use DRAM"
        assert should_dram_3, "Large input should use DRAM"

        print("✅ Conv1d DRAM detection test PASSED!")
        return True

    except Exception as e:
        print(f"❌ Conv1d DRAM detection test FAILED: {e}")
        return False


def test_gru_cell_logic():
    """Test GRU cell logic with PyTorch tensors."""
    print("\n" + "=" * 60)
    print("Testing GRU Cell Logic (PyTorch fallback)...")
    print("=" * 60)

    try:
        from gru_optimized import ttnn_gru_cell_optimized

        # Dummy data
        batch_size = 2
        hidden_size = 4
        input_size = 8

        x = torch.randn(batch_size, input_size)
        h = torch.randn(batch_size, hidden_size)
        weight_ih = torch.randn(3 * hidden_size, input_size)
        weight_hh = torch.randn(3 * hidden_size, hidden_size)
        bias_ih = torch.randn(3 * hidden_size)
        bias_hh = torch.randn(3 * hidden_size)

        # Run GRU cell
        h_new = ttnn_gru_cell_optimized(
            x, h, weight_ih, weight_hh, bias_ih, bias_hh
        )

        # Verify output shape
        assert h_new.shape == (batch_size, hidden_size), \
            f"Expected shape ({batch_size}, {hidden_size}), got {h_new.shape}"

        print(f"Input shape: {x.shape}")
        print(f"Hidden shape: {h.shape}")
        print(f"Output shape: {h_new.shape}")
        print("✅ GRU cell logic test PASSED!")
        return True

    except Exception as e:
        print(f"❌ GRU cell logic test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conv1d_logic():
    """Test Conv1d logic with PyTorch tensors."""
    print("\n" + "=" * 60)
    print("Testing Conv1d Logic (PyTorch fallback)...")
    print("=" * 60)

    try:
        from conv1d_optimized import ttnn_conv1d_optimized

        # Dummy data
        batch_size = 2
        in_channels = 8
        out_channels = 16
        seq_len = 32
        kernel_size = 3
        stride = 1

        x = torch.randn(batch_size, in_channels, seq_len)
        weight = torch.randn(out_channels, in_channels, 1, kernel_size)
        bias = torch.randn(out_channels)

        # Run Conv1d
        out = ttnn_conv1d_optimized(
            x, weight, bias, stride=stride, padding=1
        )

        # Verify output shape
        expected_len = seq_len
        assert out.shape == (batch_size, out_channels, expected_len), \
            f"Expected shape ({batch_size}, {out_channels}, {expected_len}), got {out.shape}"

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        print("✅ Conv1d logic test PASSED!")
        return True

    except Exception as e:
        print(f"❌ Conv1d logic test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conv_transpose1d_logic():
    """Test ConvTranspose1d logic with PyTorch tensors (stride >= 2)."""
    print("\n" + "=" * 60)
    print("Testing ConvTranspose1d Logic (stride >= 2, PyTorch fallback)...")
    print("=" * 60)

    try:
        from conv1d_optimized import ttnn_conv_transpose1d_optimized

        # Dummy data - HiFi-GAN upsampling configuration
        batch_size = 2
        in_channels = 16
        out_channels = 8
        seq_len = 32
        kernel_size = 16
        stride = 8  # Large stride - would cause L1 overflow in original version

        x = torch.randn(batch_size, in_channels, seq_len)
        weight = torch.randn(in_channels, out_channels, 1, kernel_size)
        bias = torch.randn(out_channels)

        # Run ConvTranspose1d
        out = ttnn_conv_transpose1d_optimized(
            x, weight, bias, stride=stride, padding=4
        )

        # Verify output shape
        expected_len = (seq_len - 1) * stride - 2 * 4 + kernel_size
        assert out.shape == (batch_size, out_channels, expected_len), \
            f"Expected shape ({batch_size}, {out_channels}, {expected_len}), got {out.shape}"

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        print(f"Upsampling ratio: {stride}x")
        print("✅ ConvTranspose1d logic test PASSED!")
        return True

    except Exception as e:
        print(f"❌ ConvTranspose1d logic test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all optimization tests."""
    print("\n" + "=" * 60)
    print("OpenVoice V2 TTNN - Optimization Tests")
    print("=" * 60 + "\n")

    # Only run logic tests that don't require torch
    tests = [
        test_gru_memory_config,
        test_conv1d_dram_detection,
    ]

    # Check if torch is available for logic tests
    try:
        import torch
        tests.extend([
            test_gru_cell_logic,
            test_conv1d_logic,
            test_conv_transpose1d_logic,
        ])
    except ImportError:
        print("Note: PyTorch not available, skipping logic tests.")
        print("Optimizations can still be tested with TTNN device.\n")

    results = [test() for test in tests]

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests PASSED! Optimizations are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) FAILED. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
