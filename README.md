# OpenVoice V2 TTNN Full On-Device (Bounty $1,500)

[![GitHub](https://img.shields.io/badge/GitHub-redlittenyoth%2Fopenvoice--v2--ttnn-brightgreen)](https://github.com/redlittenyoth/openvoice-v2-ttnn)

This project implements a **100% on-device** inference pipeline for OpenVoice V2 on Tenstorrent hardware.

## The Problem
Existing implementations (e.g., PR #36286) rely on CPU fallbacks for:
1.  **Monotonic Alignment**: Generating the expansion path from durations.
2.  **Reference Encoder (GRU)**: Offloaded to CPU due to L1 memory constraints.
3.  **ConvTranspose1d**: Fallback for certain strides in HiFi-GAN.

## Our Solution
-   **On-Device Alignment**: We use a matmul-based `cumsum` implementation to generate the alignment path entirely on the Tenstorrent chip.
-   **Optimized Memory**: We utilize DRAM buffering for long sequences to bypass L1 overflows without dropping back to the host CPU.
-   **Fully TTNN-Compatible**: All operations are expressed via `ttnn` APIs to minimize host-device synchronization.
-   **Dynamic L1/DRAM Switching**: Automatic memory management based on tensor size and sequence length.

## Recent Optimizations (Feb 13, 2026)

### GRU L1 Optimization (`gru_optimized.py`)
- **Problem:** Original implementation used `L1_MEMORY_CONFIG` for all operations, causing overflow for large hidden states (>16K elements).
- **Solution:** Dynamic L1/DRAM switching based on:
  - Hidden state size (batch_size × hidden_size)
  - Sequence length (>256 frames triggers DRAM)
- **Result:** Eliminates GRU fallback to CPU for reference encoder.

### HiFi-GAN Stride-2 Fix (`conv1d_optimized.py`)
- **Problem:** Original `conv1d.py` had hardcoded CPU fallback for `stride >= 2`, breaking on-device upsampling.
- **Solution:**
  - Removed CPU fallback for stride >= 2
  - Added automatic DRAM detection for large kernels/strides (kernel_size × stride > 1024)
  - Uses `DRAM_MEMORY_CONFIG` instead of `L1_MEMORY_CONFIG` for large operations
- **Result:** All HiFi-GAN upsampling layers now run on-device without CPU fallback.

## Structure
-   `src/`: Original modules from the TTNN port.
-   `ops.py`: Custom TTNN operations (Alignment, Sequence Mask).
-   `verify_pipeline.py`: Validation script for our optimized ops.
-   `setup_and_run.sh`: Environment setup script.

## How to Run (on CUDA/Simulator)
1.  Ensure you have `tt-metal` / `ttnn` installed.
2.  Run the setup script:
    ```bash
    chmod +x setup_and_run.sh
    ./setup_and_run.sh
    ```
3.  Verify the alignment logic:
    ```bash
    python3 verify_pipeline.py
    ```

## Bounty Progress
- [x] On-device `_generate_path` implementation.
- [x] Elimination of CPU synchronization points in `_infer_ttnn`.
- [x] GRU L1 optimization with dynamic DRAM buffering.
- [x] HiFi-GAN stride-2 fallback fix (removed CPU fallback, uses on-device DRAM).

---
**Author:** Andrey & Freya 😉
**Target Issue:** [tt-metal#32182](https://github.com/tenstorrent/tt-metal/issues/32182)
