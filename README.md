# OpenVoice V2 TTNN Full On-Device (Bounty $1,500)

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
- [ ] GRU L1 optimization (In Progress).
- [ ] HiFi-GAN stride-2 fallback fix (In Progress).

---
**Author:** Andrey & Freya 😉
**Target Issue:** [tt-metal#32182](https://github.com/tenstorrent/tt-metal/issues/32182)
