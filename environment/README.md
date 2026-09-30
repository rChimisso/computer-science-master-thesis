# Runtime environment

Both the matched QRC/CRC workflow and the optional large CRC, LSTM and Transformer use the same dependencies. CRC already requires PyTorch. Choose a CPU or GPU installation according to the intended backend; model selection remains a workflow option.

The recipes target Python $3.11$ on Linux x86-64. They contain the required package dependency closure with exact versions, excluding unrelated notebooks and development tools. Dataset acquisition and report generation are included. The audio dependency chain comes from Tonic, which the dataset acquisition code uses.

| File | Purpose |
|---|---|
| `common.txt` | Shared direct and indirect runtime dependencies. |
| `cpu.txt` | Shared dependencies, CPU PyTorch and the public CPU Aer wheel. |
| `gpu.txt` | Shared dependencies, CUDA PyTorch, CUDA runtime dependencies and the local GPU Aer wheel. |
| `build.txt` | Python dependencies used only when compiling Aer. |
| `BUILD.md` | Rebuilding the pinned GPU Aer sources and native dependencies. |
| `aer-build.json` | Source URLs, hashes, build settings and preserved binary identity. |
| `local/` | Ignored wheels, native libraries and disposable build products. |

## Installation

Run installation commands from the repository root, because the GPU wheel path is repository-relative. Use a fresh environment; these instructions do not upgrade or replace the existing `qrc-spiking` environment. Conda with Python $3.11$ or a Python virtual environment can provide the interpreter.

For CPU execution:

```bash
python3.11 -m venv /tmp/thesis-cpu
/tmp/thesis-cpu/bin/python -m pip install -r environment/cpu.txt
/tmp/thesis-cpu/bin/python -m pip check
/tmp/thesis-cpu/bin/python -m src validate --suite extended --progress off
```

For GPU execution, first rebuild Aer using [BUILD.md](BUILD.md), or restore a compatible wheel and its matching native libraries. Then:

```bash
python3.11 -m venv /tmp/thesis-gpu
/tmp/thesis-gpu/bin/python -m pip install -r environment/gpu.txt
/tmp/thesis-gpu/bin/python -m pip check
/tmp/thesis-gpu/bin/python -m src validate --suite extended --gpu --progress off
```

GPU execution needs an NVIDIA driver compatible with the CUDA $12.6$ runtime and a GPU supported by the Aer build. The preserved build targets compute capability $8.6$. Installing `gpu.txt` does not install a driver. GPU requests fail explicitly when unavailable; they do not silently select CPU.

The CPU recipe does not require the local Aer wheel or native directory. GPU verification additionally checks that Aer reports the requested device and that CPU/GPU numerical results agree. Validation uses synthetic data; it is not a new scientific evaluation.

## Execution scope

Activate the chosen environment before running the workflow:

```bash
python -m src plan --profile core
python -m src run --profile core
python -m src run --profile thesis --models large-crc lstm transformer
```

Installation does not change execution settings. For CPU execution, copy `configs/core.yaml` and set these entries in its `execution` section, preserving the other settings:

```yaml
qrc_small: CPU
qrc_large: CPU
crc_device: cpu
crc_graphs: false
device: cpu
```

Pass the copy with `--config PATH`. The checked-in settings mix CPU and GPU backends intentionally, so installing the CPU recipe alone is insufficient to run those unchanged settings. Plan and report commands do not initialize GPU backends.

## Local artifacts and reproducibility

`local/wheels/` holds the Aer wheel; `local/native/` holds its required native runtime. The optional `local/build/` holds downloaded sources, the build environment and temporary compiler products. It can be removed after successful installation and runtime verification. All of `local/` is Git-ignored, so a clone requires rebuilding or separately restoring the native artifacts.
