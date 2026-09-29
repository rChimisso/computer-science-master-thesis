# Building GPU Aer

This recipe rebuilds unmodified Aer $0.17.2$ with the CUDA Thrust backend and cuQuantum disabled. It uses the successful original dependency and linker settings. The pinned source URLs and SHA256 digests are in [aer-build.json](aer-build.json); the implementation is [aer_build.py](../src/runtime/aer_build.py). Rebuilding the same source does not promise a byte-identical wheel.

## Prerequisites

The supported recipe targets Ubuntu $24.04$, Linux x86-64 and Python $3.11$. It needs `gcc`, `g++`, CMake, Ninja and `dpkg-deb` on `PATH`. The verification host has GCC $13.3$ and CMake $3.28.3$. These system tools are not Python runtime dependencies. Other operating systems and toolchains have not been validated.

Building downloads the pinned CUDA $12.6$ compiler, runtime headers and CCCL archives. A system CUDA toolkit installation is unnecessary. Execution additionally needs a compatible NVIDIA driver; the build itself does not need GPU access. The default compute capability is $8.6$, matching the RTX $3070$ Ti Laptop GPU. Set `--architecture` to the target GPU's supported compute capability when rebuilding for other hardware; this is not a promise that the recorded CUDA toolkit supports every newer architecture.

## Fresh-clone procedure

Run these commands from the repository root. Use a separate build environment so compilation tools do not become runtime requirements.

```bash
python3.11 -m venv environment/local/build/venv
environment/local/build/venv/bin/python -m pip install -r environment/build.txt
environment/local/build/venv/bin/python -m src.runtime.aer_build --architecture 8.6 --jobs 2
```

The builder:

- Downloads only the sources and native packages listed in the manifest, verifies their hashes and extracts them locally. Debian packages are extracted with `dpkg-deb`, not installed system-wide.
- Builds the recorded JSON and spdlog dependencies, adding the `spdlog` CMake target alias required by Aer.
- Builds Aer with Conan and cuQuantum disabled, the requested CUDA architecture, OpenBLAS and explicit library paths. The `-rpath-link` setting supplies the Fortran dependencies during OpenBLAS detection.
- Saves the wheel in `environment/local/wheels/`, native libraries in `environment/local/native/`, and sources and compiler products in `environment/local/build/`.
- Writes commands, durations, failures and checksums to `environment/local/build/build.json`; a successful build also writes `environment/local/build-result.json`.

The builder does not install Python packages or modify upstream Aer source. It refuses a workspace containing an existing wheel. To preserve a working build, use a separate empty workspace:

```bash
environment/local/build/venv/bin/python -m src.runtime.aer_build --workspace /tmp/thesis-aer-rebuild --architecture 8.6 --jobs 2
```

A separate workspace produces the same `wheels/`, `native/` and `build/` layout. Install its wheel in a disposable runtime environment and validate it before replacing any existing files. Changing a build's source, compiler or architecture creates a new binary identity; retain its own build record.

## Runtime installation and validation

Once the default-location build succeeds, follow the GPU instructions in [README.md](README.md). `gpu.txt` installs the generated local wheel together with its pinned Python and CUDA runtime dependencies. PyTorch's CUDA runtime packages are still required; compiler archives do not replace them.

The application preloads OpenBLAS, `libgfortran` and `libquadmath` from `environment/local/native/` before importing Aer. This also happens in spawned quantum workers. The wheel's original absolute RPATH is not a portable installation mechanism; distribute or rebuild the matching native libraries with the wheel, and use the application loader.

```bash
python -m src validate --suite extended --gpu --progress off
```

Validation requires actual GPU execution and compares CPU/GPU quantum observations and classical features. It also exercises the small synthetic workflow, model retention and report regeneration. It does not run a scientific campaign or open real official-test data.

After successful installation and runtime validation, `environment/local/build/` is disposable. Keep the wheel, runtime libraries and `build-result.json`. Build directories must not be removed while their Python interpreter is still in use.
