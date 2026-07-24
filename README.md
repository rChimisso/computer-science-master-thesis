# Matched classical and quantum reservoir computing on spike data

## Frozen research protocol

### Research question

How does Quantum Reservoir Computing (QRC) compare with Classical Reservoir Computing (CRC) in spike-based temporal classification when both models receive the same input representation and train only a classical linear readout?

The primary comparison asks whether a fixed, untrained quantum reservoir creates useful temporal features relative to an equally controlled classical reservoir. Secondary analyses measure sensitivity to temporal bin size, adjacent-channel compression, reservoir-feature dimension, reservoir size, and finite-shot QRC measurements. A positive result means better validated predictive performance or useful performance with fewer output features; it does not establish quantum advantage.

### Dataset and representation

The primary dataset is Spiking Heidelberg Digits (SHD): $20$ spoken-digit classes encoded as events over $700$ artificial-cochlea channels. SHD is genuinely event-based, but it was generated from conventional audio by an artificial cochlea and was not captured by a physical neuromorphic sensor. The raw format is preserved as Tonic's structured NumPy events `(t, x, p)` in microseconds, where `x` is the cochlear channel and the synthetic polarity satisfies $p=1$.

DVS Gesture is deferred. It remains the first extension because it was recorded with event-camera hardware, but its `(t, x, y, p)` events require substantial spatial pooling before a small QRC can consume them. SSC, learned input feature extractors, trainable quantum circuits, additional reservoirs, and real quantum hardware are outside the core scope. See [dataset_decision.md](dataset_decision.md) for the feasibility evidence and decision.

### Frozen split and evaluation rules

The official SHD test set is used exactly once for final reporting and is never used for model or preprocessing selection. Speakers $0$ and $1$ from the official training set form a fixed, speaker-disjoint validation set; the remaining training speakers form the training set. This yields $6\,340$ training and $1\,816$ validation examples. Split membership is determined only from the official speaker metadata and is therefore deterministic.

All model selection uses validation accuracy and macro-$F_1$. Final reporting will include accuracy, macro-$F_1$, per-class $F_1$, confusion matrices, training and inference time, the number of trainable readout parameters, and mean and standard deviation over at least five fixed seeds. The matched CRC and QRC must consume identical saved tensors and use the same linear-readout family. No hyperparameter may be selected on the official test set.

### Models included

The core study contains one Echo State Network-style CRC, one fixed-circuit QRC, and ridge or logistic-regression linear readouts. A count-based linear classifier is included only as a preprocessing sanity baseline. The primary result uses the same compressed input representation; a second comparison approximately matches the reservoir-feature dimensions. A larger CRC may be reported separately as an unmatched classical reference.

The core QRC uses statevector simulation. Finite-shot measurement is the first planned extension. Simulator runtime is reported as classical computational cost and is not interpreted as quantum-hardware efficiency.

### Explicit exclusions

The initial study excludes DVS Gesture experiments, SSC, SNN/CNN baselines, learned preprocessing, trained reservoir weights, trainable quantum circuits, noise and hardware experiments, and claims of quantum advantage. These can only be reconsidered after the complete SHD comparison and its figures are reproducible.

## Reproducible setup

The project uses the existing `qrc-spiking` Conda environment and Python `3.11`. Pinned package versions are in `requirements.txt`; do not install or upgrade packages implicitly. On another machine, reproduce and verify it with:

```bash
conda env create -f environment.yml
conda activate qrc-spiking
python -m src.utils.environment_check
python -m unittest discover -s tests
```

The environment check verifies Python, NumPy, scikit-learn, Tonic, ReservoirPy, Qiskit, configuration loading, deterministic seeding, and JSON result logging.

## Dataset feasibility workflow

Raw downloads and generated results are intentionally ignored by Git. Commands create the required directories when they are missing. Tonic stores candidate datasets below `data/raw/`. To reproduce the dataset feasibility report and ten-sample visual checks, run:

```bash
conda activate qrc-spiking
python scripts/run_dataset_feasibility.py --datasets shd dvs
```

If DVS Gesture is intentionally omitted, use `--datasets shd`. The command writes summaries to `results/dataset_feasibility/` without modifying raw samples. The notebook [00_dataset_feasibility.ipynb](notebooks/00_dataset_feasibility.ipynb) provides the same analysis interactively; the reusable implementation lives in `src/data/feasibility.py`. The first DVS run downloads a $2.4\,\mathrm{GB}$ training archive. The loader uses the byte-identical Zenodo copy because the downloader identifiers bundled with Tonic `1.6.0` are no longer valid.

## Repository layout

```text
.
├── configs/
│   └── default.yaml
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   └── 00_dataset_feasibility.ipynb
├── results/
├── scripts/
│   └── run_dataset_feasibility.py
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── feasibility.py
│   │   └── splits.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── environment_check.py
│   │   ├── reproducibility.py
│   │   └── results.py
│   └── __init__.py
├── tests/
│   ├── test_feasibility.py
│   └── test_foundation.py
├── dataset_decision.md
├── environment.yml
├── README.md
└── requirements.txt
```
