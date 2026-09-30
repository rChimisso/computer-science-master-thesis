# Matched classical and quantum reservoirs for event streams

This repository implements a master's thesis study on SHD and DVS Gesture. Its central comparison gives quantum reservoirs (QRC) and matched classical reservoirs (CRC) the same numerical inputs, temporal summaries and readout procedure. Large CRC, LSTM and Transformer models provide independently optional practical references.

[Browse the completed study report](output/runs/thesis-verification/reports/index.md). It contains numerical tables, figures, captions and coverage information for the saved experiment. A fresh run produces its own report under `output/runs/<run-id>/reports/`. Published reports can be read without datasets or model packages; regenerating them requires the local run manifests and evidence, which are not tracked in Git.

## Installation and first checks

Use Python $3.11$ on Linux `x86-64`. Follow the [CPU or GPU installation recipes](environment/README.md). Both support the core study and optional models. GPU execution requires the [custom Aer build and native libraries](environment/BUILD.md), an appropriate NVIDIA driver, and compatible hardware. A fresh clone does not contain ignored wheels, datasets, caches or trained models.

Activate the environment you installed. On the original laptop it is named `qrc-spiking`:

```bash
conda activate qrc-spiking
python -m src plan --profile core
python -m src validate --suite quick
```

Run commands from the repository root. `plan` resolves the requested studies and shows screening counts, conditional work, resource reservations and admission errors without downloading data, starting workers or loading numerical backends. It does not provide the live scheduler's completion forecast. `validate` uses synthetic inputs; it does not evaluate the real official benchmarks.

The checked-in execution policy uses CPU and GPU backends. Installing the CPU recipe does not change that policy: the [environment guide](environment/README.md#execution-scope) explains the configuration changes needed for CPU-only execution.

## Scientific design and parameter choices

The main representation is SHD $32$-channel `log1p` counts in $10\,\mathrm{ms}$ bins with an audited $1.4\,\mathrm{s}$ duration policy, and DVS $8\times8$ spatial bins with separate polarities and $50\,\mathrm{ms}$ `log1p` counts. DVS clips retain their full lengths. Padding is excluded from reservoir summaries, recurrent outputs, attention and pooling.

Main input adapters are contiguous pooling for SHD and training-fitted PCA for DVS. Training-fitted coordinate standardization precedes $p=\sigma(0.5z)$. Both members of a reservoir pair receive those exact probabilities; QRC does not apply another projection or sigmoid. The small input budget is part of the comparison, not a claim that it is optimal for either dataset.

Input/memory allocations are $2/2$, $2/4$, $4/2$, $2/6$, $4/4$ and $6/2$. QRC uses partial input reset, amplitude-probability preparation, fixed seeded rotations, a CZ ring, and exact local XYZ observations. Natural matched CRC width is $3$ times total qubits. Fixed-observation and projection-only controls distinguish some input/feature-width effects. Changing allocation also changes topology and the input/memory balance; these controls do not make every internal property identical.

Fixed and separately tuned comparisons answer different questions. Fixed anchors use QRC depth $1$ for SHD and $3$ for DVS; CRC uses radius $0.9$, input scale $0.03$ and a $20\,\mathrm{ms}$ timescale. Discrete leak is derived as $1-\exp(-\Delta t/\tau)$. The declared QRC search varies depth $\{1,2,3\}$; CRC varies radius, input scale and timescale. Search sizes and costs are unequal. Core readouts use Cholesky with $\alpha=n_{\mathrm{train}}\lambda$, training-only standardization and a shared penalty grid. Supporting comparisons retain their declared protocols, including LSQR where specified.

Selection uses four held-speaker/subject folds within development training and seeds $42,44,46$. Screening precedes replication. Mean fold macro-$F_1$ determines selection; the $0.005$ equivalence band and model-specific tie-breaking are operational rules, not significance tests. Neural recipe adoption additionally requires the declared paired improvement and fold consistency. Confirmation and official evaluation use frozen settings. Earlier use of the development/confirmation populations and repeated inspection of official benchmarks limit their independence.

[The parameter register](configs/parameters.json) records values, origins, rationales, references and implementation departures. Literature-informed, development-selected, derived, controlled and practical choices remain distinct. Exact search bounds and implementation defaults are not claimed to be globally optimal.

The large CRC's $256$-unit size was fixed for the practical reference. A bounded earlier SHD screen selected the radius, scale and timescale subsequently retained as fixed reference dynamics. Its complete compact candidate evidence is embedded in the register. It used a different, previously used speaker holdout and LSQR; it does not establish a current inner-fold or DVS-specific optimum. DVS transfers the physical timescale with its own bin-width-derived leak.

The LSTM is paper-informed, not an exact reproduction: its $128$-unit architecture and Adamax family follow the recorded [SHD reference](https://arxiv.org/abs/1910.07407), while output dropout replaces recurrent dropout for fused execution. Representation, initialization, duration handling and stopping also differ. The compact Transformer dimensions are practical choices. These richer-input models answer a different question from the small matched reservoir pairs.

## Running the study

```bash
python -m src run --profile core --run-id core-study
python -m src run --profile thesis --models large-crc lstm transformer --run-id complete-study
python -m src run --profile thesis --models large-crc lstm transformer --resources --run-id costs-study
```

`--run-id` sets a single directory name; omission uses a UTC timestamp. Both datasets and all declared seeds are included by default. `--dataset shd` or `--dataset dvs` narrows the request. Practical models and resource measurements are disabled unless requested; `--resources` does not implicitly enable models.

The `core` profile contains fixed/tuned matched comparisons, projection-only and fixed-observation controls, and training/held-group diagnostics. `thesis` additionally includes mapping, joint-observation, recurrence and SHD training-size studies. The supported `--study` values are `comparison`, `mapping`, `observations`, `recurrence`, `generalization`, `learning-curves` and `preprocessing`. The core comparison accompanies supporting studies.

Preprocessing is a separate SHD diagnostic and requires the LSTM:

```bash
python -m src run --study preprocessing --models lstm --dataset shd --run-id preprocessing-study
```

Datasets are downloaded only when an action needs them. Acquisition uses provider checksums, schema checks and safe extraction. Raw files live in `data/SHD/` and `data/DVSGesture/`; derived representations and features live in ignored shared caches. Development preparation never opens official tests.

A full `run` includes frozen official evaluation after development selection, confirmation and any requested resource measurements. Use `--development-only` to stop after confirmation. The preprocessing-only diagnostic never opens official tests. Every final official-training fit completes before official-test data are opened. Confirmation and official scores never reopen selection automatically.

## Execution, monitoring and resumption

Execution launches a detached coordinator and attaches a read-only monitor. Closing the terminal or pressing Ctrl+C disconnects that monitor; computation continues. Shutdown or reboot stops computation, and resumption is explicit.

```bash
python -m src run --profile core --run-id core-study --detach
python -m src status --run core-study
python -m src status --run core-study --watch
python -m src pause --run core-study
python -m src run --profile core --run-id core-study --resume
```

`pause` requests cancellation and waits for coordinator exit. Committed feature samples and neural epochs are reusable; in-flight work may be repeated. Unexpected failures pause the run with their error rather than silently retrying correctness failures. Execution locks prevent duplicate coordinators. Multiple monitors may observe the same run.

The monitor shows completed/pending task counts, dataset/study summaries and each active task's device and CPU/GPU/memory reservations. Reservations are not utilization. Waiting reasons are not displayed. Completion forecasts use live throughput, compatible measured timings and conservative allowances for unresolved work. They are provisional ranges, not confidence intervals. Task-count fractions are not elapsed-time percentages; paused time is excluded from rates.

| Control | Meaning | Default |
|---|---|---|
| `--cpu-workers` | Concurrent CPU computation tasks | $2$ |
| `--cpu-threads` | Total managed native-thread budget, including GPU host work | Logical CPUs minus $2$, capped at $8$, minimum $1$ |
| `--gpu-workers` | Concurrent Aer GPU tasks | $1$ |
| `--detach` | Launch without attaching a monitor | Disabled |

Worker limits are ceilings. Per-task numerical thread settings, memory reservations and dependencies can leave workers idle. Neural fits reserve the GPU exclusively; independent CPU work can overlap. Neural batch calibration and resource measurements run in isolation. Spawned processes isolate RNG and numerical-library settings. Compatible feature requests share extraction and one writer.

On the original laptop, the [recorded worker-policy calibration](environment/verification.json) selected $2$ CPU workers, $12$ threads and $1$ GPU worker. Median mixed-workload elapsed times were $118.40$, $114.61$ and $90.43\,\mathrm{s}$ for policies $1/8/1$, $2/8/1$ and $2/12/1$, each with $3$ repetitions. This supports the local override below, not a portable or full-campaign speedup guarantee.

```bash
python -m src plan --profile thesis --cpu-workers 2 --cpu-threads 12 --gpu-workers 1
python -m src run --profile core --run-id core-study --resume --cpu-workers 2 --cpu-threads 12 --gpu-workers 1
```

Use `--deadline` with a quoted `ISO-8601` timestamp including its timezone. The deadline requests checkpointed cancellation; it does not prove the requested work will finish. `--progress auto|plain|off` controls foreground planning, validation and reporting output. Worker progress is recorded structurally and rendered by the coordinator.

Ordinary development resumption requires compatible source, environment and scientific scope. Worker limits and deadlines may change. Seed filters always include screening seed $42$; incomplete seed coverage cannot enter full-coverage summaries or freezing. Resuming an intact frozen run skips development selection and proceeds through saved confirmation/final work after compatibility checks.

## Frozen reproduction and evaluation

A complete development selection produces `frozen.json` with memberships, data fingerprints, settings, decision evidence, source hashes and environment identity. Each run preserves a checksummed implementation/configuration archive. Neural refit duration is the rounded median selected best epoch; final refits do not select checkpoints on their held population.

```bash
python -m src reproduce --manifest output/runs/core-study/frozen.json --run-id replay-study
python -m src evaluate --manifest output/runs/replay-study/frozen.json --official-test
```

`reproduce` skips configuration selection and runs frozen confirmation. `--run-id` gives it an independent output namespace. `evaluate --official-test` continues a frozen run, including one previously stopped with `--development-only`; repeating it resumes committed work.

Frozen compatibility permits explicitly identified presentation/execution changes while protecting scientific implementations. It does not authorize arbitrary source edits or bypass membership, configuration, environment or integrity checks. Historical producer snapshots retain their actual provenance.

## Reporting and retained artifacts

```bash
python -m src report --run complete-study
python -m src report --run complete-study --pdf
python -m src report --run complete-study --details confusion --detail-dataset shd --detail-phase official --detail-role comparison:tuned:q8-i4-qrc
```

Reports are generated from durable evidence without opening datasets, fitted model packages or GPU backends. The command verifies saved predictions, memberships, coverage and frozen relationships. Rendering a report does not imply all scientific phases have completed; pending populations remain explicit.

Each scientific topic contains a reading guide with captions and measurement boundaries, canonical CSV tables, PNG figures and relevant metadata. `--pdf` additionally exports vector figures. There are no LaTeX tables. Default plots consolidate comparisons into panels; per-class/subject values remain in tables and confusion matrices in JSON. Figure-source metadata records the exact rows and transformations.

`--details` accepts `confusion`, `regularization` and `fitting`. `--detail-dataset`, `--detail-phase` and `--detail-role` filter those additional figures only and require `--details`. Canonical tables retain all available results. Regeneration replaces the report and removes stale products. It does not generate authored conclusions or modify scientific evidence.

| Location | Purpose and retention |
|---|---|
| `configs/` | Core settings, optional neural settings and self-contained parameter provenance |
| `environment/` | Pinned recipes, Aer build instructions and compact imported verification records |
| `src/` | CLI/workflow, study definitions, data, models, features, evaluation, reporting and runtime services |
| `tests/` | Focused scientific/workflow checks using synthetic data |
| `data/` | Ignored raw datasets; `.gitkeep` preserves the directory in a clone |
| `output/datasets/`, `output/features/`, `output/locks/` | Ignored shared derived-data/features and execution locks |
| `output/models/` | Ignored content-addressed final inference pipelines, without optimizer states |
| `output/runs/<run-id>/evidence/` | Ignored memberships, predictions, candidates, histories, diagnostics and provenance |
| `output/runs/<run-id>/reports/` | Trackable numerical reports and figures |
| `output/runs/<run-id>/temporary/` | Ignored candidate models and resumable working state |
| `output/runs/<run-id>/execution/` | Ignored live status and worker logs |

Run manifests, progress and logs also remain local. Reports alone cannot regenerate themselves: retain or transfer their manifests and evidence separately. Final inference packages retain preprocessing, fitted adapters, normalization, class order and model/reservoir/readout parameters. Feature caches support retraining readouts but are not required to read or regenerate reports.

Successful completion removes only verified temporary work whose consumers have finished and records intentional pruning. Failed/interrupted work remains resumable. Shared caches and final models are not automatically deleted.

## Validation and interpretation limits

See [the test guide](tests/README.md) for the $24$ quick and $16$ integration checks and their coverage limits.

```bash
python -m src validate --suite all --list
python -m src validate --suite quick
python -m src validate --suite all --gpu
```

`--list` does not run tests or access devices. Validation records go to `output/runs/checks-*/validation.json`. Without `--gpu`, child processes hide CUDA; with it, an actual working GPU is required. The extended suite already includes a synthetic complete workflow, so adding `--smoke` to it is unnecessary. No validation command evaluates real official datasets.

Interpret model comparisons within their protocols:

- Report accuracy, macro-$F_1$, macro-precision and macro-recall. Across-seed uncertainty uses sample standard deviation; folds measure held-group variation and are not independent replicated datasets.
- Confirmation and previously inspected official benchmarks are distinct from development selection. SHD seen/unseen-speaker results are descriptive subgroups of the same frozen prediction set. Whole-test macro-$F_1$ is computed on whole-test predictions, not averaged from subgroup macro-$F_1$.
- Input compression, model capacity, initialization, fitting sample count and evaluated populations can change together across historical studies. Better scores do not identify one cause without the corresponding controlled comparison.
- Joint-observable and recurrence interventions concern accessible information in these pipelines. Correlation is not an entanglement witness, and high training fit is not a proof of superior model-family expressiveness.
- Exact simulated expectations do not collapse the carried quantum state. Real devices require finite-shot estimation, repeated preparation and management of noise/backaction. No quantum hardware speed, energy or entanglement advantage is established.
- Practical costs retain their recorded device, population, warmup, cache-reuse and memory boundaries. Cached CRC readout fitting is not cold extraction plus fitting. Inference timings use development-confirmation workloads, while joined official scores use separately refitted pipelines. Simulator telemetry is not quantum-hardware cost.
