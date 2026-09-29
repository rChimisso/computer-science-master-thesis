# Matched classical and quantum reservoirs for event streams

This repository implements the thesis comparison on SHD and DVS Gesture. Matched QRC and CRC pairs are mandatory. Large CRC, LSTM, Transformer and resource measurements are independently optional.

## Start here

Run commands from the repository root in the existing environment:

```bash
conda activate qrc-spiking
python -m src plan --profile thesis
python -m src validate
```

`plan` resolves the finite queue without downloading data or executing experiments. It reports task counts and conditional work without static runtime forecasts. Cleanup did not launch a new full campaign or official evaluation.

Generated thesis tables, figures and documents are indexed in [keeping-track/reports/thesis/index.md](keeping-track/reports/thesis/index.md). Their immutable predictions, fitted artifacts and provenance are indexed in [keeping-track/reference/thesis/index.md](keeping-track/reference/thesis/index.md). They retain their actual protocols. Read [keeping-track/hyperparameter-justification.md](keeping-track/hyperparameter-justification.md) for parameter choices and [keeping-track/repository-cleanup.md](keeping-track/repository-cleanup.md) for migration validation and limitations.

## Workflow

```bash
python -m src run --profile core --run-id core-study
python -m src run --profile thesis --models large-crc lstm transformer --run-id complete-study
python -m src run --profile thesis --models large-crc lstm transformer --resources --run-id costs-study
```

Both datasets are included unless filtered with `--dataset shd` or `--dataset dvs`. Available `--study` values are `comparison`, `mapping`, `observations`, `recurrence`, `generalization`, `learning-curves` and `preprocessing`. The core comparison accompanies supporting studies. Preprocessing is separately runnable and requires the LSTM:

```bash
python -m src run --study preprocessing --models lstm --dataset shd --run-id preprocessing-study
```

The `core` profile contains fixed and tuned matched pairs, projection-only and fixed-observation controls, plus training/held-group diagnostics. The `thesis` profile adds mapping, joint-observation, recurrence and SHD training-size studies. The main adapters remain SHD pooling and DVS PCA. Supporting protocols retain explicit differences, including LSQR where originally declared. Core readouts use aligned Cholesky and sample-normalized penalties.

Resource measurement does not implicitly enable practical models. Every core report derives reservoir structure and quantum hardware requirements without benchmarks. `--resources` adds isolated measurements for enabled practical models after fitting. Simulator timings do not represent quantum hardware cost.

Training data are acquired only when needed. Downloads use the installed dataset provider metadata, archive checksums and safe extraction. Official tests are inaccessible to ordinary development actions. Original datasets live directly under `data/SHD/` and `data/DVSGesture/`; representations and feature banks are under `output/datasets/` and `output/features/`. Dataset contents are ignored by Git; `data/.gitkeep` preserves the empty directory in a checkout.

## Execution and monitoring

Execution commands start a detached coordinator, then attach a read-only monitor. Closing the terminal or pressing Ctrl+C disconnects the monitor and leaves computation running. Add `--detach` to return immediately. Shutdown or reboot stops computation; restarting saved work remains explicit.

```bash
python -m src run --profile core --run-id core-study --detach
python -m src status --run core-study
python -m src status --run core-study --watch
python -m src pause --run core-study
python -m src run --profile core --run-id core-study --resume
```

`pause` requests cancellation and waits for verified coordinator exit. Committed feature samples and neural epochs remain reusable; an unfinished batch or sequence may be recomputed. Unexpected failures pause the whole run with the triggering error. No correctness failure is retried silently. A workspace execution lock rejects duplicate coordinators. Multiple monitors can attach without owning computation. Live state, worker logs and execution requests reside under ignored `output/runs/<run-id>/execution/`, outside sealed scientific evidence.

The monitor shows completed and pending counts for every dataset and, for each study, separately by dataset. Run-wide work has an explicit label. Active tasks show their CPU/Aer/CUDA device, reserved CPU/GPU slots, threads and RAM/VRAM; reservations are not measured utilization. Waiting reasons remain internal scheduler diagnostics and are not displayed. The full completion range includes future stages and conservative fallback allowances. Current throughput takes precedence over comparable completed measurements, then original compatible cache timings, then broad defaults. Forecasts tend toward overestimation and are planning ranges, not confidence intervals. Their evidence mix is shown. No extra ETA-only benchmarks run. Heterogeneous task counts are never percentages of runtime completed; paused time is excluded from processing rates.

| Control | Meaning | Default |
|---|---|---|
| `--cpu-workers` | Concurrent CPU computation tasks | $2$ |
| `--cpu-threads` | Total reserved native threads, including GPU host work | Available logical CPUs minus $2$, capped at $8$, minimum $1$ |
| `--gpu-workers` | Concurrent Aer GPU tasks | $1$ |
| `--detach` | Launch without attaching a monitor | Disabled |

Worker ceilings do not change numerical per-task thread settings. The scheduler reserves both nested Aer and host processing, checks memory with headroom, and may leave workers idle. Neural fits reserve the GPU exclusively while compatible CPU work may continue. Batch-size calibration runs in isolation. Resource measurements drain ordinary work before running. Scientific settings and worker policy are stored separately; changing worker limits on resume does not invalidate compatible features or checkpoints.

When `run --resume` finds an intact frozen configuration matching the requested run, it proceeds directly to confirmation and downstream work. It validates source, environment, declared coverage and dataset memberships without replaying development selection. Training datasets are prepared through the scheduler; work for one dataset can begin while another is loading. The monitor labels saved tasks awaiting registration as `restoring`, and clears waiting reasons from the previous coordinator.

On the current laptop, the [thread-budget comparison](keeping-track/cpu-gpu-budget/README.md) selected $2$ CPU workers, $12$ total threads and $1$ GPU worker. Its complete mixed extraction workload took approximately $24\%$ less time than the previous $1/8/1$ policy. This is a measured laptop override, not a portable default or a full-campaign speedup guarantee. Set `--cpu-workers 2 --cpu-threads 12 --gpu-workers 1` to use it.

```bash
python -m src plan --profile thesis --cpu-workers 2 --cpu-threads 8 --gpu-workers 1
python -m src run --profile core --run-id core-study --resume --cpu-workers 2 --cpu-threads 8 --gpu-workers 1
```

Independent datasets and supporting studies can overlap. Within a study, complete screening precedes replication, seed rounds retain their order, and data-dependent follow-ups wait for their selections. Spawned computation processes isolate numerical libraries and random generators. Only the coordinator writes shared decisions and publishes reports. Cache and dataset writers are locked; identical feature requests share their extraction task.

`--progress auto|plain|off` continues to control foreground planning, validation and reporting messages. Scientific workers always write structured progress files instead of nested terminal bars. The attached run/status monitor owns live rendering.

## Interruption and resumption

```bash
python -m src run --profile core --dataset shd --seed 42 --run-id focused-study
python -m src run --profile core --dataset shd --seed 44 46 --run-id focused-study --resume
python -m src run --profile core --run-id limited-study --deadline 2026-09-24T18:00:00+02:00
```

Screening always includes seed $42$. Later seed rounds reuse completed cells. Partial coverage cannot be frozen or enter complete comparison summaries. Resume requires unchanged source, scientific settings, dataset/study/model scope and environment. Deadlines and scheduling limits may change. The launcher clears a prior cooperative stop request only after acquiring the execution lock.

The one-time transition of `thesis-verification` is recorded in [keeping-track/parallel-transition/README.md](keeping-track/parallel-transition/README.md). Production code accepts one current format; no historical-format reader or migration command is provided.

## Frozen reproduction and evaluation

Development produces `frozen.json` only after complete selection. This contains memberships, data-content fingerprints, settings, decision evidence, source hashes and environment identity. New runs also preserve a checksummed source/configuration archive under `evidence/provenance/`. Confirmation uses fixed settings; neural refits use the rounded median selected best epoch.

```bash
python -m src reproduce --manifest output/runs/core-study/frozen.json --run-id replay-study
python -m src evaluate --manifest output/runs/replay-study/frozen.json --official-test
```

`run` includes frozen official evaluation by default, after development, confirmation and any requested isolated resource measurements. Use `--development-only` to stop after confirmation; a preprocessing-only diagnostic never opens official tests. Every final training fit must complete before official test data is opened. No configuration selection uses confirmation or official scores. `reproduce` skips selection and runs frozen confirmation; with `--run-id`, it uses an independent output namespace. The explicit `evaluate --official-test` action remains available to continue a completed development-only run, and rerunning it resumes committed work. The benchmark remains labelled as previously inspected. Source or environment changes invalidate an existing freeze and require an audited transition when deliberately updating orchestration.

## Reports and durable evidence

```bash
python -m src report --run complete-study
```

A new run uses one directory, `output/runs/<run-id>/`, containing its manifest, progress, frozen protocol, `evidence/`, `reports/` and ignored `temporary/` work. Reports are divided by scientific topic, each with one `README.md` and `tables/`, `figures/` and `metadata/` subfolders. Tables contain complete numerical CSV, shared across topics instead of copied into each topic or figure. Default figures combine allocations, paired differences, penalty sensitivity, learning curves and diagnostics into panels. Per-class and per-subject results remain in tables; confusion matrices remain in shared JSON. Figures default to high-resolution PNG; add `--pdf` for vector PDF figures. LaTeX tables are not generated. Each topic README collects captions, links to shared numerical sources and measurement boundaries. `overview/metadata/figures.json` identifies the exact source rows, checksums and transformations for each figure. These reading guides contain factual descriptions rather than authored scientific conclusions.

Detailed diagnostic figures are optional. For example:

```bash
python -m src report --run complete-study --details confusion --detail-dataset shd --detail-phase official --detail-role comparison:tuned:q8-i4-qrc
python -m src report --run complete-study --details regularization --detail-dataset dvs --detail-role comparison:tuned:q8-i4-qrc
```

`--details` accepts `confusion`, `regularization` and `fitting`; the dataset, phase and role filters affect only these extra figures. Canonical tables always retain all available results. Regeneration replaces the report, removing stale figures from earlier choices. Default penalty heatmaps show selected configurations; optional penalty curves expose all candidates for the requested roles. Missing fold or seed coverage remains visible and is never replaced with a partial mean.

`report` validates saved predictions, memberships, declared coverage and frozen settings. It never opens datasets, models or GPU backends. Render completion is distinct from scientific completion: a development report explicitly shows official evaluation as pending. Partial or missing comparisons cannot silently enter complete headline means. Official subgroup metrics are recomputed within each subgroup, not obtained by averaging group macro metrics.

Parameter declarations, candidate results and selection decisions are exported as tables. There is no `justify` command and no generated conclusions document. Authored reasoning, literature justification and conclusions remain in `keeping-track/`; reporting never edits them. Previously curated artifacts remain available at their existing locations, but old imported runs are not accepted as first-version workflow records.

## Artifact retention

- `output/datasets/`, `output/features/` and `output/locks/` are shared, ignored caches. Equivalent numerical workloads can reuse feature banks across named runs.
- `output/models/<pipeline-id>/` retains ignored inference-only packages for frozen confirmation and official-training fits, including adapters, readouts, model weights and reservoir definitions. Run evidence references them by identity and checksum.
- `output/runs/<run-id>/evidence/` retains compact predictions, shared memberships, candidate scores, scalar training histories, diagnostics, resource records and provenance. Evidence stays local and is Git-ignored at every depth under `output/`. Only the run’s `reports/` subtree is eligible for Git tracking; manifests, progress and logs also stay local. Sharing reports alone does not enable report regeneration: retain or separately transfer the local evidence and manifests.
- `output/runs/<run-id>/temporary/` holds candidate weights, optimizer state and resumable checkpoints. Successful completion verifies final packages, publishes the report and removes working artifacts whose consumers are complete. Failed or interrupted work is retained.

Cleanup records deliberate removals so resuming a completed run does not mistake pruned checkpoints for corruption. Shared caches and final models are never automatically removed. Regenerating reports requires only durable run evidence, not these heavy artifacts. Final inference packages can be replayed through the internal runtime interface; they are not optimizer checkpoints.

## Repository map

`configs/core.yaml` contains the main study settings, while `configs/supplementary.yaml` contains the optional LSTM and Transformer reference settings. These neural comparisons provide additional context for the central matched reservoir study. The configuration filenames do not change the `core` and `thesis` profile names or enable optional models automatically.

| Location | Responsibility |
|---|---|
| `src/main.py`, `src/workflow.py` | CLI, dependencies, scheduling, freeze and evaluation boundaries |
| `src/studies/` | Finite comparison definitions, selection, supporting studies and resources |
| `src/methodology/` | Parameter provenance and coverage validation |
| `src/data/` | Acquisition, event binning, memberships and ragged batches |
| `src/models/`, `src/features/`, `src/evaluation/` | Dynamics, neural models, fitted adapters, summaries, readouts and metrics |
| `src/reporting/`, `src/runtime/` | Derived reports, devices, caches, native libraries and provenance |
| `configs/`, `environment/` | Standalone scientific settings and environment/build records |
| `data/`, `output/datasets/`, `output/features/` | Ignored original datasets and reproducible intermediates |
| `output/runs/` | One namespace per execution: manifest, evidence, reports and temporary work |
| `output/models/` | Ignored retained final inference packages |
| `keeping-track/reports/` | Preserved earlier reports and audits; new run reports live inside each run |
| `keeping-track/reference/` | Immutable reference evidence, fitted artifacts and preserved source assets |
| `tests/` | Quick current-workflow checks and optional synthetic integration checks |
| `keeping-track/` | Writing material, methodology and decisions |

The ignored `keeping-track/retained-until-acceptance/` holds prior large artifacts and the remaining source records used by writing notes. It is a temporary removal boundary, not an input to the active workflow. No permanent historical archive is required. Tests remain until cleanup is accepted. The holding directory will remain until the thesis is ready.

## Validation and environment limits

[tests/README.md](tests/README.md) describes the focused suite of \(24\) quick and \(16\) integration checks. `--list` is read-only. Each validation action saves its result under `output/runs/checks-*/validation.json`.

```bash
python -m src validate --suite all --list
python -m src validate --suite quick
python -m src validate --suite all --gpu
```

The smoke workflow uses synthetic training and evaluation fixtures and is labelled accordingly. It never opens real official tests. Device checks require actual GPU visibility. [environment/README.md](environment/README.md) provides minimal pinned CPU and GPU recipes, both supporting core and optional models. [environment/BUILD.md](environment/BUILD.md) documents rebuilding the custom Aer wheel and native libraries under ignored `environment/local/`. Environment verification uses disposable installations and leaves the existing `qrc-spiking` environment unchanged.

Fresh CPU and GPU installations, a separate Aer source build, synthetic workflows and optional-model fits are checked during environment cleanup. The scope and results are recorded in [keeping-track/environment-cleanup/verification.json](keeping-track/environment-cleanup/verification.json). These installation checks do not constitute a complete scientific rerun or establish portability to other operating systems and GPUs.

The new workflow is independent of curated reference tables and preserved figures. See [keeping-track/output-generation-implementation.md](keeping-track/output-generation-implementation.md) for this implementation, its validation and remaining limits.
