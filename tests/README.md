# Verification

Use the `qrc-spiking` environment and run commands from the repository root. The suite protects the scientific calculations and the authored workflow with synthetic inputs. It requires no downloads, saved dataset fixtures, campaign results or writing documents.

| Scope | Purpose | Test methods |
|---|---|---:|
| `quick/` | Scientific arithmetic, data isolation, selection, evidence validity and bounded execution checks | \(24\) |
| `extended/` | Complete synthetic workflows, device integration, interruption recovery and offline reports | \(16\) |

Related invalid-input cases share named subtests. A failed subtest identifies its scenario; the method count does not represent the number of assertions.

## Commands

Inspect the suite without importing tests or accessing devices:

```bash
python -m src validate --suite all --list
```

Run quick checks, the integration suite, or everything with actual GPU agreement:

```bash
python -m src validate
python -m src validate --suite extended
python -m src validate --suite all --gpu
```

Without `--gpu`, child processes hide CUDA devices. With it, device preflight requires a working GPU backend and fails if unavailable. It checks Aer, eager CRC and CUDA graph agreement in a short-lived process, releasing its CUDA context before the suites start. Integration tests additionally exercise full-length CPU/GPU sequences and optional neural fits. Validation installs no packages.

`--smoke` adds the small complete synthetic workflow to quick checks. The extended suite already runs serial and parallel versions, so an additional smoke run is unnecessary. Prefer the public commands to plain unittest discovery, which includes all integration checks.

## Organization

| File | Protected behavior |
|---|---|
| `quick/test_science.py` | Event binning, memberships, train-only fitting, matched inputs, analytical quantum states, padding, metrics and selection |
| `quick/test_models.py` | Temporal classwise loss and locked dropout |
| `quick/test_readout.py` | Independent binary/multiclass ridge inference |
| `quick/test_cache.py` | Reuse, corruption detection and numerical invalidation |
| `quick/test_protocol.py` | Optional-model independence, parameter provenance, frozen source protection and official access gates |
| `quick/test_execution.py` | Resource admission, task ownership, failure propagation and frozen resumption |
| `quick/test_reporting.py` | Complete comparisons, paired memberships, compact prediction integrity and resource evidence |
| `extended/test_workflow.py` | Serial/parallel agreement, optional inference pipelines and neural RNG isolation |
| `extended/test_reservoirs.py` | Complete sequences, independent CRC trajectories, GPU graph replay and quantum interruption |
| `extended/test_training.py` | Neural checkpoint recovery and calibrated batch sizes |
| `extended/test_lifecycle.py` | Detachment, cooperative pause and worker cleanup after coordinator death |
| `extended/test_protocol.py` | Synthetic official evaluation, continuation and frozen reproduction |
| `extended/test_reporting.py` | Offline report regeneration, safe cleanup and mapping semantics |
| `extended/test_supporting.py` | Supporting controls, isolated resource measurements and declared seed budgets |
| `support.py` | Shared synthetic data, compact evidence and frozen-run setup; no test methods |
| `extended/lifecycle_fixture.py` | Subprocess helper for real coordinator lifecycle checks; no saved data |

All evaluation inputs are synthetic. No real official-test data are loaded. Device tolerances remain unchanged, including the quantum absolute tolerance of \(10^{-10}\).

The suite deliberately omits detailed terminal/ETA behavior, scheduling fairness, plot layout, PDF/filter permutations and environment-build edge cases. It also omits dedicated invalid run-name and archive-traversal checks. Production safeguards remain in place; removing their tests reduces coverage. Mapping figure tests retain numerical source and seed semantics, rather than exact labels or layout.

## Records and runtime

Each validation action writes `output/runs/checks-*/validation.json`, with per-suite process durations and exit codes. Failures stop subsequent suites. Intentional worker-failure tests may print a traceback while correctly passing their failure-propagation assertions.

Integration cases own their temporary artifacts. The end-to-end test removes its successful synthetic run directories after inspecting their results. A failed smoke run can remain under `output/runs/validation-*` for diagnosis; these are never thesis results. The standalone `--smoke` action retains its own record.

Observed laptop timings during cleanup:

| Scope | Process duration |
|---|---:|
| Quick suite | Approximately \(55\)-\(57\) seconds |
| Integration suite with GPU checks enabled | Approximately \(490\) seconds, or \(8.2\) minutes |

Allow about \(10\) minutes for the full GPU-enabled command, including device preflight. These are indicative correctness-check timings, not model resource benchmarks; competing work and hardware affect them. Planning ranges from `--list` are conservative allowances, not fresh measurements. Test counts do not predict runtime: full-sequence and subprocess checks remain intentionally substantive.
