# Dataset feasibility decision

## Decision

SHD is the primary dataset for the matched CRC-QRC study. DVS Gesture is deferred until the complete SHD experiment is reproducible, and SSC is excluded from the initial scope. The frozen SHD raw representation is Tonic's structured event array
`(t, x, p)`, with timestamps in microseconds, cochlear-channel index `x`, and synthetic polarity $p=1$.

The deciding constraint is not whether DVS Gesture is scientifically useful - it is the stronger hardware-native event dataset - but whether both reservoirs can receive a defensible common representation without turning spatial feature engineering into
the main experiment. SHD's ordered $700$-channel cochlea permits simple adjacent-channel pooling. DVS Gesture begins with $128 \times 128 \times 2$ inputs and requires aggressive spatial pooling before a small QRC can encode a time step.

## Method

Measurements were generated with Tonic `1.6.0` and the configured seed $42$ by:

```bash
python scripts/run_dataset_feasibility.py --datasets shd dvs
```

The probe selected $100$ evenly spaced training examples from each Tonic dataset. Reported event-count and duration distributions are estimates over those $100$ examples; class balance and sample counts use the complete training split.
The preprocessing benchmark excludes disk I/O and measures count binning into $10\,\mathrm{ms}$ steps. SHD is pooled into $32$ adjacent cochlear groups.
DVS Gesture is pooled into an $8 \times 8$ grid with separate polarity planes, producing $128$ features.
This probe is only a feasibility measurement and does not freeze the subsequent preprocessing choices.

The approximate QRC cost is a transparent relative proxy, not a runtime claim:

$$
\text{mean time bins} \times \text{circuit depth} \times \text{qubits} \times 2^{\text{qubits}}
$$

It assumes six statevector-simulated qubits and circuit depth two. It omits backend overhead, input-encoding gates, measurement cost, and caching, so it must not be interpreted as quantum-hardware cost.

## Measured comparison

| Measure                         |      SHD training split |            DVS Gesture training split |
| ------------------------------- | ----------------------: | ------------------------------------: |
| Tonic samples                   |                $8\,156$ |                              $1\,077$ |
| Classes                         |                    $20$ |                                  $11$ |
| Raw input dimensions            |                   $700$ | $32\,768$ ($128 \times 128 \times 2$) |
| Event fields                    |             `(t, x, p)` |                        `(x, y, p, t)` |
| Median events/sample            |              $7\,312.5$ |                          $298\,400.5$ |
| $95$th-percentile events/sample |             $12\,568.6$ |                          $754\,848.7$ |
| Median duration                 |    $720.2\,\mathrm{ms}$ |               $6\,312.8\,\mathrm{ms}$ |
| $95$th-percentile duration      | $1\,078.4\,\mathrm{ms}$ |               $8\,874.3\,\mathrm{ms}$ |
| Training class-count range      |          $393$ to $421$ |                          $97$ to $98$ |
| Local Tonic footprint           |    $516.6\,\mathrm{MB}$ |                   $14.9\,\mathrm{GB}$ |
| Probe output features           |                    $32$ |                                 $128$ |
| Binning time/sample             |      $0.4\,\mathrm{ms}$ |                   $19.0\,\mathrm{ms}$ |
| QRC proxy work/sample           |               $56\,447$ |                            $489\,467$ |
| QRC proxy, full training split  |         $460.4$ million |                       $527.2$ million |

The SHD local footprint includes both official train/test HDF5 files and their download archives. The DVS footprint includes its training archive and extracted NumPy files. Timings depend on the current machine and are included for scale, not as portable
performance guarantees. Exact machine-readable summaries and the generated figures are written below `results/dataset_feasibility/`.

Ten samples from each candidate were visually inspected. SHD inspection uses spike rasters; DVS inspection uses spatial event projections colored by polarity. The additional SHD diagnostic figure covers spike counts across all $700$ channels, the full
training-duration distribution, event counts over normalized time, and speaker metadata. The plots show nontrivial temporal structure and no obvious schema or timestamp failures.

## SHD split policy

The official test split remains untouched during preprocessing and model selection. It contains $2\,264$ samples. Speakers $4$ and $5$ occur only in that split, accounting for $1\,840$ test samples and providing a strong unseen-speaker component.

Within the official training split, speakers $0$ and $1$ are the fixed validation speakers:

| Partition  | Speaker IDs                              |  Samples |
| ---------- | ---------------------------------------- | -------: |
| Training   | $2$, $3$, $6$, $7$, $8$, $9$, $10$, $11$ | $6\,340$ |
| Validation | $0$, $1$                                 | $1\,816$ |
| Test       | official split; never used for tuning    | $2\,264$ |

Every one of the $20$ classes is present in both the derived training and validation partitions. Validation has $84$ to $96$ samples per class and training has $304$ to $328$. The split is reconstructed deterministically from `extra/speaker` in the
official HDF5 file by `src.data.splits.speaker_disjoint_indices`; no random sample split is permitted.

## Why DVS Gesture is deferred

DVS Gesture contains real DVS128 camera events and remains the preferred extension for testing whether SHD conclusions transfer to physical event hardware. The original study reports $1\,342$ recordings from $29$ subjects and $11$ classes under three
lighting conditions. Tonic's pre-segmented copy exposes $1\,341$ clips ($1\,077$ train and $264$ test) across its standard subject-disjoint split, a small preprocessing discrepancy that must be retained and documented if the extension is activated.

Its disadvantage for this thesis is dimensionality and event volume. Even the feasibility probe reduces $32\,768$ sensor/polarity inputs to $128$ features, four times the provisional SHD input, before reservoir simulation. Selecting spatial pooling,
polarity handling, and perhaps regions of interest would introduce dataset-specific assumptions absent from the primary comparison. DVS Gesture is therefore deferred, not rejected.

## Frozen scope and provenance

The primary dataset, split rule, and raw event format are now frozen before model development. The preprocessing study may compare only the predefined temporal bins and adjacent-channel counts for SHD, and must save identical tensors for CRC and QRC. Changes to the
dataset or split require a new dated decision record.

Sources: the [SHD dataset page](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/), the [DVS Gesture paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Amir_A_Low_Power_CVPR_2017_paper.pdf), the
[Tonic dataset documentation](https://tonic.readthedocs.io/en/latest/datasets.html), and the [maintained DVS Gesture archive](https://zenodo.org/records/8060604).
