# Resources

Classification scores use fractions. Figure captions identify the population and aggregation. Initialization variability and held-person variability are distinct; error bars are not significance tests.

## Shared numerical tables

These tables retain all topics. Filter by dataset, phase and role to inspect this topic.

[performance.csv](../performance/tables/performance.csv)
[performance-cells.csv](../performance/tables/performance-cells.csv)
[paired-differences.csv](../comparisons/tables/paired-differences.csv)
[paired-cells.csv](../comparisons/tables/paired-cells.csv)

## Practical cost measurement boundaries

Recorded fit time includes extraction or cache loading, fitting and evaluation. Rows marked cached features are not cold training-from-scratch costs. Fitting timings retain their original execution conditions; only inference repetitions were isolated.

Inference uses warmed, length-stratified development-confirmation workloads and confirmation-fit packages. Joined official-test metrics use separately refitted models with identical settings and seeds. Device is explicit: CPU and CUDA timings are implementation/device measurements, not hardware-neutral architecture costs.

Storage measures the inference package. RAM is whole-process inference peak RSS including runtime and metadata. VRAM is PyTorch allocated memory, not total GPU use or peak training memory. No whole-system energy comparison is available. Exhaustive boundaries and measurements are retained in practical-measurements.csv.

## Figures

### `practical-cost-performance`

[PNG](figures/practical-cost-performance.png) | [practical-costs.csv](tables/practical-costs.csv) | [Plot recipe](../overview/metadata/figures.json)

Each point retains a seed. Inference repetitions use warmed development-confirmation workloads and confirmation fits; performance uses official-training refits with the same settings. Legends identify devices. These are implementation/device measurements, not hardware-neutral costs. Storage is inference-package size; no simulator timing is presented as quantum-hardware cost.

## Tables

- [practical-costs.csv](tables/practical-costs.csv)
- [practical-measurements.csv](tables/practical-measurements.csv)
- [reservoir-structure.csv](tables/reservoir-structure.csv)
