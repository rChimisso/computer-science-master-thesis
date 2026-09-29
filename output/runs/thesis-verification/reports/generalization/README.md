# Generalization

Classification scores use fractions. Figure captions identify the population and aggregation. Initialization variability and held-person variability are distinct; error bars are not significance tests.

## Shared numerical tables

These tables retain all topics. Filter by dataset, phase and role to inspect this topic.

[performance.csv](../performance/tables/performance.csv)
[performance-cells.csv](../performance/tables/performance-cells.csv)
[paired-differences.csv](../comparisons/tables/paired-differences.csv)
[paired-cells.csv](../comparisons/tables/paired-cells.csv)

## Figures

### `dvs-fitting-gaps`

[PNG](figures/dvs-fitting-gaps.png) | [performance.csv](../performance/tables/performance.csv) | [Plot recipe](../overview/metadata/figures.json)

Each point is a complete configuration aggregate. Phases remain separate panels. Means average folds within seeds before averaging seeds; the diagonal denotes equal training and held scores. Points are descriptive, not independent replications.

### `shd-fitting-gaps`

[PNG](figures/shd-fitting-gaps.png) | [performance.csv](../performance/tables/performance.csv) | [Plot recipe](../overview/metadata/figures.json)

Each point is a complete configuration aggregate. Phases remain separate panels. Means average folds within seeds before averaging seeds; the diagonal denotes equal training and held scores. Points are descriptive, not independent replications.

### `shd-speaker-subsets-1`

[PNG](figures/shd-speaker-subsets-1.png) | [speaker-subsets.csv](tables/speaker-subsets.csv) | [Plot recipe](../overview/metadata/figures.json)

Post-hoc SHD official subsets. Colors show complete seed means. Seed variability and individual subgroup supports are retained in speaker-subsets.csv and speaker-cells.csv. Whole-test macro-F1 is independently recomputed, never averaged from subgroup macro-F1.

### `shd-speaker-subsets-2`

[PNG](figures/shd-speaker-subsets-2.png) | [speaker-subsets.csv](tables/speaker-subsets.csv) | [Plot recipe](../overview/metadata/figures.json)

Post-hoc SHD official subsets. Colors show complete seed means. Seed variability and individual subgroup supports are retained in speaker-subsets.csv and speaker-cells.csv. Whole-test macro-F1 is independently recomputed, never averaged from subgroup macro-F1.

### `shd-speaker-subsets-3`

[PNG](figures/shd-speaker-subsets-3.png) | [speaker-subsets.csv](tables/speaker-subsets.csv) | [Plot recipe](../overview/metadata/figures.json)

Post-hoc SHD official subsets. Colors show complete seed means. Seed variability and individual subgroup supports are retained in speaker-subsets.csv and speaker-cells.csv. Whole-test macro-F1 is independently recomputed, never averaged from subgroup macro-F1.

## Tables

- [per-class.csv](tables/per-class.csv)
- [per-subject.csv](tables/per-subject.csv)
- [speaker-cells.csv](tables/speaker-cells.csv)
- [speaker-subsets.csv](tables/speaker-subsets.csv)
- [training-minus-held.csv](tables/training-minus-held.csv)

## Metadata

- [confusions.json](metadata/confusions.json)
