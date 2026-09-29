# Training Size

Classification scores use fractions. Figure captions identify the population and aggregation. Initialization variability and held-person variability are distinct; error bars are not significance tests.

## Shared numerical tables

These tables retain all topics. Filter by dataset, phase and role to inspect this topic.

[performance.csv](../performance/tables/performance.csv)
[performance-cells.csv](../performance/tables/performance-cells.csv)
[paired-differences.csv](../comparisons/tables/paired-differences.csv)
[paired-cells.csv](../comparisons/tables/paired-cells.csv)

## Figures

### `learning-curves`

[PNG](figures/learning-curves.png) | [performance.csv](../performance/tables/performance.csv) | [Plot recipe](../overview/metadata/figures.json)

SHD nested-subset learning curves with fixed selected hyperparameters. Means and sample seed SD use complete folds and seeds only. Missing aggregates are gaps. Full denotes each fold training partition, not one common sample count.

### `learning-development-effects-1`

[PNG](figures/learning-development-effects-1.png) | [paired-differences.csv](../comparisons/tables/paired-differences.csv) | [Plot recipe](../overview/metadata/figures.json)

Each row is an explicitly validated left-minus-right contrast. Bars show sample seed SD after averaging development folds within seeds. Datasets and phases are separate; no significance claim. Individual paired cells and fold consistency remain in the shared comparison tables.
