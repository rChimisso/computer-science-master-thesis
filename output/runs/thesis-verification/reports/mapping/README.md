# Mapping

Classification scores use fractions. Figure captions identify the population and aggregation. Initialization variability and held-person variability are distinct; error bars are not significance tests.

## Shared numerical tables

These tables retain all topics. Filter by dataset, phase and role to inspect this topic.

[performance.csv](../performance/tables/performance.csv)
[performance-cells.csv](../performance/tables/performance-cells.csv)
[paired-differences.csv](../comparisons/tables/paired-differences.csv)
[paired-cells.csv](../comparisons/tables/paired-cells.csv)

## Figures

### `mapping-development-effects-1`

[PNG](figures/mapping-development-effects-1.png) | [paired-differences.csv](../comparisons/tables/paired-differences.csv) | [Plot recipe](../overview/metadata/figures.json)

Panels separate QRC-minus-CRC from selected-mapping-minus-anchor effects and contain only the pairs run for each dataset. Positive values favor the first term in each title. Means and sample seed SD follow fold averaging within initialization seeds. Axes match across datasets for the same contrast. Anchor is the legacy adapter; no significance claim.

### `mapping-development-performance-1`

[PNG](figures/mapping-development-performance-1.png) | [performance.csv](../performance/tables/performance.csv) | [Plot recipe](../overview/metadata/figures.json)

Panels show only comparisons actually run for each dataset and input width. Anchor denotes the legacy random adapter; the other mappings use train-fitted coordinate standardization. QRC, CRC and projection-only ridge use the advanced mapping and anchor. Points are means over development folds and initialization seeds; bars show sample seed SD after averaging folds.

### `mapping-robustness-performance-1`

[PNG](figures/mapping-robustness-performance-1.png) | [performance.csv](../performance/tables/performance.csv) | [Plot recipe](../overview/metadata/figures.json)

Panels show only comparisons actually run for each dataset and input width. Anchor denotes the legacy random adapter; the other mappings use train-fitted coordinate standardization. The horizontal axis varies the projection seed, with reservoir initialization fixed at $42$. Points average the four development folds. These are individual projection-seed results, not dynamics-seed error bars.

### `mapping-screen-performance-1`

[PNG](figures/mapping-screen-performance-1.png) | [performance.csv](../performance/tables/performance.csv) | [Plot recipe](../overview/metadata/figures.json)

Panels show only comparisons actually run for each dataset and input width. Anchor denotes the legacy random adapter; the other mappings use train-fitted coordinate standardization. Projection-only ridge, screening seed $42$. Points are means across the four development folds. No initialization-seed uncertainty is estimated from this single-seed screen.

## Tables

- [conditioning.csv](tables/conditioning.csv)
