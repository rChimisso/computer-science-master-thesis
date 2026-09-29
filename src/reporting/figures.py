import csv
import math
from pathlib import Path

import numpy as np

from src.reporting.products import topic
from src.runtime.records import checksum, read, write

def colors() -> dict:
  """Define stable model-family colors.

  :return: Family color mapping.
  """
  return {'qrc': '#7B3294', 'crc': '#008B8B', 'lstm': '#357A38', 'transformer': '#C66B13', 'projection': '#777777'}

def family(role: str) -> str:
  """Resolve the family for display only, never for scientific pairing.

  :param role: Declared comparison role.
  :return: Plot color family.
  """
  return next((name for name in colors() if name in role), 'projection')

def number(row: dict, key: str) -> float:
  """Read a numeric CSV cell without converting missing values to zero.

  :param row: Canonical CSV record.
  :param key: Numeric column.
  :return: Numeric value or NaN.
  """
  return float(row.get(key) or 'nan')

def spread(row: dict, key: str) -> float | None:
  """Omit uncertainty bars where replication does not establish an SD.

  :param row: Aggregate record.
  :param key: Uncertainty column.
  :return: Finite SD or None.
  """
  value = number(row, key)
  return value if np.isfinite(value) else None

def label(role: str) -> str:
  """Shorten displayed labels while full identities remain in source tables.

  :param role: Full comparison identifier.
  :return: Readable role label.
  """
  return role.removeprefix('comparison:').replace('practical:', '').replace(':', ' / ')

def grid(count: int, height: float = 4, columns: int = 2):
  """Create bounded-size panels and hide unused axes.

  :param count: Required panels.
  :param height: Height per panel in inches.
  :param columns: Maximum columns.
  :return: Matplotlib figure and flat active axes.
  """
  import matplotlib.pyplot as plt
  columns = min(columns, count)
  figure, axes = plt.subplots(math.ceil(count / columns), columns, figsize=(7 * columns, height * math.ceil(count / columns)), squeeze=False)
  active = list(axes.flat)
  for axis in active[count:]:
    axis.set_visible(False)
  return figure, active[:count]

class Figures:
  """Render views referencing canonical records instead of duplicating source CSVs.

  :ivar root: Staged reporting destination.
  :ivar pdf: Whether vector exports are enabled.
  :ivar tables: Lazily loaded canonical rows with internal source positions.
  :ivar hashes: Checksums of canonical inputs.
  :ivar captions: Topic guides' factual captions and source references.
  :ivar recipes: Exact row references and transformation descriptions for every figure.
  """

  def __init__(self, root: Path, pdf: bool = False):
    """Initialize a figure writer without loading data or numerical backends.

    :param root: Staged report directory.
    :param pdf: Include PDF alongside PNG.
    """
    self.root, self.pdf = root, pdf
    self.tables, self.hashes, self.captions, self.recipes = {}, {}, {}, {}

  def rows(self, name: str) -> list:
    """Read canonical CSV or JSON records with stable zero-based row identities.

    :param name: Report-relative source path.
    :return: Rows augmented only in memory with source positions.
    """
    if name not in self.tables:
      path = self.root / name
      if not path.exists():
        return []
      if path.suffix == '.csv':
        with path.open() as stream:
          values = list(csv.DictReader(stream))
      else:
        values = read(path)
      self.hashes[name] = checksum(path)
      self.tables[name] = [row | {'_table': name, '_row': index} for index, row in enumerate(values)]
    return self.tables[name]

  def publish(self, section: str, name: str, figure, rows: list, caption: str, transformation: dict) -> None:
    """Save an image and an auditable recipe pointing to canonical input rows.

    :param section: Scientific topic.
    :param name: Unique figure basename.
    :param figure: Complete Matplotlib figure.
    :param rows: Every canonical row consumed by the plotted calculation.
    :param caption: Factual scope and interpretation boundaries.
    :param transformation: Filters, grouping and calculation definition.
    """
    import matplotlib.pyplot as plt
    key = section + '/' + name
    if key in self.recipes:
      raise ValueError('Duplicate figure identity: ' + key)
    selected = {}
    for row in rows:
      selected.setdefault(row['_table'], set()).add(row['_row'])
    sources = [{'path': path, 'sha256': self.hashes[path], 'rows': sorted(indices)} for path, indices in sorted(selected.items())]
    if not sources:
      raise ValueError('Figure has no canonical input records: ' + key)
    directory = self.root / section / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for extension in (('png', 'pdf') if self.pdf else ('png',)):
      figure.savefig(directory / f'{name}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(figure)
    recipe = {'caption': caption, 'sources': sources, 'transformation': transformation, 'row_numbering': 'zero-based data rows, excluding CSV header'}
    self.recipes[key] = recipe
    self.captions.setdefault(section, {})[name] = {'text': caption, 'sources': sources}

  def finish(self) -> dict:
    """Persist shared plot recipes after all views have rendered.

    :return: Guide captions indexed by topic and figure.
    """
    write(self.root / 'overview/metadata/figures.json', self.recipes)
    return self.captions

def render(root: Path, evidence: dict, pdf: bool = False, details: dict | None = None) -> dict:
  """Render comprehensive overviews plus explicitly requested detailed diagnostics.

  :param root: Staged canonical tables and metadata.
  :param evidence: Validated evidence, used only for declared allocation ordering.
  :param pdf: Include vector PDFs.
  :param details: Optional diagnostic kinds and exact identity filters.
  :return: Captions and canonical source references for topic guides.
  """
  import matplotlib
  matplotlib.use('Agg')
  writer = Figures(root, pdf)
  performance(writer, evidence['manifest']['config']['allocations'])
  effects(writer)
  regularization(writer)
  distributions(writer)
  generalization(writer)
  learning(writer, [str(value) for value in evidence['manifest']['config']['supporting']['learning_sizes']] + ['full'])
  fitting(writer)
  resources(writer)
  detailed(writer, details or {})
  return writer.finish()

def performance(writer: Figures, allocations: list) -> None:
  """Show all allocation protocols and other studies in separate phase panels.

  :param writer: Canonical figure writer.
  :param allocations: Declared total/input-qubit pairs.
  """
  summaries = writer.rows('performance/tables/performance.csv')
  cells = writer.rows('performance/tables/performance-cells.csv')
  held = [row for row in summaries if row['population'] == 'held']
  mapping_performance(writer, [row for row in held if topic(row['role'], row['phase']) == 'mapping'])
  for phase in sorted({row['phase'] for row in held if row['role'].startswith('comparison:') and ':n=' not in row['role']}):
    selected = [row for row in held if row['phase'] == phase and row['role'].startswith('comparison:')]
    datasets = sorted({row['dataset'] for row in selected})
    modes = ['fixed', 'tuned', 'fixed12']
    figure, axes = grid(len(datasets) * len(modes), columns=len(datasets))
    used = []
    for axis, (mode, dataset) in zip(axes, ((mode, dataset) for mode in modes for dataset in datasets)):
      for model in ('qrc', 'crc'):
        for position, (qubits, inputs) in enumerate(allocations):
          role = f'comparison:{mode}:q{qubits}-i{inputs}-{model}'
          rows = [row for row in selected if row['dataset'] == dataset and row['role'] == role]
          if not rows:
            continue
          row = rows[0]
          used.append(row)
          raw = [cell for cell in cells if cell['dataset'] == dataset and cell['phase'] == phase and cell['role'] == role and cell['population'] == 'held']
          used.extend(raw)
          x = position + (-0.1 if model == 'qrc' else 0.1)
          axis.errorbar(x, number(row, 'macro_f1'), yerr=spread(row, 'macro_f1_seed_sd'), fmt='o', color=colors()[model])
          for seed in sorted({cell['seed'] for cell in raw}):
            values = [number(cell, 'macro_f1') for cell in raw if cell['seed'] == seed]
            axis.plot(x, np.mean(values), '.', color=colors()[model], alpha=0.5)
      for model in ('qrc', 'crc'):
        axis.plot([], [], 'o', color=colors()[model], label=model.upper())
      axis.set_xticks(range(len(allocations)), [f'{inputs}/{total-inputs}' for total, inputs in allocations])
      axis.set(xlabel='Input/memory allocation', ylabel='Macro-F1', title=f'{dataset.upper()} / {phase} / {mode}', ylim=(0, 1))
      axis.legend(fontsize=8)
    if used:
      writer.publish('comparisons', f'{phase}-allocations', figure, used,
        'Each panel preserves dataset, evaluation phase and fixed/tuned/fixed-observation protocol. Small points retain initialization seeds after averaging development folds; bars show sample seed '
        'SD. Allocations are discrete comparisons, not a monotonic size axis.',
        {'operation': 'allocation points', 'phase': phase, 'allocations': allocations, 'seed_aggregation': 'mean over folds, then sample SD over seeds'})
    else:
      import matplotlib.pyplot as plt
      plt.close(figure)
  for section, phase in sorted({(topic(row['role'], row['phase']), row['phase']) for row in held}):
    if section in ('comparisons', 'observations', 'recurrence', 'training-size', 'mapping'):
      continue
    rows = [row for row in held if topic(row['role'], row['phase']) == section and row['phase'] == phase]
    datasets = sorted({row['dataset'] for row in rows})
    roles = sorted({row['role'] for row in rows})
    for page, start in enumerate(range(0, len(roles), 20), 1):
      names = roles[start:start + 20]
      figure, axes = grid(len(datasets), max(3, len(names) * 0.3), columns=len(datasets))
      used = []
      for axis, dataset in zip(axes, datasets):
        for position, role in enumerate(names):
          values = [row for row in rows if row['dataset'] == dataset and row['role'] == role]
          if values:
            row = values[0]
            used.append(row)
            axis.errorbar(number(row, 'macro_f1'), position, xerr=spread(row, 'macro_f1_seed_sd'), fmt='o', color=colors()[family(role)])
        axis.set_yticks(range(len(names)), [label(role) for role in names], fontsize=7)
        axis.set(xlabel='Macro-F1', xlim=(0, 1), title=f'{dataset.upper()} / {phase}')
      writer.publish(section, f'{phase}-performance-{page}', figure, used,
        'Complete aggregate performance, with sample SD across initialization seeds where replication exists. Development folds are averaged within seeds; screening and robustness protocols remain separate. Blank positions are unavailable comparisons.',
        {'operation': 'aggregate dot plot', 'phase': phase, 'roles': names, 'metric': 'macro_f1', 'error': 'macro_f1_seed_sd'})

def mapping_parts(role: str) -> tuple:
  """Decode declared mapping roles for display without defining scientific pairings.

  :param role: Mapping comparison identifier from canonical records.
  :return: Input width, adapter, model family and optional projection seed.
  """
  parts = role.split(':')
  if parts[0] == 'mapping' and parts[1] == 'screen':
    return int(parts[2][1:]), parts[3], 'projection', None
  if parts[0] in ('mapping-paired', 'mapping-robustness'):
    return int(parts[1][1:]), parts[2], parts[3], int(parts[4][1:]) if len(parts) == 5 else None
  raise ValueError('Unrecognized mapping display identity: ' + role)

def mapping_performance(writer: Figures, rows: list) -> None:
  """Group mapping performance by dataset and input width with explicit seed semantics.

  :param writer: Canonical figure writer.
  :param rows: Complete held-population mapping aggregates.
  """
  adapter_names = {'legacy': 'Anchor', 'orthogonal': 'Orthogonal', 'pooling': 'Pooling', 'pca': 'PCA'}
  model_names = {'qrc': 'QRC', 'crc': 'CRC', 'projection': 'Projection-only ridge'}
  descriptions = {
    'mapping-screen': 'Projection-only mapping screen',
    'mapping-development': 'Selected mapping and anchor',
    'mapping-robustness': 'Sensitivity to projection seed'
  }
  for phase in sorted({row['phase'] for row in rows}):
    subset = [row for row in rows if row['phase'] == phase]
    panels = sorted({(row['dataset'], mapping_parts(row['role'])[0], mapping_parts(row['role'])[1] if phase == 'mapping-robustness' else '') for row in subset}, key=lambda key: (key[1], key[2], key[0]))
    figure, axes = grid(len(panels), 3.5)
    for axis, (dataset, width, adapter) in zip(axes, panels):
      selected = [row for row in subset if row['dataset'] == dataset and mapping_parts(row['role'])[0] == width and (not adapter or mapping_parts(row['role'])[1] == adapter)]
      identities = [mapping_parts(row['role']) for row in selected]
      if len(set(identities)) != len(identities):
        raise ValueError('Duplicate mapping performance identity')
      coordinates = sorted({key[3] for key in identities}) if adapter else [mode for mode in adapter_names if any(key[1] == mode for key in identities)]
      models = [model for model in model_names if any(key[2] == model for key in identities)]
      for model_index, model in enumerate(models):
        values = [row for row in selected if mapping_parts(row['role'])[2] == model]
        positions = [coordinates.index(mapping_parts(row['role'])[3 if adapter else 1]) + (model_index - (len(models) - 1) / 2) * 0.16 for row in values]
        axis.errorbar(
          positions,
          [number(row, 'macro_f1') for row in values],
          yerr=None if phase != 'mapping-development' else [number(row, 'macro_f1_seed_sd') for row in values],
          fmt={'qrc': 'o', 'crc': 's', 'projection': '^'}[model],
          color=colors()[model],
          label=model_names[model],
          capsize=3
        )
      axis.set_xticks(range(len(coordinates)), [str(value) if adapter else adapter_names[value] for value in coordinates])
      axis.set(xlabel='Projection seed' if adapter else 'Input mapping', ylabel='Development macro-F1', ylim=(0, 1), xlim=(-0.5, len(coordinates) - 0.5))
      axis.set_title(f'{dataset.upper()} / {width} inputs' + (' / ' + adapter_names[adapter] if adapter else ''))
      axis.legend(fontsize=8)
    figure.suptitle(descriptions[phase])
    common = 'Panels show only comparisons actually run for each dataset and input width. Anchor denotes the legacy random adapter; the other mappings use train-fitted coordinate standardization. '
    if phase == 'mapping-screen':
      caption = common + 'Projection-only ridge, screening seed $42$. Points are means across the four development folds. No initialization-seed uncertainty is estimated from this single-seed screen.'
    elif phase == 'mapping-development':
      caption = common + 'QRC, CRC and projection-only ridge use the advanced mapping and anchor. Points are means over development folds and initialization seeds; bars show sample seed SD after averaging folds.'
    else:
      caption = common + 'The horizontal axis varies the projection seed, with reservoir initialization fixed at $42$. Points average the four development folds. These are individual projection-seed results, not dynamics-seed error bars.'
    writer.publish('mapping', f'{phase}-performance-1', figure, subset, caption,
      {'operation': 'mapping performance by dataset and input width', 'phase': phase, 'panels': panels, 'metric': 'macro_f1', 'seed_axis': 'projection' if phase == 'mapping-robustness' else None})

def mapping_effects(writer: Figures, rows: list, phase: str) -> None:
  """Separate mapping and model contrasts using only validated dataset-specific pairs.

  :param writer: Canonical figure writer.
  :param rows: Validated paired aggregates for this mapping phase.
  :param phase: Evaluation protocol identity.
  """
  titles = {'mapping-QRC-minus-CRC': 'QRC minus CRC', 'mapping-minus-anchor': 'Selected mapping minus anchor'}
  adapters = {'legacy': 'Anchor', 'orthogonal': 'Orthogonal', 'pooling': 'Pooling', 'pca': 'PCA'}
  names = {'qrc': 'QRC', 'crc': 'CRC', 'projection': 'Projection-only'}
  panels = sorted({(row['contrast'], row['dataset']) for row in rows})
  height = max(3, max(sum((row['contrast'], row['dataset']) == key for row in rows) for key in panels) * 0.45)
  figure, axes = grid(len(panels), height)
  for axis, (contrast, dataset) in zip(axes, panels):
    selected = sorted([row for row in rows if row['contrast'] == contrast and row['dataset'] == dataset], key=lambda row: mapping_parts(row['left'])[:3])
    labels = []
    for position, row in enumerate(selected):
      width, adapter, model, _ = mapping_parts(row['left'])
      labels.append(f'{width} inputs / {adapters[adapter]}' + (' / ' + names[model] if contrast == 'mapping-minus-anchor' else ''))
      axis.errorbar(number(row, 'macro_f1'), position, xerr=spread(row, 'macro_f1_seed_sd'), fmt='o', color=colors()[model], capsize=3)
    limit = max(abs(number(row, 'macro_f1')) + (spread(row, 'macro_f1_seed_sd') or 0) for row in rows if row['contrast'] == contrast) * 1.15
    axis.axvline(0, color='grey', linewidth=0.8)
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    axis.invert_yaxis()
    axis.set(xlabel='Paired macro-F1 difference', xlim=(-max(limit, 0.01), max(limit, 0.01)), title=f'{dataset.upper()} / {titles[contrast]}')
  figure.suptitle('Mapping development contrasts')
  writer.publish('mapping', f'{phase}-effects-1', figure, rows,
    'Panels separate QRC-minus-CRC from selected-mapping-minus-anchor effects and contain only the pairs run for each dataset. Positive values favor the first term in each title. '
    'Means and sample seed SD follow fold averaging within initialization seeds. Axes match across datasets for the same contrast. Anchor is the legacy adapter; no significance claim.',
    {'operation': 'dataset-specific mapping paired differences', 'phase': phase, 'panels': panels, 'metric': 'macro_f1', 'error': 'macro_f1_seed_sd'})

def effects(writer: Figures) -> None:
  """Combine declared paired effects into readable dataset panels without new pairings.

  :param writer: Canonical figure writer.
  """
  rows = writer.rows('comparisons/tables/paired-differences.csv')
  for section, phase in sorted({(topic(row['left'], row['phase']), row['phase']) for row in rows}):
    selected = [row for row in rows if topic(row['left'], row['phase']) == section and row['phase'] == phase]
    if section == 'mapping':
      mapping_effects(writer, selected, phase)
      continue
    identities = sorted({(row['contrast'], row['left'], row['right']) for row in selected})
    datasets = sorted({row['dataset'] for row in selected})
    for page, start in enumerate(range(0, len(identities), 16), 1):
      keys = identities[start:start + 16]
      figure, axes = grid(len(datasets), max(3, len(keys) * 0.45), columns=len(datasets))
      used = []
      for axis, dataset in zip(axes, datasets):
        for position, key in enumerate(keys):
          values = [row for row in selected if row['dataset'] == dataset and (row['contrast'], row['left'], row['right']) == key]
          if values:
            row = values[0]
            used.append(row)
            axis.errorbar(number(row, 'macro_f1'), position, xerr=spread(row, 'macro_f1_seed_sd'), fmt='o', color=colors()[family(row['left'])])
        axis.axvline(0, color='grey', linewidth=0.8)
        axis.set_yticks(range(len(keys)), [label(left) + '\nminus ' + label(right) for _, left, right in keys], fontsize=6)
        axis.set(xlabel='Paired macro-F1 difference', title=f'{dataset.upper()} / {phase}')
      writer.publish(section, f'{phase}-effects-{page}', figure, used,
        'Each row is an explicitly validated left-minus-right contrast. Bars show sample seed SD after averaging development folds within seeds. Datasets and phases are separate; no significance '
        'claim. Individual paired cells and fold consistency remain in the shared comparison tables.',
        {'operation': 'paired aggregate dot plot', 'phase': phase, 'contrasts': keys, 'metric': 'macro_f1', 'error': 'macro_f1_seed_sd'})

def penalty_grid(rows: list) -> tuple:
  """Compute complete screening-fold means and regret without averaging duplicates.

  :param rows: One dataset/phase collection of seed-42 candidate cells.
  :return: Configuration identities, ordered penalties, absolute means and regret.
  """
  keys = sorted({(row['role'], row['configuration']) for row in rows})
  penalties = sorted({number(row, 'lambda') for row in rows if np.isfinite(number(row, 'lambda'))})
  absolute = np.full((len(keys), len(penalties)), np.nan)
  grouped = {}
  for row in rows:
    if row['status'] != 'completed' or not np.isfinite(number(row, 'lambda')):
      continue
    key = (row['role'], row['configuration'], number(row, 'lambda'))
    folds = grouped.setdefault(key, {})
    fold = str(row['fold'])
    value = number(row, 'macro_f1')
    if fold in folds and folds[fold] != value:
      raise ValueError('Conflicting duplicate candidate score')
    folds[fold] = value
  for i, key in enumerate(keys):
    for j, penalty in enumerate(penalties):
      folds = grouped.get((*key, penalty), {})
      if set(folds) == {'0', '1', '2', '3'}:
        absolute[i, j] = np.mean(list(folds.values()))
  regret = np.full_like(absolute, np.nan)
  for i, values in enumerate(absolute):
    if np.isfinite(values).any():
      regret[i] = np.nanmax(values) - values
  return keys, penalties, absolute, regret

def regularization(writer: Figures) -> None:
  """Show selected configurations' penalty sensitivity; retain all searches in tables.

  :param writer: Canonical figure writer.
  """
  candidates = writer.rows('methodology/tables/candidate-cells.csv')
  decisions = writer.rows('methodology/tables/selection-decisions.csv')
  selected = {(row['dataset'], row['phase'], row['role'], row['configuration']) for row in decisions}
  rows = [row for row in candidates if str(row['seed']) == '42' and (row['dataset'], row['phase'], row['role'], row['configuration']) in selected and row.get('lambda')]
  for dataset, phase in sorted({(row['dataset'], row['phase']) for row in rows}):
    subset = [row for row in rows if row['dataset'] == dataset and row['phase'] == phase]
    keys, penalties, absolute, regret = penalty_grid(subset)
    for page, start in enumerate(range(0, len(keys), 28), 1):
      chosen = keys[start:start + 28]
      figure, axes = grid(2, max(4, len(chosen) * 0.23))
      for axis, values, title, upper in zip(axes, (absolute, regret), ('Absolute macro-F1', 'Loss from best tested penalty'), (1, 0.2)):
        image = axis.imshow(np.ma.masked_invalid(values[start:start + 28]), aspect='auto', vmin=0, vmax=upper, cmap='viridis')
        axis.set_yticks(range(len(chosen)), [label(role) + ' / ' + identity[:6] for role, identity in chosen], fontsize=6)
        axis.set_xticks(range(len(penalties)), [f'{value:g}' for value in penalties], rotation=45)
        axis.set(xlabel='Sample-normalized ridge penalty', title=title)
        figure.colorbar(image, ax=axis, extend='max' if title.startswith('Loss') else 'neither')
      figure.suptitle(f'{dataset.upper()} / {phase} / selected configurations')
      used = [row for row in subset if (row['role'], row['configuration']) in chosen]
      used += [row for row in decisions if row['dataset'] == dataset and row['phase'] == phase and (row['role'], row['configuration']) in chosen]
      writer.publish('methodology', f'{dataset}-{phase}-penalties-{page}', figure, used,
        'Selected configurations only, screening seed $42$. Each cell requires all four development folds. Left: absolute mean macro-F1. Right: loss from the best complete tested penalty for that '
        'configuration; colors saturate above $0.2$. Missing coverage is blank. All candidates, including unsuccessful and incomplete records, remain in canonical tables.',
        {'operation': 'complete four-fold penalty mean and row-wise regret', 'keys': chosen, 'seed': 42, 'phase': phase, 'dataset': dataset, 'regret_color_cap': 0.2})

def distributions(writer: Figures) -> None:
  """Compare dataset partitions in shared duration/event-count panels.

  :param writer: Canonical figure writer.
  """
  rows = writer.rows('data/tables/samples.csv')
  settings = writer.rows('data/tables/dataset-settings.csv')
  datasets = sorted({row['dataset'] for row in rows})
  if not datasets:
    return
  figure, axes = grid(len(datasets) * 2)
  for index, dataset in enumerate(datasets):
    selected = [row for row in rows if row['dataset'] == dataset]
    width = number(next(row for row in settings if row['dataset'] == dataset), 'bin_us') / 1e6
    for offset, metric in enumerate(('steps', 'events')):
      axis = axes[index * 2 + offset]
      multiplier = width if metric == 'steps' else 1
      bins = np.histogram_bin_edges([number(row, metric) * multiplier for row in selected], bins=20)
      for partition in sorted({row['partition'] for row in selected}):
        values = [number(row, metric) * multiplier for row in selected if row['partition'] == partition]
        axis.hist(values, bins=bins, alpha=0.45, label=partition)
      axis.set(xlabel='Valid-bin duration (s)' if metric == 'steps' else 'Event count', ylabel='Examples', title=dataset.upper())
      axis.legend(fontsize=8)
  writer.publish('data', 'partition-distributions', figure, rows + settings,
    'Dataset-specific panels use common bins across their development, confirmation and official partitions. Duration is valid-bin count times bin width, not exact event span. Counts describe examples, not probability densities.',
    {'operation': 'partition histograms', 'bins': 20, 'duration': 'steps * bin_us / 1000000', 'bin_policy': 'shared per dataset and metric'})

def generalization(writer: Figures) -> None:
  """Summarize fitting gaps and post-hoc speaker populations without class plot floods.

  :param writer: Canonical figure writer.
  """
  rows = writer.rows('performance/tables/performance.csv')
  for dataset in sorted({row['dataset'] for row in rows}):
    phases = sorted({row['phase'] for row in rows if row['dataset'] == dataset})
    figure, axes = grid(len(phases))
    used = []
    for axis, phase in zip(axes, phases):
      subset = [row for row in rows if row['dataset'] == dataset and row['phase'] == phase]
      for held in [row for row in subset if row['population'] == 'held']:
        train = next((row for row in subset if row['role'] == held['role'] and row['population'] == 'train'), None)
        if train:
          axis.scatter(number(train, 'macro_f1'), number(held, 'macro_f1'), color=colors()[family(held['role'])], alpha=0.65, s=16)
          used.extend((train, held))
      axis.plot([0, 1], [0, 1], color='grey', linestyle='--')
      axis.set(xlabel='Training macro-F1', ylabel='Held macro-F1', title=phase, xlim=(0, 1), ylim=(0, 1))
      for model in sorted({family(row['role']) for row in subset}):
        axis.scatter([], [], color=colors()[model], label=model)
      axis.legend(fontsize=7)
    figure.suptitle(dataset.upper())
    writer.publish('generalization', f'{dataset}-fitting-gaps', figure, used,
      'Each point is a complete configuration aggregate. Phases remain separate panels. Means average folds within seeds before averaging seeds; the diagonal denotes equal training and held scores. Points are descriptive, not independent replications.',
      {'operation': 'join train and held aggregates by dataset, phase and role', 'dataset': dataset, 'metric': 'macro_f1'})
  speakers = writer.rows('generalization/tables/speaker-subsets.csv')
  roles = sorted({row['role'] for row in speakers})
  for page, start in enumerate(range(0, len(roles), 28), 1):
    names = roles[start:start + 28]
    groups = ['seen', 'unseen', 'whole']
    figure, axes = grid(1, max(4, len(names) * 0.23))
    values = np.full((len(names), 3), np.nan)
    used = []
    for i, role in enumerate(names):
      for j, group in enumerate(groups):
        match = [row for row in speakers if row['role'] == role and row['group'] == group and row['population'] == 'held']
        if match:
          values[i, j] = number(match[0], 'macro_f1')
          used.append(match[0])
    axis = axes[0]
    image = axis.imshow(values, vmin=0, vmax=1, aspect='auto', cmap='viridis')
    axis.set_xticks(range(3), groups)
    axis.set_yticks(range(len(names)), [label(role) for role in names], fontsize=7)
    axis.set_title('SHD official / post-hoc speaker subsets')
    figure.colorbar(image, ax=axis, label='Mean macro-F1')
    writer.publish('generalization', f'shd-speaker-subsets-{page}', figure, used,
      'Post-hoc SHD official subsets. Colors show complete seed means. Seed variability and individual subgroup supports are retained in speaker-subsets.csv and speaker-cells.csv. Whole-test '
      'macro-F1 is independently recomputed, never averaged from subgroup macro-F1.',
      {'operation': 'role by speaker-subset mean matrix', 'roles': names, 'groups': groups, 'metric': 'macro_f1'})

def learning(writer: Figures, sizes: list) -> None:
  """Combine matched and rich-input learning curves while retaining incomplete gaps.

  :param writer: Canonical figure writer.
  :param sizes: Declared subset sizes including full fold training.
  """
  rows = [row for row in writer.rows('performance/tables/performance.csv') if row['phase'] == 'learning-development' and row['population'] == 'held']
  if not rows:
    return
  figure, axes = grid(2)
  used = []
  for axis, practical in zip(axes, (False, True)):
    selected = [row for row in rows if row['role'].startswith('practical:') == practical]
    bases = sorted({row['role'].split(':n=')[0] for row in selected})
    for index, role in enumerate(bases):
      values = [next((row for row in selected if row['role'] == role + ':n=' + size), None) for size in sizes]
      axis.errorbar(
        range(len(sizes)),
        [number(row, 'macro_f1') if row else np.nan for row in values],
        yerr=[number(row, 'macro_f1_seed_sd') if row else np.nan for row in values],
        marker=('o', 's', '^', 'D')[index % 4],
        linestyle=('-', '--')[index % 2],
        color=colors()[family(role)],
        label=label(role)
      )
      used.extend(row for row in values if row)
    axis.set_xticks(range(len(sizes)), sizes)
    axis.set(xlabel='Fold-training examples', ylabel='Development macro-F1', title='Practical references' if practical else 'Reservoirs and input controls')
    if bases:
      axis.legend(fontsize=7)
  writer.publish('training-size', 'learning-curves', figure, used,
    'SHD nested-subset learning curves with fixed selected hyperparameters. Means and sample seed SD use complete folds and seeds only. Missing aggregates are gaps. Full denotes each fold training partition, not one common sample count.',
    {'operation': 'ordered training-size aggregate curves', 'sizes': sizes, 'metric': 'macro_f1', 'error': 'macro_f1_seed_sd'})

def fitting(writer: Figures, selection: dict | None = None) -> None:
  """Combine actual stopped histories in role panels without epoch-coverage averaging.

  :param writer: Canonical figure writer.
  :param selection: Optional exact dataset, phase and role filters for detailed output.
  """
  rows = writer.rows('practical/tables/epoch-history.csv')
  if selection:
    rows = [row for row in rows if matches(row, selection)]
  for dataset, phase in sorted({(row['dataset'], row['phase']) for row in rows}):
    selected = [row for row in rows if row['dataset'] == dataset and row['phase'] == phase]
    roles = sorted({row['role'] for row in selected})
    for page, start in enumerate(range(0, len(roles), 4), 1):
      names = roles[start:start + 4]
      figure, axes = grid(len(names) * 2, 3)
      used = []
      for index, role in enumerate(names):
        values = [row for row in selected if row['role'] == role]
        used.extend(values)
        for seed, fold, population in sorted({(row['seed'], row['fold'], row['population']) for row in values}):
          history = sorted([row for row in values if (row['seed'], row['fold'], row['population']) == (seed, fold, population)], key=lambda row: int(row['epoch']))
          for axis, metric in zip(axes[index * 2:index * 2 + 2], ('loss', 'macro_f1')):
            axis.plot([int(row['epoch']) for row in history], [number(row, metric) for row in history], color='#357A38' if population == 'train' else '#C66B13', alpha=0.25)
            axis.set(xlabel='Epoch', ylabel=metric, title=label(role))
      figure.suptitle(f'{dataset.upper()} / {phase} | green: train; orange: validation')
      section = 'preprocessing' if phase == 'preprocessing-diagnostic' else 'practical'
      writer.publish(section, f'{dataset}-{phase}-fitting-{page}' + ('-detail' if selection else ''), figure, used,
        'Each line is one actual fold/seed history, ending at its recorded stopping epoch. Green is training and orange validation. Epoch coverage changes are not averaged away. Configuration, '
        'selected epoch and stopping reason remain in the canonical history table.',
        {'operation': 'individual history lines', 'dataset': dataset, 'phase': phase, 'roles': names, 'group_by': ['role', 'seed', 'fold', 'population']})

def resources(writer: Figures) -> None:
  """Join practical test performance with measured latency and storage in dataset panels.

  :param writer: Canonical figure writer.
  """
  rows = [row for row in writer.rows('resources/tables/practical-costs.csv') if row['test_status'] == 'completed']
  datasets = sorted({row['dataset'] for row in rows})
  if not datasets:
    return
  figure, axes = grid(len(datasets) * 2)
  for index, dataset in enumerate(datasets):
    selected = [row for row in rows if row['dataset'] == dataset]
    for axis, metric, scale, title in zip(axes[index * 2:index * 2 + 2], ('inference_seconds_per_example', 'storage_mib'), (1000, 1), ('Batched inference (ms/example)', 'Inference package (MiB)')):
      labelled = set()
      for row in selected:
        model = {'crc': 'Large CRC', 'lstm': 'LSTM', 'transformer': 'Transformer'}[row['model']]
        description = model + ' (' + row['device'] + ')'
        axis.scatter(number(row, metric) * scale, number(row, 'macro_f1'), color=colors()[row['model']], label=None if description in labelled else description)
        labelled.add(description)
      axis.set(xlabel=title, ylabel='Official-test macro-F1', title=dataset.upper())
      axis.legend(fontsize=8)
  writer.publish('resources', 'practical-cost-performance', figure, rows,
    'Each point retains a seed. Inference repetitions use warmed development-confirmation workloads and confirmation fits; performance uses official-training refits with the same settings. '
    'Legends identify devices. These are implementation/device measurements, not hardware-neutral costs. Storage is inference-package size; no simulator timing is presented as quantum-hardware cost.',
    {'operation': 'cost/performance scatter', 'latency_scale': 1000, 'latency_unit': 'ms/example', 'storage_unit': 'MiB'})

def matches(row: dict, selection: dict) -> bool:
  """Apply optional exact detail filters without filtering canonical evidence.

  :param row: Diagnostic record.
  :param selection: Dataset, role and phase lists, any of which may be absent.
  :return: Whether this diagnostic should be drawn.
  """
  return all(not selection.get(key) or row.get(key) in selection[key] for key in ('dataset', 'role', 'phase'))

def detailed(writer: Figures, selection: dict) -> None:
  """Render explicitly requested confusion matrices and full-candidate penalty views.

  :param writer: Canonical figure writer.
  :param selection: Diagnostic kinds and exact optional identity filters.
  """
  kinds = selection.get('kinds', [])
  if set(kinds) - {'confusion', 'regularization', 'fitting'}:
    raise ValueError('Unknown detailed diagnostic kind')
  rows = writer.rows('generalization/metadata/confusions.json') if 'confusion' in kinds else []
  rows = [row for row in rows if row['population'] == 'held' and matches(row, selection)]
  if 'confusion' in kinds and not rows:
    raise ValueError('No confusion matrices match the requested detail filters')
  for dataset, phase, role in sorted({(row['dataset'], row['phase'], row['role']) for row in rows}):
    selected = [row for row in rows if (row['dataset'], row['phase'], row['role']) == (dataset, phase, role)]
    normalized = []
    for row in selected:
      matrix = np.asarray(row['matrix'], dtype=float)
      normalized.append(np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) > 0))
    figure, axes = grid(1, 5)
    image = axes[0].imshow(np.mean(normalized, axis=0), vmin=0, vmax=1, cmap='Blues')
    axes[0].set(xlabel='Predicted class', ylabel='True class', title=f'{dataset.upper()} / {phase}\n{label(role)}')
    figure.colorbar(image, ax=axes[0], label='Mean row-normalized proportion')
    writer.publish('generalization', f'{dataset}-{phase}-{role.replace(":", "-")}-confusion-detail', figure, selected,
      'Held population. Each fold/seed count matrix is row-normalized before averaging; absent true classes contribute zero rows. These are proportions across repeated evaluations, not '
      'unique-sample counts. Exact individual count matrices remain in the shared JSON.',
      {'operation': 'mean of row-normalized matrices', 'dataset': dataset, 'phase': phase, 'role': role, 'zero_support': 'zero row'})
  candidates = writer.rows('methodology/tables/candidate-cells.csv') if 'regularization' in kinds else []
  candidates = [row for row in candidates if str(row['seed']) == '42' and row.get('lambda') and matches(row, selection)]
  if 'regularization' in kinds and not candidates:
    raise ValueError('No ridge candidates match the requested detail filters')
  for dataset, phase, role in sorted({(row['dataset'], row['phase'], row['role']) for row in candidates}):
    selected = [row for row in candidates if (row['dataset'], row['phase'], row['role']) == (dataset, phase, role)]
    keys, penalties, absolute, _ = penalty_grid(selected)
    figure, axes = grid(1)
    for index, (_, configuration) in enumerate(keys):
      axes[0].plot(penalties, absolute[index], marker='.', label=configuration[:8])
    axes[0].set(xscale='log', xlabel='Sample-normalized penalty', ylabel='Macro-F1', title=f'{dataset.upper()} / {phase}\n{label(role)}')
    axes[0].legend(fontsize=6, ncol=3)
    writer.publish('methodology', f'{dataset}-{phase}-{role.replace(":", "-")}-regularization-detail', figure, selected,
      'Screening seed $42$ only. Every point requires all four development folds; incomplete comparisons remain gaps. Configuration hashes resolve to candidate settings in the canonical candidate-scores table.',
      {'operation': 'complete four-fold penalty curves', 'dataset': dataset, 'phase': phase, 'role': role, 'seed': 42})
  if 'fitting' in kinds:
    if not any(matches(row, selection) for row in writer.rows('practical/tables/epoch-history.csv')):
      raise ValueError('No fitting histories match the requested detail filters')
    fitting(writer, selection)
