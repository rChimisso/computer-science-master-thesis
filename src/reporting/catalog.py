import os
import shlex
from pathlib import Path
from urllib.parse import quote

from src.runtime.records import checksum, write

def performance_navigation(root: Path, directory: Path, captions: dict) -> list:
  """Link shared performance tables to topic-owned figures without duplicating images.

  :param root: Staged report directory.
  :param directory: Reading guide receiving relative links.
  :param captions: Validated figure inventory grouped by scientific topic.
  :return: Markdown navigation section, or an empty list when no views exist.
  """
  labels = {
    'comparisons': 'Matched QRC and CRC allocations',
    'practical': 'Large CRC, LSTM and Transformer',
    'mapping': 'Input mapping comparisons',
    'preprocessing': 'Preprocessing comparisons'
  }
  links = []
  for section, entries in sorted(captions.items()):
    for name in sorted(entries):
      if not (name.endswith('-allocations') or '-performance-' in name):
        continue
      images = [root / section / 'figures' / (name + '.' + extension) for extension in ('png', 'pdf')]
      formats = [f'[{path.suffix[1:].upper()}]({quote(os.path.relpath(path, directory))})' for path in images if path.is_file()]
      if not formats:
        continue
      phase = name.split('-performance-')[0].removesuffix('-allocations')
      guide = quote(os.path.relpath(root / section / 'README.md', directory))
      links.append(f"| {labels.get(section, section.title())} | {phase} | {' / '.join(formats)} | [Captions and sources]({guide}) |")
  if not links:
    return []
  return [
    '## Performance figures',
    '',
    'This section holds the shared numerical tables. Performance figures live with their scientific topics; use these direct links. Evaluation phases remain separate.',
    '',
    '| Comparison | Phase | Figure | Reading guide |',
    '|---|---|---|---|',
    *links,
    ''
  ]

def guides(root: Path, captions: dict, notes: dict) -> None:
  """Write one factual reading guide per topic after all report products exist.

  :param root: Staged report root.
  :param captions: Topic-to-figure caption records with canonical source references.
  :param notes: Additional topic-specific measurement boundaries.
  :raises ValueError: If a figure lacks its caption or source table.
  """
  topics = {path.name for path in root.iterdir() if path.is_dir()} | set(captions) | set(notes)
  for topic in sorted(topics):
    directory = root / topic
    entries = captions.get(topic, {})
    figures = {path.stem for path in (directory / 'figures').glob('*') if path.suffix in ('.png', '.pdf')}
    if figures != set(entries):
      raise ValueError('Figure/caption coverage differs in topic: ' + topic)
    lines = [
      '# ' + topic.replace('-', ' ').title(),
      '',
      'Classification scores use fractions. Figure captions identify the population and aggregation. Initialization variability and held-person variability are distinct; error bars are not significance tests.',
      ''
    ]
    shared = [root / 'performance/tables' / name for name in ('performance.csv', 'performance-cells.csv')]
    shared += [root / 'comparisons/tables' / name for name in ('paired-differences.csv', 'paired-cells.csv')]
    shared = [path for path in shared if path.is_file() and path.parent.parent != directory]
    if shared:
      lines += ['## Shared numerical tables', '', 'These tables retain all topics. Filter by dataset, phase and role to inspect this topic.', '']
      lines += [f'[{path.name}]({quote(os.path.relpath(path, directory))})' for path in shared]
      lines.append('')
    if topic == 'performance':
      lines += performance_navigation(root, directory, captions)
    if notes.get(topic):
      note = notes[topic].strip()
      lines += ['#' + note if note.startswith('# ') else '## Measurement notes\n\n' + note, '']
    if entries:
      lines += ['## Figures', '']
    for name, record in sorted(entries.items()):
      sources = record['sources']
      for source in sources:
        path = root / source['path']
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or checksum(path) != source['sha256']:
          raise ValueError('Figure canonical source is missing or changed: ' + str(path))
      images = [directory / 'figures' / (name + '.' + extension) for extension in ('png', 'pdf')]
      links = [f'[{path.suffix[1:].upper()}]({quote(path.relative_to(directory).as_posix())})' for path in images if path.is_file()]
      links += [f"[{Path(source['path']).name}]({quote(os.path.relpath(root / source['path'], directory))})" for source in sources]
      recipe = root / 'overview/metadata/figures.json'
      if not recipe.is_file():
        raise ValueError('Figure recipes are missing')
      links.append(f'[Plot recipe]({quote(os.path.relpath(recipe, directory))})')
      lines += [f'### `{name}`', '', ' | '.join(links), '', record['text'].strip(), '']
    tables = sorted((directory / 'tables').glob('*.csv'))
    if tables:
      lines += ['## Tables', '']
      lines += [f'- [{path.name}]({quote(path.relative_to(directory).as_posix())})' for path in tables]
      lines.append('')
    metadata = [path for path in sorted((directory / 'metadata').glob('*')) if path.is_file()]
    if metadata:
      lines += ['## Metadata', '']
      lines += [f'- [{path.name}]({quote(path.relative_to(directory).as_posix())})' for path in metadata]
      lines.append('')
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'README.md').write_text('\n'.join(lines))

def index(root: Path, source: Path, record: dict) -> None:
  """Index generated products by semantic section and file type.

  :param root: Report product root.
  :param source: Source evidence directory.
  :param record: Reporting status and interpretation limits.
  """
  command = ['python', '-m', 'src', 'report', '--run', str(source)]
  if 'pdf' in record.get('formats', {}).get('figures', []):
    command.append('--pdf')
  details = record.get('presentation', {}).get('details', {})
  if details.get('kinds'):
    command.extend(['--details', *details['kinds']])
    for key in ('dataset', 'role', 'phase'):
      if details.get(key):
        command.extend(['--detail-' + key, *details[key]])
  lines = [
    "# Scientific reports",
    "",
    f"Source evidence: `{source}`. Regenerate with `{shlex.join(command)}`.",
    "",
    "Tables and plot axes retain metric fractions. Initialization variability is distinct from held-person variability. Real official benchmarks were previously inspected; synthetic validation records are not scientific results.",
    ""
  ]
  if (root / 'performance/README.md').is_file():
    lines += ['[Performance figures and shared tables](performance/README.md)', '']
  products = {}
  for path in sorted(root.rglob("*")):
    if not path.is_file() or path == root / "index.md" or path.name == "products.json":
      continue
    relative = path.relative_to(root)
    products[str(relative)] = checksum(path)
  for semantic in sorted({name.split("/")[0] for name in products}):
    lines += [f"## {semantic.replace('-', ' ').title()}", ""]
    if semantic + "/README.md" in products:
      lines += [f"[Reading guide and figure captions]({semantic}/README.md)", ""]
    lines += [f"- [{name}]({name})" for name in products if name.split("/")[0] == semantic and name != semantic + "/README.md"]
    lines.append("")
  if record.get("limitations"):
    lines += ["## Interpretation and regeneration limits", ""]
    lines += ["- " + value for value in record["limitations"]]
    lines.append("")
  root.mkdir(parents=True, exist_ok=True)
  (root / "index.md").write_text("\n".join(lines))
  write(root / "products.json", {
    "source": str(source),
    "products": products,
    "formats": record.get("formats", {}),
    "dependencies": record.get("sources", {}),
    "coverage": record.get("coverage", []),
    "rendering_status": record.get("rendering_status")
  })
