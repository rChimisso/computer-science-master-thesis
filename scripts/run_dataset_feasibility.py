import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""Absolute path to the repository root."""

sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.feasibility import analyze_dataset, load_dataset, plot_event_samples, plot_shd_diagnostics, save_summary
from src.utils.config import load_config, repository_path
from src.utils.reproducibility import set_global_seed

def parse_arguments() -> argparse.Namespace:
  """Parse command-line options.

  :return: Parsed command-line namespace.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", default="configs/default.yaml")
  parser.add_argument("--datasets", nargs="+", choices=("shd", "dvs"), default=("shd", "dvs"))
  parser.add_argument("--sample-limit", type=int)
  return parser.parse_args()

def main() -> None:
  """Measure requested datasets and save their summaries and figures.

  :return: ``None``.
  """
  arguments = parse_arguments()
  config = load_config(arguments.config)
  set_global_seed(int(config["project"]["seed"]))
  settings = config["feasibility"]
  sample_limit = arguments.sample_limit or int(settings["sample_limit"])
  raw_path = repository_path(config["paths"]["raw_data"])
  output_path = repository_path(config["paths"]["results"]) / "dataset_feasibility"
  output_path.mkdir(parents=True, exist_ok=True)

  for name in arguments.datasets:
    dataset = load_dataset(name, raw_path, train=True)
    summary = analyze_dataset(
      dataset,
      name,
      "train",
      sample_limit=sample_limit,
      temporal_bin_us=int(settings["temporal_bin_us"]),
      shd_pooled_channels=int(settings["shd_pooled_channels"]),
      dvs_spatial_bins=int(settings["dvs_spatial_bins"]),
      qrc_qubits=int(settings["qrc_qubits"]),
      qrc_circuit_depth=int(settings["qrc_circuit_depth"])
    )
    stem = "shd" if name == "shd" else "dvs_gesture"
    save_summary(summary, output_path / f"{stem}_train_summary.json")
    plot_event_samples(dataset, name, output_path / f"{stem}_ten_samples.png", count=int(settings["visual_sample_count"]))
    if name == "shd":
      plot_shd_diagnostics(dataset, output_path / "shd_diagnostics.png", sample_limit=sample_limit)
    print(f"Saved {stem} summary and figures to {output_path}")

if __name__ == "__main__":
  main()
