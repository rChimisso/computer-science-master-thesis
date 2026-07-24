import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def append_json_result(path: str | Path, parameters: dict[str, Any], metrics: dict[str, Any]) -> None:
  """Append one timestamped parameter and metric record as JSON Lines.

  :param path: Destination JSON Lines path.
  :param parameters: Parameters that produced the metrics.
  :param metrics: Measured experiment outputs.
  :return: ``None``.
  """
  result_path = Path(path)
  result_path.parent.mkdir(parents=True, exist_ok=True)
  record = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "parameters": parameters,
    "metrics": metrics
  }
  with result_path.open("a", encoding="utf-8") as result_file:
    result_file.write(json.dumps(record, sort_keys=True) + "\n")
