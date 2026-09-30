from collections.abc import Mapping
from dataclasses import dataclass

@dataclass(frozen=True)
class FittingProtocol:
  """Validated fitting method shared by execution and scientific reporting.

  :ivar model: Registered model family.
  :ivar solver: Ridge solver or neural optimizer, with a single canonical name.
  """

  model: str
  solver: str

  @classmethod
  def from_spec(cls, spec: Mapping, recorded: Mapping | None = None) -> "FittingProtocol":
    """Resolve a declaration and reject contradictory persisted fitting metadata.

    Older records may omit the method; their explicit model declaration determines it.

    :param spec: Numerical model specification.
    :param recorded: Optional fitting result containing an explicitly recorded solver.
    :return: Validated model and fitting method.
    :raises ValueError: If the family, solver or recorded method is inconsistent.
    """
    model = spec.get("model")
    if model in ("lstm", "transformer"):
      solver = "adamax" if model == "lstm" else "adamw"
      if spec.get("solver", solver) != solver:
        raise ValueError("Neural fitting requires its declared optimizer")
    elif model in ("qrc", "crc", "projection"):
      solver = spec.get("solver", "cholesky")
      if solver not in ("cholesky", "lsqr"):
        raise ValueError("Unsupported ridge solver")
    else:
      raise ValueError("Unknown model family")
    if recorded is not None and recorded.get("solver", solver) != solver:
      raise ValueError("Recorded fitting method differs from the model declaration")
    return cls(model, solver)
