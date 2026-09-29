import argparse
import datetime
import json
from pathlib import Path

def parser() -> argparse.ArgumentParser:
  """Define one public entry point with explicit scientific access boundaries.

  :return: Argument parser.
  """
  result = argparse.ArgumentParser(description="Reproducible matched reservoir thesis workflow")
  commands = result.add_subparsers(dest="action", required=True)
  for action in ("plan", "run"):
    command = commands.add_parser(action)
    command.add_argument("--config", default="configs/core.yaml")
    command.add_argument("--profile", choices=("core", "thesis"), default="thesis")
    command.add_argument("--models", nargs="*", choices=("large-crc", "lstm", "transformer"), default=[])
    command.add_argument("--dataset", nargs="+", choices=("shd", "dvs"))
    command.add_argument("--study", nargs="+")
    command.add_argument("--seed", nargs="+", type=int, choices=(42, 44, 46))
    command.add_argument("--resources", action="store_true")
    command.add_argument("--development-only", action="store_true", help="Stop after confirmation; full runs otherwise include frozen official evaluation")
    if action == "run":
      command.add_argument("--deadline")
      command.add_argument("--run-id")
      command.add_argument("--resume", action="store_true")
    else:
      command.add_argument("--output", help="Write the resolved queue as JSON")
  for action in ("reproduce", "evaluate"):
    command = commands.add_parser(action)
    command.add_argument("--manifest", required=True)
    command.add_argument("--deadline")
    if action == "reproduce":
      command.add_argument("--run-id")
    if action == "evaluate":
      command.add_argument("--official-test", action="store_true", required=True)
  command = commands.add_parser("report")
  command.add_argument("--run", required=True)
  command.add_argument("--pdf", action="store_true", help="Also export vector PDF figures; defaults to PNG and CSV")
  command.add_argument("--details", nargs="+", choices=("confusion", "regularization", "fitting"), default=[], help="Add detailed diagnostic figures; default reports retain all numerical data")
  command.add_argument("--detail-dataset", nargs="+", choices=("shd", "dvs"), help="Filter detailed figures only")
  command.add_argument("--detail-role", nargs="+", help="Exact comparison roles for detailed figures only")
  command.add_argument("--detail-phase", nargs="+", help="Exact evaluation phases for detailed figures only")
  for action in ("status", "pause"):
    command = commands.add_parser(action)
    command.add_argument("--run", required=True)
    if action == "status":
      command.add_argument("--watch", action="store_true")
  command = commands.add_parser("validate")
  command.add_argument("--suite", choices=("quick", "extended", "all"), default="quick")
  command.add_argument("--list", action="store_true", dest="list_tests", help="List the selected checks without running tests or opening datasets")
  command.add_argument("--gpu", action="store_true")
  command.add_argument("--smoke", action="store_true")
  for action in ("plan", "run", "reproduce", "evaluate"):
    command = commands.choices[action]
    command.add_argument("--cpu-workers", type=int)
    command.add_argument("--cpu-threads", type=int)
    command.add_argument("--gpu-workers", type=int)
    if action != "plan":
      command.add_argument("--detach", action="store_true")
  for command in commands.choices.values():
    command.add_argument("--progress", choices=("auto", "plain", "off"), default=None, help="Task progress on stderr: terminal bars, plain events or disabled")
  return result

def main(argv: list | None = None) -> None:
  """Dispatch requested workflows, reporting and monitoring actions.

  :param argv: Optional argument vector for tests.
  """
  args = parser().parse_args(argv)
  if args.progress is not None:
    from src.runtime.progress import configure
    configure(args.progress)
  deadline = float("inf")
  if getattr(args, "deadline", None):
    value = datetime.datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if value.tzinfo is None:
      raise ValueError("Deadline must contain an explicit timezone")
    deadline = value.timestamp()
  if args.action in ("run", "plan"):
    from src.workflow import plan, resolve
    request = resolve(args)
    from src.runtime.resources import policy
    limits = policy(args.cpu_workers, args.cpu_threads, args.gpu_workers)
    if args.action == "plan":
      result = plan(request)
      result["resources"] = limits
      from src.runtime.resources import request as reserve
      families = [{"model": "crc"}, {"model": "qrc", "qubits": 6}, {"model": "qrc", "qubits": 8}]
      families += [{"model": model} for model in request['models'] if model in ('lstm', 'transformer')]
      reservations = [{"spec": spec, **reserve('neural' if spec['model'] in ('lstm', 'transformer') else 'feature', spec, request['config'])} for spec in families]
      result['task_resource_floors'] = reservations
      result['admission_errors'] = [f"{row['spec']} requires {row['cpu_threads']} CPU threads, exceeding the total budget" for row in reservations if row['cpu_threads'] > limits['cpu_threads']]
      if args.output:
        from src.runtime.records import write
        write(args.output, result)
      for dataset in result["datasets"].values():
        dataset.pop("queue", None)
    else:
      from src.runtime.execution import launch
      name = args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime("study-%Y%m%dT%H%M%S%fZ")
      if name in (".", "..") or Path(name).name != name:
        raise ValueError("Run identifier must be a single directory name")
      result = launch('run', request, Path(request['config']['paths']['runs']) / name, limits, deadline, args.resume, args.detach)
  elif args.action in ("evaluate", "reproduce"):
    from src.runtime.execution import launch
    from src.runtime.resources import policy
    from src.runtime.records import frozen_record
    from src.runtime.reproduction import prepare
    path = Path(args.manifest) if args.action == 'evaluate' else prepare(args.manifest, args.run_id)
    frozen_record(path)
    result = launch(args.action, {'manifest': str(path)}, path.parent, policy(args.cpu_workers, args.cpu_threads, args.gpu_workers), deadline, True, args.detach)
  elif args.action in ('status', 'pause'):
    from src.runtime.monitor import pause, status, watch
    root = Path(args.run) if Path(args.run).exists() else Path('output/runs') / args.run
    result = pause(root) if args.action == 'pause' else watch(root) if args.watch else status(root)
    if args.action == 'status' and not args.watch:
      from src.runtime.monitor import render
      print(render(result))
      return
  elif args.action == "report":
    root = Path(args.run)
    if not root.exists():
      root = Path("output/runs") / args.run
    from src.reporting.study import report
    if not args.details and any((args.detail_dataset, args.detail_role, args.detail_phase)):
      raise ValueError('Detail filters require --details')
    result = report(root, pdf=args.pdf, details={"kinds": args.details, "dataset": args.detail_dataset, "role": args.detail_role, "phase": args.detail_phase})
  else:
    from src.runtime.validation import validate
    result = validate(args.smoke, args.gpu, args.suite, args.list_tests)
  print(json.dumps(result, indent=2, allow_nan=False))
