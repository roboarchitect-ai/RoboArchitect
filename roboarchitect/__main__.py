from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboarchitect.coding_model import CodingModel
from roboarchitect.evaluation import Evaluator, write_json
from roboarchitect.inputs import CODE_ROOT, Experiment, Paths, load_paths
from roboarchitect.model_servers import POLICY_REPOSITORIES, ModelServers
from roboarchitect.optimizer import HarnessOptimizer
from roboarchitect.processes import Interrupted, ProcessRegistry, interruption_handlers, process_registry
from roboarchitect.prompt import experiment_facts

from auxiliary.common import MODELS


def _gpu_ids(value: str) -> tuple[int, ...]:
    gpu_ids = tuple(int(item) for item in value.split(",") if item.strip())
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("--gpus takes distinct GPU indices such as 0,1")
    return gpu_ids


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def check_paths(experiment: Experiment, paths: Paths) -> None:
    pythons = ["harness"]
    repositories = []
    for setup in experiment.setups.values():
        pythons += [setup.environment.benchmark, setup.policy.server]
        pythons += [MODELS[name].python for name in setup.available_tools.values()]
        if setup.policy.server in POLICY_REPOSITORIES:
            repositories.append(POLICY_REPOSITORIES[setup.policy.server])
    problems = []
    for name in dict.fromkeys(pythons):
        try:
            paths.python_for(name)
        except FileNotFoundError as exc:
            problems.append(str(exc))
    for name in dict.fromkeys(repositories):
        try:
            paths.repository(name)
        except FileNotFoundError as exc:
            problems.append(str(exc))
    if problems:
        raise SystemExit("this experiment cannot start yet:\n" + "\n".join(problems))


def _prepare(folder: Path, experiment: Experiment, paths: Paths, args: argparse.Namespace) -> None:
    check_paths(experiment, paths)
    snapshot = experiment.snapshot(paths.file, args.gpus, args.episodes_per_gpu)
    if args.command == "evolve":
        snapshot.update(
            {
                "starting_harness": str(args.starting_harness.resolve()),
                "coding_model": args.coding_model,
                "iterations": args.iterations,
            }
        )
    target = folder / "experiment.json"
    if target.is_file():
        recorded = json.loads(target.read_text(encoding="utf-8"))
        if any(recorded.get(key) != snapshot[key] for key in ("setups", "metric", "split_file")):
            raise SystemExit(f"{folder} belongs to a different experiment; choose another --out")
    folder.mkdir(parents=True, exist_ok=True)
    write_json(target, snapshot)


def _evaluator(experiment: Experiment, episodes: str, servers: ModelServers, paths: Paths, args: argparse.Namespace) -> Evaluator:
    return Evaluator(
        experiment,
        episodes,
        servers=servers,
        paths=paths,
        episodes_per_gpu=args.episodes_per_gpu,
    )


def command_evolve(args: argparse.Namespace) -> int:
    experiment = Experiment.load(args.setup, args.split)
    paths = load_paths(args.paths)
    folder = args.out.resolve()
    _prepare(folder, experiment, paths, args)
    coding_model = CodingModel(args.coding_model)
    facts = experiment_facts(experiment.setups, sorted(experiment.setups)[0], experiment.metric, args.gpus)
    with ModelServers(experiment.setups, paths=paths, gpu_ids=args.gpus, log_root=folder / "server_logs") as servers:
        optimizer = HarnessOptimizer(
            experiment_dir=folder,
            evaluator=_evaluator(experiment, "training", servers, paths, args),
            held_out_evaluator=_evaluator(experiment, "held_out", servers, paths, args),
            coding_model=coding_model,
            facts=facts,
            starting_harness=args.starting_harness,
            iterations=args.iterations,
        )
        result = optimizer.optimize()
    json.dump(result, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    experiment = Experiment.load(args.setup, args.split)
    paths = load_paths(args.paths)
    folder = args.out.resolve()
    _prepare(folder, experiment, paths, args)
    harness = args.harness.resolve()
    with ModelServers(experiment.setups, paths=paths, gpu_ids=args.gpus, log_root=folder / "server_logs") as servers:
        target = folder / args.episodes
        (target / "harness").mkdir(parents=True, exist_ok=True)
        (target / "harness" / "harness.py").write_bytes(harness.read_bytes())
        result = _evaluator(experiment, args.episodes, servers, paths, args).evaluate(target / "harness", target / "evaluation")
        write_json(target / "result.json", result.to_mapping())
    summary = {"metric": result.score.metric, "score": result.score.value, "errored_episodes": result.errored_episodes}
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m roboarchitect")
    commands = parser.add_subparsers(dest="command", required=True)
    evolve = commands.add_parser("evolve", help="evolve a harness on the training episodes, then evaluate the starting and the final harness on the held-out episodes")
    evaluate = commands.add_parser("evaluate", help="evaluate one harness on the training or the held-out episodes")
    for command in (evolve, evaluate):
        command.add_argument("--setup", type=Path, required=True, help="a setup file, such as setups/vlabench/xvla_pt.json")
        command.add_argument("--split", type=Path, required=True, help="a split file, such as splits/scenes/vlabench.json")
    evolve.add_argument("--starting-harness", type=Path, default=CODE_ROOT / "roboarchitect" / "starting_harness.py", help="the harness.py to start from; by default, one that passes the policy's actions through unchanged")
    evolve.add_argument("--coding-model", default="claude-opus-5", help="the model Claude Code uses to revise the harness")
    evolve.add_argument("--iterations", type=_positive, default=5, help="the number of revisions")
    evaluate.add_argument("--harness", type=Path, required=True, help="the harness.py to evaluate")
    evaluate.add_argument("--episodes", choices=("training", "held_out"), default="held_out", help="the episodes to evaluate on")
    for command in (evolve, evaluate):
        command.add_argument("--gpus", type=_gpu_ids, required=True, help="the GPUs to use, such as 0,1")
        command.add_argument("--episodes-per-gpu", type=_positive, default=4, help="the number of episodes that run at the same time on each GPU")
        command.add_argument("--paths", type=Path, default=CODE_ROOT / "paths.json", help="the paths file, paths.json in the repository by default")
        command.add_argument("--out", type=Path, required=True, help="the folder to write the experiment to")
    evolve.set_defaults(handler=command_evolve)
    evaluate.set_defaults(handler=command_evaluate)
    args = parser.parse_args(argv)
    registry = ProcessRegistry()
    with process_registry(registry), interruption_handlers(registry):
        try:
            try:
                return args.handler(args)
            finally:
                registry.terminate_all()
        except Interrupted as exc:
            print(f"interrupted: {exc.signal_name}", file=sys.stderr)
            return exc.exit_status


if __name__ == "__main__":
    raise SystemExit(main())
