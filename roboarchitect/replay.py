from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from roboarchitect.episode import SimulatorProcess
from roboarchitect.inputs import Setup, load_paths, load_split
from roboarchitect.protocol import ActionChunk, StrictSchemaError
from roboarchitect.records import EpisodeKey


def find_experiment_folder(episode_dir: Path) -> Path:
    for parent in [episode_dir, *episode_dir.parents]:
        if (parent / "experiment.json").is_file():
            return parent
    raise FileNotFoundError(f"no experiment folder above {episode_dir}: expected a folder holding experiment.json")


def read_actions(trace: Path) -> list[list[list[float]]]:
    actions = []
    with trace.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index == 0:
                continue
            row = json.loads(line)
            if row.get("action") is not None:
                actions.append(row["action"]["values"])
    return actions


def replay(
    episode_dir: Path,
    output_dir: Path,
    *,
    render_gpu: int,
    action_source: Path | None,
    max_steps: int,
    save_depth: bool,
) -> dict[str, Any]:
    episode_dir = Path(episode_dir).resolve()
    manifest = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    key = EpisodeKey.from_mapping(manifest["key"])
    experiment_dir = find_experiment_folder(episode_dir)
    snapshot = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    setups = {suite: Setup.from_mapping(value) for suite, value in snapshot["setups"].items()}
    suite = next(iter(setups)) if len(setups) == 1 else key.task_id.partition("::")[0]
    setup = setups[suite]
    _, scenes = load_split(snapshot["split_file"])
    paths = load_paths(snapshot["paths_file"])
    values = read_actions(episode_dir / "trace.jsonl" if action_source is None else action_source)
    if not values:
        raise StrictSchemaError("no actions found; expected 'action' entries in the trace")
    if max_steps > 0:
        values = values[:max_steps]
    output_dir = Path(output_dir).resolve()
    runtime_dir = output_dir / "simulator"
    if runtime_dir.exists():
        raise FileExistsError(f"the simulator folder of an earlier replay is still there, remove it first: {runtime_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    simulator = SimulatorProcess(setup, key, scenes.get(suite), paths=paths, gpu_id=render_gpu, runtime_dir=runtime_dir)
    from PIL import Image

    success = False
    executed = 0
    completed = False
    try:
        simulator.start()
        full_horizon = simulator.judged_at_horizon
        simulator.reset()
        for index, row in enumerate(values):
            observation = simulator.observe()
            for name in sorted(observation.cameras):
                camera = observation.cameras[name]
                Image.fromarray(camera.rgb, mode="RGB").save(output_dir / f"{name}-{observation.step_index:08d}.png")
                if save_depth and camera.depth_m is not None:
                    np.save(output_dir / f"{name}-depth-{observation.step_index:08d}.npy", camera.depth_m)
            if not full_horizon and simulator.succeeded():
                success = True
                break
            chunk = ActionChunk(
                request_id=f"replay-{index}",
                session_id=key.folder_name(),
                start_step=observation.step_index,
                spec=setup.policy.action_spec,
                values=np.asarray(row, dtype=np.float32),
                execution_count=setup.policy.execution_count,
            )
            simulator.apply(setup.validate_action(chunk))
            executed += 1
            if not full_horizon and simulator.succeeded():
                success = True
                break
        if full_horizon:
            success = simulator.succeeded()
        metrics = simulator.benchmark_metrics()
        completed = True
    finally:
        simulator.close(force=True)
        if completed:
            shutil.rmtree(runtime_dir, ignore_errors=True)
    report = {
        "episode_id": key.folder_name(),
        "task_id": key.task_id,
        "actions_applied": executed,
        "actions_available": len(values),
        "success": bool(success),
        "recorded_success": manifest.get("success"),
        "benchmark_metrics": metrics,
        "pictures": str(output_dir),
    }
    (output_dir / "replay.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roboarchitect.replay", description="Play one recorded episode again in the simulator and save every camera picture. Reads the task, scene and seeds from episode.json and the actions from trace.jsonl, so it repeats the episode that was recorded. Pass --actions to play a different action sequence instead and see what would have happened. No policy and no other model is loaded.")
    parser.add_argument("--episode", type=Path, required=True, help="an episode folder holding episode.json and trace.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="the folder to write the camera images to")
    parser.add_argument("--render-gpu", type=int, default=0, help="the GPU that renders the simulator")
    parser.add_argument("--actions", type=Path, help="a trace.jsonl to take the actions from instead")
    parser.add_argument("--steps", type=int, default=0, help="stop after this many steps (0 means all)")
    parser.add_argument("--depth", action="store_true", help="also save each camera's depth as .npy")
    args = parser.parse_args(argv)
    report = replay(
        args.episode,
        args.out,
        render_gpu=args.render_gpu,
        action_source=args.actions,
        max_steps=args.steps,
        save_depth=args.depth,
    )
    json.dump(report, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
