from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from roboarchitect.protocol import StrictSchemaError
from roboarchitect.records import BenchmarkOutcome, Score, EpisodeManifest, validate_benchmark_metrics

from benchmarks.robocasa import TARGET_TASK_GROUPS
from benchmarks.robocerebra import CONDITIONS, parse_task_id
from benchmarks.vlabench import parse_vlabench_scenario


REQUIRED_OUTCOME_METRICS = {
    "equal_track_task_macro_progress_score": frozenset({"progress_score"}),
}
OPTIONAL_OUTCOME_METRICS = {
    "equal_benchmark_task_macro_score": frozenset({"progress_score"}),
}


def _checked(values: Iterable[BenchmarkOutcome]) -> tuple[BenchmarkOutcome, ...]:
    rows = tuple(sorted(values, key=lambda item: item.key))
    if not rows or any(not isinstance(item, BenchmarkOutcome) for item in rows):
        raise StrictSchemaError("a score needs at least one episode outcome")
    if len({item.key for item in rows}) != len(rows):
        raise StrictSchemaError("an episode outcome appears twice")
    return rows


def _mean(values: Iterable[float], path: str) -> float:
    rows = tuple(float(value) for value in values)
    if not rows or any(not math.isfinite(value) for value in rows):
        raise StrictSchemaError(f"{path}: expected finite values")
    return sum(rows) / len(rows)


def _task_means(rows: tuple[BenchmarkOutcome, ...], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.metrics.get(metric)
        if type(value) is bool:
            number = float(value)
        elif type(value) in (int, float) and math.isfinite(float(value)):
            number = float(value)
        else:
            raise StrictSchemaError(f"an episode outcome lacks a number for {metric!r}")
        grouped.setdefault(row.key.task_id, []).append(number)
    return {task: _mean(values, f"task {task}") for task, values in sorted(grouped.items())}


def task_macro_success(rows: tuple[BenchmarkOutcome, ...]) -> Score:
    tasks = _task_means(rows, "success")
    return Score(
        "task_macro_success",
        _mean(tasks.values(), "task macro"),
        {"task_success": tasks, "n_tasks": len(tasks), "n_episodes": len(rows)},
    )


def equal_cell_task_macro_success(rows: tuple[BenchmarkOutcome, ...]) -> Score:
    tasks = _task_means(rows, "success")
    cells: dict[str, list[float]] = {}
    for task, value in tasks.items():
        if "::" not in task:
            raise StrictSchemaError("LIBERO-Pro task lacks a cell namespace")
        cell, task_name = task.split("::", 1)
        if not cell.startswith("libero_pro_") or not task_name:
            raise StrictSchemaError("LIBERO-Pro task namespace differs")
        cells.setdefault(cell, []).append(value)
    cell_success = {cell: _mean(values, cell) for cell, values in sorted(cells.items())}
    return Score(
        "equal_cell_task_macro_success",
        _mean(cell_success.values(), "LIBERO-Pro cell macro"),
        {"cell_success": cell_success, "task_success": tasks, "n_episodes": len(rows)},
    )


def equal_track_task_macro_progress_score(rows: tuple[BenchmarkOutcome, ...]) -> Score:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        value = row.metrics.get("progress_score")
        if type(value) not in (int, float) or type(value) is bool or not 0.0 <= float(value) <= 1.0:
            raise StrictSchemaError("VLABench outcome requires progress_score in [0, 1]")
        track, _ = parse_vlabench_scenario(row.key.scenario_id)
        grouped.setdefault(track, {}).setdefault(row.key.task_id, []).append(float(value))
    task_progress = {
        track: {task: _mean(values, f"{track}/{task}") for task, values in sorted(tasks.items())}
        for track, tasks in sorted(grouped.items())
    }
    track_progress = {track: _mean(tasks.values(), track) for track, tasks in task_progress.items()}
    return Score(
        "equal_track_task_macro_progress_score",
        _mean(track_progress.values(), "VLABench track macro"),
        {"track_progress": track_progress, "task_progress": task_progress, "n_episodes": len(rows)},
    )


def equal_group_task_macro_success(rows: tuple[BenchmarkOutcome, ...]) -> Score:
    tasks = _task_means(rows, "success")
    expected = {task for values in TARGET_TASK_GROUPS.values() for task in values}
    unknown = set(tasks) - expected
    if unknown:
        raise StrictSchemaError(f"RoboCasa scalar contains unknown task {sorted(unknown)[0]!r}")
    groups = {
        group: _mean((tasks[task] for task in group_tasks if task in tasks), group)
        for group, group_tasks in TARGET_TASK_GROUPS.items()
        if any(task in tasks for task in group_tasks)
    }
    return Score(
        "equal_group_task_macro_success",
        _mean(groups.values(), "RoboCasa group macro"),
        {"group_success": groups, "task_success": tasks, "n_episodes": len(rows)},
    )


def equal_condition_case_macro_success(rows: tuple[BenchmarkOutcome, ...]) -> Score:
    tasks = _task_means(rows, "success")
    grouped: dict[str, list[float]] = {condition: [] for condition in CONDITIONS}
    for task, value in tasks.items():
        condition, _ = parse_task_id(task)
        grouped[condition].append(value)
    conditions = {condition: _mean(values, condition) for condition, values in grouped.items() if values}
    return Score(
        "equal_condition_case_macro_success",
        _mean(conditions.values(), "RoboCerebra condition macro"),
        {"condition_success": conditions, "case_success": tasks, "n_episodes": len(rows)},
    )


def benchmark_family(task_id: str) -> str:
    suite, separator, _ = task_id.partition("::")
    if not separator:
        return "unnamespaced"
    if suite.startswith("libero_pro_"):
        return "libero_pro"
    if suite.startswith("simpler_google"):
        return "simpler_google"
    if suite.startswith("simpler_widowx"):
        return "simpler_widowx"
    if suite.startswith("vlabench"):
        return "vlabench"
    return suite


def equal_benchmark_task_macro_score(rows: tuple[BenchmarkOutcome, ...]) -> Score:
    per_task: dict[str, list[float]] = {}
    for row in rows:
        value = row.metrics.get("progress_score")
        if type(value) not in (int, float) or type(value) is bool:
            value = float(bool(row.metrics["success"]))
        per_task.setdefault(row.key.task_id, []).append(float(value))
    task_score = {task: _mean(values, f"task {task}") for task, values in sorted(per_task.items())}
    families: dict[str, list[float]] = {}
    for task, value in task_score.items():
        families.setdefault(benchmark_family(task), []).append(value)
    family_score = {name: _mean(values, name) for name, values in sorted(families.items())}
    return Score(
        "equal_benchmark_task_macro_score",
        _mean(family_score.values(), "unified benchmark macro"),
        {
            "benchmark_score": family_score,
            "task_score": task_score,
            "n_benchmarks": len(family_score),
            "n_tasks": len(task_score),
            "n_episodes": len(rows),
        },
    )


METRICS = {
    "task_macro_success": task_macro_success,
    "equal_cell_task_macro_success": equal_cell_task_macro_success,
    "equal_track_task_macro_progress_score": equal_track_task_macro_progress_score,
    "equal_group_task_macro_success": equal_group_task_macro_success,
    "equal_condition_case_macro_success": equal_condition_case_macro_success,
    "equal_benchmark_task_macro_score": equal_benchmark_task_macro_score,
}


def compute_metric(metric: str, values: Iterable[BenchmarkOutcome]) -> Score:
    if metric not in METRICS:
        raise StrictSchemaError(f"unknown metric {metric!r}; known metrics are {sorted(METRICS)}")
    return METRICS[metric](_checked(values))


def episode_outcome(episode_dir: Path, manifest: EpisodeManifest, metric: str) -> BenchmarkOutcome:
    required = REQUIRED_OUTCOME_METRICS.get(metric, frozenset())
    optional = OPTIONAL_OUTCOME_METRICS.get(metric, frozenset())
    if manifest.state == "error":
        return BenchmarkOutcome(manifest.key, {"success": False, **{name: 0.0 for name in sorted(required)}})
    metrics: dict[str, bool | float] = {"success": bool(manifest.success)}
    source = Path(episode_dir) / "benchmark_metrics.json"
    if (required or optional) and source.is_file():
        measured = validate_benchmark_metrics(json.loads(source.read_text(encoding="utf-8")))
        if "success" in measured and measured["success"] is not manifest.success:
            raise StrictSchemaError("the benchmark's own success differs from episode.json")
        missing = required - set(measured)
        if missing:
            raise StrictSchemaError(f"the episode lacks the metric {sorted(missing)[0]!r} its score needs")
        metrics.update({name: measured[name] for name in sorted(required)})
        metrics.update({name: measured[name] for name in sorted(optional & set(measured))})
    elif required:
        raise StrictSchemaError("the episode lacks the benchmark_metrics.json its score needs")
    return BenchmarkOutcome(manifest.key, metrics)
