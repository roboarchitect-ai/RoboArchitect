from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from roboarchitect.coding_model import CodingModel
from roboarchitect.evaluation import EvaluationResult, Evaluator, write_json
from roboarchitect.protocol import StrictSchemaError
from roboarchitect.prompt import revision_prompt


HARNESS_FILE = "harness.py"
GENERATED = ("__pycache__",)
STARTING = "iterations/0000"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _harness_files(root: Path) -> dict[str, bytes]:
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise StrictSchemaError(f"a harness folder must not hold a symlink: {path}")
        relative = path.relative_to(root)
        if path.is_file() and not any(part in GENERATED for part in relative.parts) and path.suffix not in {".pyc", ".pyo"}:
            result[relative.as_posix()] = path.read_bytes()
    if set(result) != {HARNESS_FILE}:
        raise StrictSchemaError(f"a harness folder must hold harness.py and nothing else: {sorted(result)}")
    compile(result[HARNESS_FILE].decode("utf-8"), str(root / HARNESS_FILE), "exec")
    return result


def decision(best: float, revision: float) -> dict[str, Any]:
    if not math.isfinite(best) or not math.isfinite(revision):
        raise StrictSchemaError("scores must be finite")
    return {
        "accepted": revision > best,
        "best_score": float(best),
        "revision_score": float(revision),
        "change": float(revision) - float(best),
    }


class HarnessOptimizer:
    def __init__(
        self,
        *,
        experiment_dir: Path,
        evaluator: Evaluator,
        held_out_evaluator: Evaluator,
        coding_model: CodingModel,
        facts: str,
        starting_harness: Path,
        iterations: int,
    ) -> None:
        self.experiment_dir = Path(experiment_dir).resolve()
        self.evaluator = evaluator
        self.held_out_evaluator = held_out_evaluator
        self.coding_model = coding_model
        self.facts = facts
        self.starting_harness = Path(starting_harness).resolve()
        self.iterations = iterations

    @property
    def state_path(self) -> Path:
        return self.experiment_dir / "state.json"

    def _progress(self, message: str) -> None:
        with open(self.experiment_dir / "progress.log", "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}\n")

    def _score_row(self, iteration: int, score: Any, best: Any, accepted: Any, errored: Any, seconds: Any) -> None:
        path = self.experiment_dir / "scores.csv"
        header = not path.exists()
        with open(path, "a", encoding="utf-8") as handle:
            if header:
                handle.write("iteration,score,best_score,accepted,errored_episodes,seconds\n")
            handle.write(f"{iteration},{score},{best},{accepted},{errored},{seconds}\n")

    def _warning(self, result: EvaluationResult) -> str:
        if not result.errored_episodes:
            return ""
        return f"  WARNING: {result.errored_episodes} of {len(result.outcomes)} episodes stopped with an error and count as failures; read evaluation/episodes/*/episode.json for the error text"

    def _write_state(self, state: Mapping[str, Any]) -> None:
        write_json(self.state_path, dict(state))

    def _archive(self, staging: Path, label: str, error: str) -> None:
        if not staging.exists():
            return
        failures = self.experiment_dir / "failures"
        failures.mkdir(exist_ok=True)
        index = 1
        while (failures / f"{label}-{index:04d}").exists():
            index += 1
        target = failures / f"{label}-{index:04d}"
        os.rename(staging, target)
        write_json(target / "failure.json", {"label": label, "error": error, "archived_ns": time.time_ns()})

    def _evaluate(self, evaluator: Evaluator, harness_dir: Path, folder: Path) -> tuple[EvaluationResult, float]:
        started = time.monotonic()
        result = evaluator.evaluate(harness_dir, folder / "evaluation")
        write_json(folder / "result.json", result.to_mapping())
        return result, time.monotonic() - started

    def _starting_harness(self) -> dict[str, Any]:
        staging = self.experiment_dir / "iterations" / ".0000-staging"
        if not (staging / "harness" / HARNESS_FILE).is_file():
            shutil.rmtree(staging, ignore_errors=True)
            (staging / "harness").mkdir(parents=True)
            shutil.copyfile(self.starting_harness, staging / "harness" / HARNESS_FILE)
        _harness_files(staging / "harness")
        self._progress("iteration 0000: evaluating the starting harness ...")
        try:
            result, seconds = self._evaluate(self.evaluator, staging / "harness", staging)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            self._archive(staging, "iteration-0000-failed", f"{type(exc).__name__}: {exc}")
            raise
        os.replace(staging, self.experiment_dir / STARTING)
        state = {"next_iteration": 1, "best": STARTING, "finished": False}
        self._write_state(state)
        count = len(result.outcomes)
        self._progress(f"iteration 0000: {result.score.metric}={result.score.value:.4f} ({count} episode{'' if count == 1 else 's'}, {seconds:.0f}s)" + self._warning(result))
        self._score_row(0, f"{result.score.value:.6f}", "", "", result.errored_episodes, f"{seconds:.0f}")
        return state

    def _best_result(self, state: Mapping[str, Any]) -> EvaluationResult:
        return EvaluationResult.load(self.experiment_dir / state["best"] / "result.json")

    def _feedback(self, state: Mapping[str, Any], index: int, staging: Path) -> None:
        best = self.experiment_dir / state["best"]
        previous = self.experiment_dir / "iterations" / f"{index - 1:04d}"
        rejected = index > 1 and previous != best and (previous / "result.json").is_file()
        best_memory = best / "evaluation" / "memory.json"
        rejected_memory = previous / "evaluation" / "memory.json"
        write_json(
            staging / "feedback.json",
            {
                "iteration": index,
                "metric": self.evaluator.metric,
                "current_harness_score": self._best_result(state).score.value,
                "current_harness_recordings": str(best / "evaluation" / "episodes"),
                "current_harness_memory": str(best_memory) if best_memory.is_file() else None,
                "rejected_revision_score": EvaluationResult.load(previous / "result.json").score.value if rejected else None,
                "rejected_revision_recordings": str(previous / "evaluation" / "episodes") if rejected else None,
                "rejected_revision_memory": str(rejected_memory) if rejected and rejected_memory.is_file() else None,
            },
        )

    def _keep_workspace(self, staging: Path) -> int:
        moved = 0
        for path in sorted((staging / "harness").iterdir()):
            if path.name == HARNESS_FILE:
                continue
            target = staging / "workspace" / path.name
            suffix = 1
            while target.exists():
                suffix += 1
                target = staging / "workspace" / f"{path.name}.{suffix}"
            target.parent.mkdir(exist_ok=True)
            os.rename(path, target)
            moved += 1
        return moved

    def _iteration(self, state: dict[str, Any]) -> dict[str, Any]:
        index = state["next_iteration"]
        staging = self.experiment_dir / "iterations" / f".{index:04d}-staging"
        best = self.experiment_dir / state["best"]
        resuming = (staging / "evaluation").is_dir() and (staging / "harness" / HARNESS_FILE).is_file()
        if staging.exists() and not resuming:
            self._archive(staging, f"iteration-{index:04d}-interrupted", "restarted before its evaluation began")
        try:
            if resuming:
                self._progress(f"iteration {index:04d}: resuming the evaluation of the revised harness ...")
            else:
                (staging / "workspace").mkdir(parents=True)
                shutil.copytree(best / "harness", staging / "harness")
                (staging / "harness" / HARNESS_FILE).chmod(0o600)
                self._feedback(state, index, staging)
                prompt = revision_prompt(self.facts)
                (staging / "prompt.txt").write_text(prompt, encoding="utf-8")
                self._progress(f"iteration {index:04d}: the coding model is revising {state['best']} ...")
                try:
                    self.coding_model.revise(prompt, staging / "harness", staging / "coding_model_logs")
                finally:
                    kept = self._keep_workspace(staging)
                self._progress(f"iteration {index:04d}: revised; moved {kept} extra file(s) from the harness folder to workspace/; evaluating ...")
            if _harness_files(staging / "harness") == _harness_files(best / "harness"):
                raise StrictSchemaError("the revision did not change harness.py")
            result, seconds = self._evaluate(self.evaluator, staging / "harness", staging)
            best_score = self._best_result(state).score.value
            outcome = decision(best_score, result.score.value)
            write_json(staging / "decision.json", outcome)
            self._progress(f"iteration {index:04d}: {result.score.metric}={result.score.value:.4f} vs best {best_score:.4f} -> {'accepted' if outcome['accepted'] else 'rejected'} ({seconds:.0f}s)" + self._warning(result))
            self._score_row(
                index, f"{result.score.value:.6f}", f"{best_score:.6f}",
                outcome["accepted"], result.errored_episodes, f"{seconds:.0f}",
            )
            os.replace(staging, self.experiment_dir / "iterations" / f"{index:04d}")
            state = dict(state, next_iteration=index + 1)
            if outcome["accepted"]:
                state["best"] = f"iterations/{index:04d}"
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            self._archive(staging, f"iteration-{index:04d}-failed", f"{type(exc).__name__}: {exc}")
            self._progress(f"iteration {index:04d}: FAILED ({type(exc).__name__}: {exc}); the best harness so far is kept")
            self._score_row(index, "failed", "", "", "", "")
            state = dict(state, next_iteration=index + 1)
        self._write_state(state)
        return state

    def _resume_state(self) -> dict[str, Any] | None:
        if self.state_path.is_file():
            state = dict(_load_json(self.state_path))
        elif (self.experiment_dir / STARTING / "result.json").is_file():
            state = {"next_iteration": 1, "best": STARTING, "finished": False}
        else:
            return None
        while (self.experiment_dir / "iterations" / f"{state['next_iteration']:04d}" / "decision.json").is_file():
            folder = self.experiment_dir / "iterations" / f"{state['next_iteration']:04d}"
            if _load_json(folder / "decision.json")["accepted"]:
                state["best"] = f"iterations/{state['next_iteration']:04d}"
            state["next_iteration"] += 1
        return state

    def _finish(self, state: dict[str, Any]) -> dict[str, Any]:
        final = self.experiment_dir / "final"
        if not final.exists():
            staging = self.experiment_dir / ".final-staging"
            shutil.rmtree(staging, ignore_errors=True)
            shutil.copytree(self.experiment_dir / state["best"] / "harness", staging / "harness")
            write_json(staging / "final.json", {"from": state["best"], "finished_ns": time.time_ns()})
            os.replace(staging, final)
            self._progress(f"final harness: {state['best']}")
        state = dict(state, finished=True)
        self._write_state(state)
        return state

    def _held_out(self) -> dict[str, Any]:
        root = self.experiment_dir / "held_out"
        if (root / "comparison.json").is_file():
            return _load_json(root / "comparison.json")
        self._progress("held-out: evaluating the starting harness and the final harness ...")
        starting, _ = self._evaluate(self.held_out_evaluator, self.experiment_dir / STARTING / "harness", root / "starting")
        final, _ = self._evaluate(self.held_out_evaluator, self.experiment_dir / "final" / "harness", root / "final")
        comparison = {
            "metric": starting.score.metric,
            "starting_harness": starting.score.value,
            "final_harness": final.score.value,
            "change": final.score.value - starting.score.value,
        }
        write_json(root / "comparison.json", comparison)
        self._progress(f"held-out: starting harness {starting.score.value:.4f} vs final harness {final.score.value:.4f}" + self._warning(starting) + self._warning(final))
        return comparison

    def optimize(self) -> dict[str, Any]:
        (self.experiment_dir / "iterations").mkdir(parents=True, exist_ok=True)
        state = self._resume_state() or self._starting_harness()
        while not state["finished"] and state["next_iteration"] <= self.iterations:
            state = self._iteration(state)
        state = self._finish(state)
        return {"state": state, "held_out": self._held_out()}
