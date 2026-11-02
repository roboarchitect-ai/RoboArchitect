from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


HARNESS_FILE = "harness.py"
EFFORT = "max"
MAX_HOURS_PER_REVISION = 6


class CodingModel:
    def __init__(self, model: str) -> None:
        executable = shutil.which("claude")
        if executable is None:
            raise FileNotFoundError("the claude command (Claude Code) is not on PATH")
        self.executable = Path(executable).resolve()
        self.model = model
        self.timeout_s = MAX_HOURS_PER_REVISION * 3600.0

    def command(self) -> list[str]:
        return [
            str(self.executable),
            "-p",
            "--model",
            self.model,
            "--permission-mode",
            "acceptEdits",
            "--allowedTools",
            "Read,Edit,Write,Bash,Grep,Glob",
            "--output-format",
            "stream-json",
            "--verbose",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--disable-slash-commands",
            "--no-session-persistence",
            "--effort",
            EFFORT,
        ]

    def revise(self, prompt: str, harness_dir: Path, log_dir: Path) -> None:
        harness_dir = Path(harness_dir).resolve()
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        harness = harness_dir / HARNESS_FILE
        before = harness.read_text(encoding="utf-8")
        environment = dict(os.environ)
        environment["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
        transcript = log_dir / "transcript.jsonl"
        started = time.time()
        with open(transcript, "w") as handle:
            try:
                completed = subprocess.run(
                    self.command(),
                    cwd=str(harness_dir),
                    input=prompt,
                    stdout=handle,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_s,
                    env=environment,
                )
            except subprocess.TimeoutExpired as exc:
                (log_dir / "stderr.txt").write_text(f"TIMEOUT after {time.time() - started:.0f}s\n{exc}", encoding="utf-8")
                raise RuntimeError(f"the coding model timed out after {self.timeout_s:.0f}s") from exc
        stderr = completed.stderr or ""
        if stderr:
            (log_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
        transcript_bytes = transcript.stat().st_size if transcript.is_file() else 0
        after = harness.read_text(encoding="utf-8")
        if completed.returncode != 0 or transcript_bytes == 0:
            head = stderr.strip().splitlines()
            raise RuntimeError(f"the coding model failed: exit code {completed.returncode}, transcript {transcript_bytes} bytes, stderr {(head[0] if head else '(none)')[:300]}")
        if after == before:
            raise RuntimeError("the coding model did not change harness.py")
        compile(after, str(harness), "exec")
