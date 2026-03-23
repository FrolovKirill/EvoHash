"""Snapshot manager: tracks MAP-Elites archive changes and saves history to disk.

Storage layout:
    snapshots/{phf}/{run_id}/
        history.json          — bin histories (grows, never shrinks)
        bin_{cell}/
            {program_name}.py — program source code

history.json structure:
    {
        "bins": {
            "0":   [{"name": "a1b2c3d4", "efficiency": 0.023, "ASR": 1.0, "L2": 42.1}, ...],
            "1,2": [...]
        }
    }

Polling logic:
    Every POLL_INTERVAL seconds, read {phf}:archive (Redis hash: cell -> program_id).
    When a cell's program_id changes (or a new cell appears), fetch the program data
    and append to that cell's history array.  This catches replacements as long as
    the poll interval is shorter than the typical generation time (~seconds).
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import redis as redis_lib

PROJECT_ROOT = Path(__file__).parent.parent.parent
SNAPSHOTS_DIR = PROJECT_ROOT / "snapshots"
POLL_INTERVAL = 3  # seconds between archive polls


def generate_run_id() -> str:
    from datetime import datetime
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def list_runs(phf: str) -> list[dict]:
    """Return metadata for all saved runs of a given PHF, newest first."""
    phf_dir = SNAPSHOTS_DIR / phf
    if not phf_dir.exists():
        return []
    runs = []
    for run_dir in sorted(phf_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        history_file = run_dir / "history.json"
        entry: dict = {"run_id": run_dir.name}
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text(encoding="utf-8"))
                bins = data.get("bins", {})
                total = sum(len(v) for v in bins.values())
                entry["programs"] = total
                entry["bins"] = len(bins)
            except Exception:
                pass
        runs.append(entry)
    return runs


def _cell_to_folder(cell_key: str) -> str:
    """Convert cell key like '3' or '3,5' to a safe folder name."""
    return "bin_" + cell_key.replace(",", "_")


def _load_history(run_dir: Path) -> dict:
    history_file = run_dir / "history.json"
    if history_file.exists():
        try:
            return json.loads(history_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"bins": {}}


def _save_history(run_dir: Path, history: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    history_file = run_dir / "history.json"
    history_file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_program_code(run_dir: Path, cell_key: str, name: str, code: str) -> None:
    bin_dir = run_dir / _cell_to_folder(cell_key)
    bin_dir.mkdir(parents=True, exist_ok=True)
    py_file = bin_dir / f"{name}.py"
    if not py_file.exists():
        py_file.write_text(code, encoding="utf-8")


def _fetch_program(r: redis_lib.Redis, phf: str, pid: str) -> Optional[dict]:
    key = f"{phf}:program:{pid}"
    raw = r.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _program_name(data: dict, pid: str) -> str:
    """Return a short, filesystem-safe name for a program."""
    name = data.get("name", "") or data.get("id", pid)
    # Trim to first 32 chars and strip non-alphanum
    safe = "".join(c for c in str(name) if c.isalnum() or c in "_-")[:32]
    return safe or pid[:16]


class SnapshotManager:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stop_flag = False
        # cell_key -> program_id of the last snapshot we recorded
        self._last_archive: dict[str, str] = {}

    def start(self, phf: str, run_id: str, redis_port: int = 6379, redis_db: int = 0) -> None:
        """Start background polling. Safe to call multiple times (stops previous run)."""
        self.stop()
        self._stop_flag = False
        self._last_archive = {}
        self._task = asyncio.create_task(
            self._poll_loop(phf, run_id, redis_port, redis_db)
        )

    def stop(self) -> None:
        self._stop_flag = True
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _poll_loop(self, phf: str, run_id: str, redis_port: int, redis_db: int) -> None:
        run_dir = SNAPSHOTS_DIR / phf / run_id
        loop = asyncio.get_event_loop()

        while not self._stop_flag:
            try:
                await loop.run_in_executor(
                    None, self._tick, phf, run_dir, redis_port, redis_db
                )
            except Exception:
                pass
            await asyncio.sleep(POLL_INTERVAL)

    def _tick(self, phf: str, run_dir: Path, redis_port: int, redis_db: int) -> None:
        """Synchronous tick: diff archive, save new programs."""
        try:
            r = redis_lib.Redis(port=redis_port, db=redis_db,
                                decode_responses=True, socket_connect_timeout=2)
        except Exception:
            return

        try:
            # {phf}:archive is a hash: cell_key -> program_id
            archive_key = f"{phf}:archive"
            current_archive: dict[str, str] = r.hgetall(archive_key) or {}

            if not current_archive:
                return

            # Find changed/new cells
            changed: list[tuple[str, str]] = []
            for cell_key, pid in current_archive.items():
                if self._last_archive.get(cell_key) != pid:
                    changed.append((cell_key, pid))

            if not changed:
                return

            # Load history from disk once
            history = _load_history(run_dir)
            bins: dict = history.setdefault("bins", {})
            dirty = False

            for cell_key, pid in changed:
                data = _fetch_program(r, phf, pid)
                if data is None:
                    continue

                metrics = data.get("metrics", {})
                if isinstance(metrics, str):
                    try:
                        metrics = json.loads(metrics)
                    except Exception:
                        metrics = {}

                name = _program_name(data, pid)
                entry = {
                    "name": name,
                    "efficiency": metrics.get("efficiency"),
                    "ASR": metrics.get("asr"),
                    "L2": metrics.get("l2"),
                }

                bin_history: list = bins.setdefault(cell_key, [])

                # Avoid duplicate entries (same program_id seen again after resume)
                existing_names = {e["name"] for e in bin_history}
                if name not in existing_names:
                    bin_history.append(entry)
                    dirty = True

                    code = data.get("code", "")
                    if code:
                        _save_program_code(run_dir, cell_key, name, code)

                self._last_archive[cell_key] = pid

            if dirty:
                _save_history(run_dir, history)

        finally:
            try:
                r.close()
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()


snapshot_manager = SnapshotManager()
