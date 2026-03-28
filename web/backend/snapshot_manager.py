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
        entry["resumable"] = (run_dir / REDIS_SNAPSHOT_FILE).exists()
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


REDIS_SNAPSHOT_FILE = "redis_snapshot.json"


def load_redis_snapshot(run_dir: Path) -> Optional[dict]:
    """Load a saved Redis snapshot from a run folder. Returns None if not found."""
    f = run_dir / REDIS_SNAPSHOT_FILE
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def restore_redis_from_snapshot(snap: dict, redis_port: int = 6379, redis_db: int = 0) -> int:
    """Restore Redis from a saved snapshot. Returns number of programs restored."""
    r = redis_lib.Redis(port=redis_port, db=redis_db,
                        decode_responses=True, socket_connect_timeout=2)
    try:
        r.flushdb()
        pipe = r.pipeline(transaction=False)

        # Restore program keys
        programs = snap.get("programs", {})
        for key, raw in programs.items():
            pipe.set(key, raw)

        # Restore archive hash (cell -> pid)
        archive = snap.get("archive", {})
        if archive:
            pipe.hset(snap["archive_key"], mapping=archive)

        # Restore reverse index (pid -> cell)
        reverse = snap.get("archive_reverse", {})
        if reverse:
            pipe.hset(snap["archive_reverse_key"], mapping=reverse)

        # Restore atomic counter
        if snap.get("ts_key") and snap.get("ts"):
            pipe.set(snap["ts_key"], snap["ts"])

        pipe.execute()
        return len(programs)
    finally:
        r.close()


class SnapshotManager:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stop_flag = False
        self._last_archive: dict[str, str] = {}
        # stored so stop() can save the Redis snapshot
        self._phf: str = ""
        self._run_id: str = ""
        self._redis_port: int = 6379
        self._redis_db: int = 0

    def start(self, phf: str, run_id: str, redis_port: int = 6379, redis_db: int = 0) -> None:
        """Start background polling. Safe to call multiple times (stops previous run)."""
        self.stop()
        self._stop_flag = False
        self._last_archive = {}
        self._phf = phf
        self._run_id = run_id
        self._redis_port = redis_port
        self._redis_db = redis_db
        self._task = asyncio.create_task(
            self._poll_loop(phf, run_id, redis_port, redis_db)
        )

    def stop(self) -> None:
        self._stop_flag = True
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        if self._phf and self._run_id:
            try:
                self._save_redis_snapshot(
                    self._phf,
                    SNAPSHOTS_DIR / self._phf / self._run_id,
                    self._redis_port,
                    self._redis_db,
                )
            except Exception:
                pass

    def _save_redis_snapshot(self, phf: str, run_dir: Path, redis_port: int, redis_db: int) -> None:
        """Capture current Redis state to redis_snapshot.json for future resume."""
        try:
            r = redis_lib.Redis(port=redis_port, db=redis_db,
                                decode_responses=True, socket_connect_timeout=2)
        except Exception:
            return
        try:
            archive_keys = [k for k in r.keys("*:archive")
                            if not k.endswith(":archive:reverse")]
            if not archive_keys:
                return
            archive_key = archive_keys[0]
            archive_reverse_key = archive_key + ":reverse"
            ts_key = f"{phf}:ts"

            archive: dict = r.hgetall(archive_key) or {}
            if not archive:
                return  # nothing to save

            # Save raw program strings for current elites only
            programs: dict[str, str] = {}
            for pid in archive.values():
                prog_key = f"{phf}:program:{pid}"
                raw = r.get(prog_key)
                if raw:
                    programs[prog_key] = raw

            snap = {
                "phf": phf,
                "archive_key": archive_key,
                "archive": archive,
                "archive_reverse_key": archive_reverse_key,
                "archive_reverse": r.hgetall(archive_reverse_key) or {},
                "ts_key": ts_key,
                "ts": r.get(ts_key),
                "programs": programs,
            }

            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / REDIS_SNAPSHOT_FILE).write_text(
                json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        finally:
            try:
                r.close()
            except Exception:
                pass

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
        """Synchronous tick: save every newly evaluated program from phash:program:*.

        Mirrors what the Results tab shows — reads phash:program:* directly.
        Bin is taken from archive:reverse if available, otherwise 'unplaced'.
        """
        try:
            r = redis_lib.Redis(port=redis_port, db=redis_db,
                                decode_responses=True, socket_connect_timeout=2)
        except Exception:
            return

        try:
            prog_keys = r.keys(f"{phf}:program:*")
            if not prog_keys:
                return

            # Build reverse index: program_id -> cell_key (best-effort)
            reverse: dict[str, str] = {}
            rev_keys = [k for k in r.keys("*:archive:reverse")]
            for rk in rev_keys:
                reverse.update(r.hgetall(rk) or {})

            history = _load_history(run_dir)
            bins: dict = history.setdefault("bins", {})
            dirty = False

            # Reclassify programs previously saved as "unplaced" that now
            # have a known cell assignment in the archive reverse index.
            for pid, cell in list(self._last_archive.items()):
                if cell != "unplaced":
                    continue
                actual_cell = reverse.get(pid)
                if actual_cell is None:
                    continue
                raw = r.get(f"{phf}:program:{pid}")
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                name = _program_name(data, pid)
                unplaced_bin = bins.get("unplaced", [])
                entry = next((e for e in unplaced_bin if e["name"] == name), None)
                if entry is not None:
                    unplaced_bin.remove(entry)
                    if not unplaced_bin:
                        bins.pop("unplaced", None)
                    bins.setdefault(actual_cell, []).append(entry)
                    src = run_dir / _cell_to_folder("unplaced") / f"{name}.py"
                    dst_dir = run_dir / _cell_to_folder(actual_cell)
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst = dst_dir / f"{name}.py"
                    if src.exists() and not dst.exists():
                        src.rename(dst)
                    dirty = True
                self._last_archive[pid] = actual_cell

            for pk in prog_keys:
                pid = pk.split(":")[-1]  # UUID part
                if pid in self._last_archive:
                    continue  # already saved

                raw = r.get(pk)
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue

                if data.get("state") != "done":
                    continue

                metrics = data.get("metrics", {})
                if isinstance(metrics, str):
                    try:
                        metrics = json.loads(metrics)
                    except Exception:
                        metrics = {}

                efficiency = metrics.get("efficiency")
                if efficiency is None:
                    continue  # not yet evaluated

                cell_key = reverse.get(pid, "unplaced")
                name = _program_name(data, pid)

                entry = {
                    "name": name,
                    "efficiency": efficiency,
                    "ASR": metrics.get("asr"),
                    "L2": metrics.get("l2"),
                }

                bin_history: list = bins.setdefault(cell_key, [])
                existing_names = {e["name"] for e in bin_history}
                if name not in existing_names:
                    bin_history.append(entry)
                    dirty = True
                    code = data.get("code", "")
                    if code:
                        _save_program_code(run_dir, cell_key, name, code)

                self._last_archive[pid] = cell_key  # mark as seen

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
