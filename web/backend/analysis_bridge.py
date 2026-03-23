"""Background LLM analysis of MAP-Elites archive programs.

Runs alongside evolution: periodically fetches done programs from Redis,
sends new ones to LLM for baseline similarity classification, saves snapshots.

Storage layout:
    snapshots/{phf}/{run_id}/
        _history.json          — all_seen_ids + analysis_cache
        {phf}_{timestamp}.json — periodic archive snapshots
"""
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
SNAPSHOTS_DIR = PROJECT_ROOT / "snapshots"


def _make_prompt(code: str) -> str:
    return (
        "You are an expert in adversarial attacks on perceptual hash functions.\n"
        "Below is evolved attack code. Classify it by similarity to these INDIVIDUAL baselines:\n"
        "NES (antithetic gradient estimation), SimBa (DCT basis coordinate descent),\n"
        "HSJA (decision-based boundary walk from target), Prokos (frequency-weighted DCT gradient),\n"
        "AtkScopes (multi-scale patch sensitivity), ZO-SignSGD (sign of forward-diff gradient),\n"
        "RandomNoise (random sampling).\n\n"
        "IMPORTANT: score each baseline INDEPENDENTLY. Do NOT combine baselines (no 'NES+HSJA' etc).\n"
        "If the code uses ideas from multiple baselines, list each one separately with its own score.\n\n"
        'Return ONLY a JSON object (no markdown):\n'
        '{"top3": [["NES", 0.85], ["HSJA", 0.40], ["SimBa", 0.15]]}\n'
        'top3 = three most similar individual baselines with scores 0.0-1.0. Use "Novel" if none fit.\n\n'
        f"```python\n{code}\n```"
    )


def _load_api_key() -> Optional[str]:
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        for line in dotenv_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY=") and not line.endswith("..."):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")


def generate_run_id() -> str:
    """Generate a human-readable run ID like 'run_20260323_145500'."""
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}"


def list_runs(phf: str) -> list[dict[str, Any]]:
    """List all available runs for a PHF with metadata."""
    phf_dir = SNAPSHOTS_DIR / phf
    if not phf_dir.exists():
        return []
    runs = []
    for run_dir in sorted(phf_dir.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        history_path = run_dir / "_history.json"
        meta: dict[str, Any] = {"run_id": run_dir.name, "phf": phf}
        # Count snapshots
        snapshots = sorted(run_dir.glob(f"{phf}_*.json"))
        meta["snapshot_count"] = len(snapshots)
        if history_path.exists():
            try:
                hist = json.loads(history_path.read_text(encoding="utf-8"))
                meta["last_updated"] = hist.get("last_updated", "")
                meta["total_seen"] = len(hist.get("all_seen_ids", []))
                meta["analyzed"] = len(hist.get("analysis_cache", {}))
            except Exception:
                pass
        if snapshots:
            # Get program count from latest snapshot
            try:
                latest = json.loads(snapshots[-1].read_text(encoding="utf-8"))
                meta["programs"] = latest.get("total_keys", 0)
                meta["timestamp"] = latest.get("timestamp", "")
            except Exception:
                pass
        runs.append(meta)
    return runs


class AnalysisManager:
    """Background analysis state, runs in asyncio."""

    def __init__(self) -> None:
        self.results: dict[str, dict[str, Any]] = {}
        self.last_updated: str = ""
        self.tick_started: str = ""
        self.tick_finished: str = ""
        self.running: bool = False
        self.run_id: str = ""
        self.phf: str = ""
        self._task: Optional[asyncio.Task] = None
        self._client = None
        self._model = "openai/gpt-oss-120b"
        self._all_seen_ids: set[str] = set()
        self._analysis_cache: dict[str, list] = {}

    @property
    def run_dir(self) -> Path:
        return SNAPSHOTS_DIR / self.phf / self.run_id

    def start(self, phf: str, run_id: str, redis_port: int = 6379,
              redis_db: int = 0, interval: int = 300,
              model: str = "openai/gpt-oss-120b") -> None:
        if self.running:
            return
        self.phf = phf
        self.run_id = run_id
        self.results = {}
        self.last_updated = ""
        self._all_seen_ids = set()
        self._analysis_cache = {}
        self._model = model
        self.running = True

        # Load prior history if resuming
        self._load_history()

        self._task = asyncio.create_task(
            self._loop(phf, redis_port, redis_db, interval)
        )

    async def final_tick(self, phf: str, redis_port: int = 6379, redis_db: int = 0) -> None:
        """Run one last analysis+snapshot cycle (called after evolution ends)."""
        import redis as redis_lib
        try:
            # Ensure LLM client is ready
            if not self._client:
                api_key = _load_api_key()
                if api_key:
                    from openai import OpenAI
                    self._client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            r = redis_lib.Redis(port=redis_port, db=redis_db,
                                decode_responses=True, socket_connect_timeout=2)
            await self._tick(r, phf)
            r.close()
        except Exception:
            pass

    def stop(self) -> None:
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        self._client = None

    def get_state(self) -> dict[str, Any]:
        programs = []
        for pid, data in sorted(
            self.results.items(),
            key=lambda x: x[1].get("metrics", {}).get("efficiency", -1e9),
            reverse=True,
        ):
            programs.append({
                "id": pid,
                "id_short": pid[:8],
                "metrics": data.get("metrics", {}),
                "top3": data.get("top3"),
            })
        return {
            "programs": programs,
            "last_updated": self.last_updated,
            "tick_started": self.tick_started,
            "tick_finished": self.tick_finished,
            "active_count": len(self.results),
            "total_ever": len(self._all_seen_ids),
            "running": self.running,
            "run_id": self.run_id,
            "phf": self.phf,
        }

    def _load_history(self) -> None:
        history_path = self.run_dir / "_history.json"
        if history_path.exists():
            try:
                hist = json.loads(history_path.read_text(encoding="utf-8"))
                self._all_seen_ids = set(hist.get("all_seen_ids", []))
                self._analysis_cache = hist.get("analysis_cache", {})
            except Exception:
                pass

    def _save_history(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        history_path = self.run_dir / "_history.json"
        history_path.write_text(json.dumps({
            "all_seen_ids": list(self._all_seen_ids),
            "analysis_cache": self._analysis_cache,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False), encoding="utf-8")

    async def _loop(self, phf: str, redis_port: int, redis_db: int, interval: int) -> None:
        import redis as redis_lib

        api_key = _load_api_key()
        if api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

        try:
            while self.running:
                try:
                    r = redis_lib.Redis(port=redis_port, db=redis_db,
                                        decode_responses=True, socket_connect_timeout=2)
                    await self._tick(r, phf)
                    r.close()
                except Exception:
                    pass

                for _ in range(interval):
                    if not self.running:
                        break
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self._save_history()

    async def _tick(self, r, phf: str) -> None:
        self.tick_started = time.strftime("%H:%M:%S")
        self.tick_finished = ""
        programs = self._fetch_done(r, phf)
        current_ids = set(programs.keys())
        self._all_seen_ids |= current_ids

        new_results: dict[str, dict[str, Any]] = {}
        for pid, prog in programs.items():
            cached_top3 = self._analysis_cache.get(pid)
            new_results[pid] = {
                "metrics": prog["metrics"],
                "top3": cached_top3,
            }

        # Find programs needing analysis
        to_analyze = []
        if self._client:
            for pid in current_ids:
                cached = self._analysis_cache.get(pid)
                if cached is None or (cached and cached[0][0] == "error"):
                    to_analyze.append(pid)

        if to_analyze:
            loop = asyncio.get_event_loop()
            for pid in to_analyze:
                prog = programs.get(pid)
                if not prog:
                    continue
                top3 = await loop.run_in_executor(
                    None, self._analyze_one, prog["code"]
                )
                if not (top3 and top3[0][0] == "error"):
                    self._analysis_cache[pid] = top3
                    new_results[pid]["top3"] = top3

        self.results = new_results
        self.tick_finished = time.strftime("%H:%M:%S")
        self.last_updated = self.tick_finished

        # Save snapshot + history
        self._save_snapshot(r, phf)
        self._save_history()

    def _fetch_done(self, r, phf: str) -> dict[str, dict]:
        programs = {}
        for key in r.scan_iter(f"{phf}:program:*", count=1000):
            raw = r.get(key)
            if not raw:
                continue
            try:
                prog = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if prog.get("state") == "done" and prog.get("code") and prog.get("metrics"):
                pid = prog.get("id", key)
                programs[pid] = {
                    "code": prog["code"],
                    "metrics": prog["metrics"],
                }
        return programs

    def _analyze_one(self, code: str) -> list:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": _make_prompt(code)}],
                temperature=0.1,
                max_tokens=4096,
            )
            content = resp.choices[0].message.content
            if not content:
                return [["error", "empty response"]]
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
            return json.loads(content).get("top3", [])
        except Exception as e:
            return [["error", str(e)[:80]]]

    def _save_snapshot(self, r, phf: str) -> None:
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            programs = {}
            for key in r.scan_iter(f"{phf}:program:*", count=1000):
                raw = r.get(key)
                if raw:
                    try:
                        programs[key] = json.loads(raw)
                    except json.JSONDecodeError:
                        programs[key] = {"_raw": raw}
            snapshot = {
                "phf": phf,
                "run_id": self.run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_keys": len(programs),
                "programs": programs,
            }
            out_path = self.run_dir / f"{phf}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            out_path.write_text(
                json.dumps(snapshot, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass


analysis_manager = AnalysisManager()