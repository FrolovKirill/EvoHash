"""Subprocess management for EvoHash pipeline."""
import asyncio
import os
import signal
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Status(str, Enum):
    IDLE = "idle"
    DOWNLOADING = "downloading"
    RUNNING = "running"
    EVALUATING = "evaluating"
    DONE = "done"
    ERROR = "error"


class Runner:
    def __init__(self) -> None:
        self.status: Status = Status.IDLE
        self.process: Optional[subprocess.Popen] = None
        self.log_lines: list[str] = []
        self.log_callbacks: list = []
        self.current_phf: str = ""
        self.generations_done: int = 0
        self.total_generations: int = 0
        self.error_message: str = ""
        self._log_task: Optional[asyncio.Task] = None

    def _emit(self, line: str) -> None:
        self.log_lines.append(line)
        if len(self.log_lines) > 2000:
            self.log_lines = self.log_lines[-2000:]
        for cb in self.log_callbacks:
            try:
                cb(line)
            except Exception:
                pass

    @staticmethod
    def _count_images() -> int:
        data_dir = PROJECT_ROOT / "data" / "imagenet_val"
        if not data_dir.exists():
            return 0
        images = list(data_dir.glob("*.JPEG")) + list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
        return len(images)

    async def _stream_process(self, proc: subprocess.Popen) -> None:
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, proc.stdout.readline)
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                self._emit(decoded)
                # Parse generation progress from gigaevo log lines:
                #   "selecting elites (gen=N, ...)" — emitted each generation
                #   "Stop: max_generations=N" — final line
                if "selecting elites (gen=" in decoded:
                    try:
                        gen_str = decoded.split("gen=")[1].split(",")[0]
                        self.generations_done = int(gen_str) + 1  # gen is 0-based
                    except Exception:
                        pass
                elif "Stop: max_generations=" in decoded:
                    self.generations_done = self.total_generations

        await loop.run_in_executor(None, proc.wait)

    @staticmethod
    def _kill_port(port: int) -> None:
        """Kill any process listening on the given port (macOS/Linux)."""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True, text=True
            )
            for pid in result.stdout.strip().split():
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except Exception:
                    pass
        except Exception:
            pass

    async def start(self, phf: str, generations: int, llm_config: str,
                    n_pairs: int = 10, redis_port: int = 6379, redis_db: int = 0) -> None:
        if self.status == Status.RUNNING:
            return

        # Kill any orphaned proxy from a previous run
        self._kill_port(8100)

        self.status = Status.IDLE
        self.log_lines = []
        self.current_phf = phf
        self.total_generations = generations
        self.generations_done = 0
        self.error_message = ""

        # Flush the Redis DB so gigaevo loads initial programs fresh
        # (resume mode skips archive population — programs never enter the island)
        try:
            import redis as redis_lib
            r = redis_lib.Redis(port=redis_port, db=redis_db, socket_connect_timeout=2)
            r.flushdb()
            r.close()
        except Exception:
            pass

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + str(PROJECT_ROOT / "gigaevo-core")

        # Load .env so subprocess gets API keys (WANDB_API_KEY, OPENROUTER_API_KEY, etc.)
        dotenv_path = PROJECT_ROOT / ".env"
        if dotenv_path.exists():
            with open(dotenv_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in env:
                        env[key] = value

        env["EVOHASH_N_PAIRS"] = str(n_pairs)

        # Step 1: download/top-up images if needed (n_pairs pairs need 2*n_pairs images)
        needed = n_pairs * 2
        have = self._count_images()
        if have < needed:
            self.status = Status.DOWNLOADING
            want = max(needed, 100)  # download at least 100 to avoid re-downloading often
            self._emit(f"[web] Нужно {needed} изображений, есть {have} — докачиваю до {want}...")
            dl_proc = subprocess.Popen(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "download_dataset.py"),
                 "--synthetic", "--n-images", str(want)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
            await self._stream_process(dl_proc)
            if dl_proc.returncode and dl_proc.returncode != 0:
                self.status = Status.ERROR
                self.error_message = "Ошибка скачивания датасета"
                return

        # Step 2: run evohash
        self.status = Status.RUNNING
        self._emit(f"[web] Запускаю эволюцию для {phf}, {generations} поколений...")

        # Use the same interpreter that's running the web server
        python = sys.executable

        cmd = [
            python,
            str(PROJECT_ROOT / "run_evohash.py"),
            phf,
            "--max-generations", str(generations),
            "--llm", llm_config,
            "--redis-db", str(redis_db),
            "--", f"redis.port={redis_port}",
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        self._log_task = asyncio.create_task(self._finish(self.process))

    async def _finish(self, proc: subprocess.Popen) -> None:
        await self._stream_process(proc)
        if proc.returncode == 0:
            self.status = Status.DONE
            self._emit("[web] Эволюция завершена!")
        else:
            self.status = Status.ERROR
            self.error_message = f"Процесс завершился с кодом {proc.returncode}"
            self._emit(f"[web] Ошибка: {self.error_message}")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except Exception:
                self.process.terminate()
            self._emit("[web] Остановлено пользователем")
        self.status = Status.IDLE
        self.process = None


runner = Runner()
