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

    def _has_data(self) -> bool:
        data_dir = PROJECT_ROOT / "data" / "imagenet_val"
        if not data_dir.exists():
            return False
        images = list(data_dir.glob("*.JPEG")) + list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
        return len(images) >= 10

    async def _stream_process(self, proc: subprocess.Popen) -> None:
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, proc.stdout.readline)
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                self._emit(decoded)
                # Parse generation progress
                if "generation" in decoded.lower() or "iter" in decoded.lower():
                    try:
                        for token in decoded.split():
                            if token.isdigit():
                                n = int(token)
                                if n <= self.total_generations:
                                    self.generations_done = n
                                    break
                    except Exception:
                        pass

        await loop.run_in_executor(None, proc.wait)

    async def start(self, phf: str, generations: int, llm_config: str, n_pairs: int, redis_port: int = 6380) -> None:
        if self.status == Status.RUNNING:
            return

        self.status = Status.IDLE
        self.log_lines = []
        self.current_phf = phf
        self.total_generations = generations
        self.generations_done = 0
        self.error_message = ""

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + str(PROJECT_ROOT / "gigaevo-core")
        env["EVOHASH_REDIS_PORT"] = str(redis_port)

        # Step 1: download data if needed
        if not self._has_data():
            self.status = Status.DOWNLOADING
            self._emit("[web] Датасет не найден, скачиваю синтетические изображения...")
            dl_proc = subprocess.Popen(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "download_dataset.py"), "--synthetic", "--n-images", "100"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
            loop = asyncio.get_event_loop()
            await self._stream_process(dl_proc)
            if dl_proc.returncode and dl_proc.returncode != 0:
                self.status = Status.ERROR
                self.error_message = "Ошибка скачивания датасета"
                return

        # Step 2: run evohash
        self.status = Status.RUNNING
        self._emit(f"[web] Запускаю эволюцию для {phf}, {generations} поколений...")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_evohash.py"),
            "--phf", phf,
            "--generations", str(generations),
            "--llm", llm_config,
            "--n-pairs", str(n_pairs),
            "--redis-port", str(redis_port),
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
