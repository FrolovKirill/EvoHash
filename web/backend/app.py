"""EvoHash Web UI — FastAPI backend."""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from redis_bridge import get_best_metrics, get_latest_metrics, get_programs, reset_client, build_metrics_history
from runner import Status, runner
from snapshot_manager import (snapshot_manager, generate_run_id, list_runs,
                              load_redis_snapshot, restore_redis_from_snapshot,
                              SNAPSHOTS_DIR)

PROJECT_ROOT = Path(__file__).parent.parent.parent
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

app = FastAPI(title="EvoHash Web UI")


@app.on_event("shutdown")
def shutdown_event():
    """Kill any running evolution subprocess when the server stops."""
    runner.stop()
    snapshot_manager.stop()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve built frontend if available
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


# ─── Models ────────────────────────────────────────────────────────────────────

class RunConfig(BaseModel):
    phf: str = "phash"
    generations: int = 50
    llm_config: str = "openrouter_gpt_oss"
    n_pairs: int = 10
    redis_port: int = 6379
    redis_db: int = 0


class ResumeConfig(BaseModel):
    phf: str
    run_id: str
    generations: int = 50
    llm_config: str = "openrouter_gpt_oss"
    n_pairs: int = 10
    redis_port: int = 6379
    redis_db: int = 0


# ─── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/models")
async def get_models():
    import yaml
    models_file = PROJECT_ROOT / "config" / "models.yaml"
    if not models_file.exists():
        return {"models": []}
    with open(models_file) as f:
        data = yaml.safe_load(f)
    return {"models": data.get("models", [])}


@app.get("/api/status")
async def get_status():
    return {
        "status": runner.status,
        "phf": runner.current_phf,
        "generations_done": runner.generations_done,
        "total_generations": runner.total_generations,
        "error": runner.error_message,
    }


@app.post("/api/run")
async def start_run(config: RunConfig):
    if runner.status == Status.RUNNING:
        return {"ok": False, "message": "Уже запущено"}

    run_id = generate_run_id()

    async def _start():
        await runner.start(
            phf=config.phf,
            generations=config.generations,
            llm_config=config.llm_config,
            n_pairs=config.n_pairs,
            redis_port=config.redis_port,
            redis_db=config.redis_db,
        )
        snapshot_manager.stop()

    snapshot_manager.start(config.phf, run_id, config.redis_port, config.redis_db)
    asyncio.create_task(_start())
    return {"ok": True, "run_id": run_id, "message": f"Запускаю {config.phf} (новый run: {run_id})..."}


@app.get("/api/runs")
async def get_runs(phf: str = "phash"):
    return {"runs": list_runs(phf)}


@app.post("/api/run-resume")
async def resume_run(config: ResumeConfig):
    """Resume evolution from a saved Redis snapshot, continuing into the same run folder."""
    if runner.status == Status.RUNNING:
        return {"ok": False, "message": "Уже запущено"}

    run_dir = SNAPSHOTS_DIR / config.phf / config.run_id
    if not run_dir.exists():
        return {"ok": False, "message": f"Run {config.run_id} не найден"}

    snap = load_redis_snapshot(run_dir)
    if snap is None:
        return {"ok": False, "message": f"Redis-снапшот для {config.run_id} не найден. Запустите новую эволюцию."}

    try:
        n = restore_redis_from_snapshot(snap, config.redis_port, config.redis_db)
    except Exception as e:
        return {"ok": False, "message": f"Ошибка восстановления Redis: {e}"}

    async def _resume():
        await runner.start(
            phf=config.phf,
            generations=config.generations,
            llm_config=config.llm_config,
            n_pairs=config.n_pairs,
            redis_port=config.redis_port,
            redis_db=config.redis_db,
            resume=True,
        )
        snapshot_manager.stop()

    snapshot_manager.start(config.phf, config.run_id, config.redis_port, config.redis_db)
    asyncio.create_task(_resume())
    return {"ok": True, "run_id": config.run_id,
            "message": f"Восстановлено {n} программ из {config.run_id}, возобновляю {config.phf}..."}


@app.post("/api/stop")
async def stop_run():
    runner.stop()
    snapshot_manager.stop()
    return {"ok": True}


@app.post("/api/clear-data")
async def clear_data(redis_port: int = 6379, redis_db: int = 0):
    if runner.status == Status.RUNNING:
        return {"ok": False, "message": "Сначала остановите процесс"}
    # Clear logs
    runner.log_lines = []
    runner.status = Status.IDLE
    runner.generations_done = 0
    runner.total_generations = 0
    runner.error_message = ""
    # Clear grid images
    try:
        grid_dir = Path(__file__).parent / "static" / "grids"
        if grid_dir.exists():
            for f in grid_dir.iterdir():
                f.unlink()
    except Exception:
        pass
    # Flush Redis DB
    try:
        import redis as redis_lib
        r = redis_lib.Redis(port=redis_port, db=redis_db, socket_connect_timeout=2)
        r.flushdb()
        r.close()
    except Exception:
        pass
    return {"ok": True}


@app.get("/api/logs")
async def get_logs(offset: int = 0):
    lines = runner.log_lines[offset:]
    return {"lines": lines, "total": len(runner.log_lines)}


@app.get("/api/programs")
async def get_programs_endpoint(phf: str = "phash", redis_port: int = 6379):
    reset_client()
    programs = get_programs(phf, redis_port)
    return {"programs": programs}


@app.get("/api/metrics")
async def get_metrics(phf: str = "phash", redis_port: int = 6379):
    reset_client()
    best = get_best_metrics(phf, redis_port)
    return {"metrics": best}


@app.get("/api/metrics-latest")
async def get_metrics_latest(phf: str = "phash", redis_port: int = 6379):
    reset_client()
    latest = get_latest_metrics(phf, redis_port)
    return {"metrics": latest}


@app.get("/api/metrics-history")
async def get_metrics_history_endpoint(phf: str = "phash", redis_port: int = 6379):
    reset_client()
    history = build_metrics_history(phf, redis_port)
    return {"history": history}


@app.get("/api/grid-image")
async def grid_image(phf: str = "phash", variant: str = "latest"):
    filename = f"{phf}_best.png" if variant == "best" else f"{phf}_latest.png"
    grid_path = Path(__file__).parent / "static" / "grids" / filename
    if not grid_path.exists():
        return {"error": "no image yet"}
    return FileResponse(str(grid_path), media_type="image/png", headers={"Cache-Control": "no-cache"})


@app.get("/api/baselines")
async def get_baselines():
    csv_path = PROJECT_ROOT / "results_baselines.csv"
    if not csv_path.exists():
        return {"rows": []}
    rows = []
    with open(csv_path) as f:
        lines = f.readlines()
    if not lines:
        return {"rows": []}
    headers = [h.strip() for h in lines[0].split(",")]
    for line in lines[1:]:
        vals = [v.strip() for v in line.split(",")]
        if len(vals) == len(headers):
            rows.append(dict(zip(headers, vals)))
    return {"rows": rows}


@app.post("/api/run-baselines")
async def run_baselines(phf: str = "phash"):
    if runner.status == Status.RUNNING:
        return {"ok": False, "message": "Сначала останови текущий запуск"}

    async def _run():
        runner.status = Status.EVALUATING
        runner.current_phf = phf
        runner.log_lines = []
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + str(PROJECT_ROOT / "gigaevo-core")
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate.py"),
             "--phf", phf, "--all-seeds", "--csv", str(PROJECT_ROOT / "results_baselines.csv")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        runner.process = proc
        await runner._stream_process(proc)
        runner.status = Status.DONE if proc.returncode == 0 else Status.ERROR

    asyncio.create_task(_run())
    return {"ok": True}


@app.get("/api/check-env")
async def check_env():
    checks = {}
    # Check gigaevo-core
    checks["gigaevo_core"] = (PROJECT_ROOT / "gigaevo-core" / "run.py").exists()
    # Check data
    data_dir = PROJECT_ROOT / "data" / "imagenet_val"
    images = []
    if data_dir.exists():
        images = list(data_dir.glob("*.JPEG")) + list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
    checks["dataset"] = len(images) >= 10
    checks["dataset_count"] = len(images)
    # Check .env
    env_file = PROJECT_ROOT / ".env"
    checks["env_file"] = env_file.exists()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"')
                    break
    checks["api_key"] = bool(api_key and api_key != "your_key_here")
    # Check Redis
    try:
        import redis as redis_lib
        r = redis_lib.Redis(port=6379, socket_connect_timeout=1)
        r.ping()
        checks["redis"] = True
    except Exception:
        checks["redis"] = False
    return checks


# ─── WebSocket for live logs ────────────────────────────────────────────────────

active_ws: list[WebSocket] = []


@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    await ws.accept()
    active_ws.append(ws)

    queue: asyncio.Queue = asyncio.Queue()

    def on_line(line: str):
        queue.put_nowait(line)

    runner.log_callbacks.append(on_line)
    last_gen = -1
    try:
        # Send existing logs
        if runner.log_lines:
            await ws.send_text(json.dumps({"type": "history", "lines": runner.log_lines}))

        while True:
            try:
                line = await asyncio.wait_for(queue.get(), timeout=1.0)
                await ws.send_text(json.dumps({"type": "line", "line": line}))
            except asyncio.TimeoutError:
                # Send heartbeat + status
                await ws.send_text(json.dumps({
                    "type": "status",
                    "status": runner.status,
                    "generations_done": runner.generations_done,
                    "total_generations": runner.total_generations,
                }))
                # Send metric update when generation advances
                if runner.status == Status.RUNNING and runner.generations_done > last_gen:
                    last_gen = runner.generations_done
                    try:
                        reset_client()
                        best = get_best_metrics(runner.current_phf or "phash")
                        if best:
                            await ws.send_text(json.dumps({
                                "type": "metric",
                                "generations_done": runner.generations_done,
                                "efficiency": best.get("efficiency", 0),
                                "asr": best.get("asr", 0),
                            }))
                    except Exception:
                        pass
    except WebSocketDisconnect:
        pass
    finally:
        runner.log_callbacks.remove(on_line)
        active_ws.remove(ws)


# ─── Serve frontend ─────────────────────────────────────────────────────────────

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    if FRONTEND_DIST.exists():
        index = FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>EvoHash</title>
<style>
body{font-family:monospace;background:#0f0f0f;color:#00ff88;display:flex;
     align-items:center;justify-content:center;height:100vh;margin:0}
.box{text-align:center;padding:2rem;border:1px solid #00ff8844;border-radius:8px}
h1{font-size:2rem;margin-bottom:1rem}
p{color:#888;margin:0.5rem 0}
code{background:#1a1a1a;padding:0.2rem 0.5rem;border-radius:4px;color:#ffd700}
</style>
</head>
<body>
<div class="box">
  <h1>⚡ EvoHash</h1>
  <p>Frontend ещё не собран.</p>
  <p>Запусти: <code>cd web/frontend && npm install && npm run build</code></p>
  <p>API работает: <a href="/docs" style="color:#00ff88">/docs</a></p>
</div>
</body>
</html>
""")
