#!/usr/bin/env python3
"""
EvoHash Web UI — single entry point.

Usage:
    python web/run_web.py [--port 8765] [--no-browser]
"""
import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent.parent
WEB_DIR = Path(__file__).parent
FRONTEND_DIST = WEB_DIR / "frontend" / "dist"
BACKEND_DIR = WEB_DIR / "backend"


def build_frontend_if_needed():
    if FRONTEND_DIST.exists() and any(FRONTEND_DIST.iterdir()):
        return  # already built
    print("[web] Building frontend...")
    npm = "npm"
    frontend_dir = WEB_DIR / "frontend"
    subprocess.run([npm, "install"], cwd=frontend_dir, check=True)
    subprocess.run([npm, "run", "build"], cwd=frontend_dir, check=True)
    print("[web] Frontend built.")


def check_backend_deps():
    req = BACKEND_DIR / "requirements.txt"
    try:
        import fastapi  # noqa
        import uvicorn  # noqa
    except ImportError:
        print("[web] Installing backend dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req)],
            check=True,
        )


def open_browser(port: int, delay: float = 1.5):
    time.sleep(delay)
    webbrowser.open(f"http://localhost:{port}")


def main():
    parser = argparse.ArgumentParser(description="EvoHash Web UI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--dev", action="store_true", help="Skip frontend build (use vite dev server)")
    args = parser.parse_args()

    os.chdir(ROOT)

    check_backend_deps()

    if not args.dev:
        build_frontend_if_needed()

    if not args.no_browser:
        t = threading.Thread(target=open_browser, args=(args.port,), daemon=True)
        t.start()

    print(f"[web] Starting EvoHash at http://localhost:{args.port}")

    import uvicorn  # noqa — imported after deps check

    # Add backend dir to path so app.py can import runner/redis_bridge
    sys.path.insert(0, str(BACKEND_DIR))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=args.port,
        reload=False,
        app_dir=str(BACKEND_DIR),
    )


if __name__ == "__main__":
    main()
