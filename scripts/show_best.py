"""Dump the best evolved attack programs from Redis.

Usage:
    python scripts/show_best.py phash            # top-5 by efficiency
    python scripts/show_best.py pdq --top 10
    python scripts/show_best.py phash --metric asr
    python scripts/show_best.py phash --save     # save code to files
    python scripts/show_best.py phash --watch    # refresh every 30s, log to W&B
    python scripts/show_best.py phash --watch 10 # refresh every 10s
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import redis


def _init_wandb(phf: str):
    """Init a W&B run for live monitoring. Returns wandb module or None."""
    try:
        import wandb
        from evohash.reporter import _ensure_wandb_key
        _ensure_wandb_key()
        if wandb.run is None:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "evohash"),
                group=phf,
                name=f"{phf} monitor {time.strftime('%Y-%m-%d %H:%M')}",
                tags=[phf, "monitor"],
                config={"phf": phf, "script": "show_best"},
            )
        return wandb
    except Exception as e:
        print(f"W&B init failed: {e}")
        return None


def _log_to_wandb(wandb, programs: list[dict], metric: str) -> None:
    """Log best program metrics + code to W&B."""
    if not programs or wandb is None or wandb.run is None:
        return

    best = programs[0]
    m = best["metrics"]

    code = best["code"]
    html = (
        "<html><body style='background:#0d1117;color:#c9d1d9'>"
        f"<pre style='font-size:12px;padding:16px;white-space:pre-wrap'>"
        f"{code}"
        f"</pre></body></html>"
    )

    log: dict = {
        "monitor/total_evaluated": len(programs),
        "monitor/best_efficiency": m.get("efficiency", 0),
        "monitor/best_asr":        m.get("asr", 0),
        "monitor/best_l2":         m.get("l2", 0),
        "monitor/best_n_queries":  m.get("n_queries", 0),
        "monitor/best_program":    wandb.Html(html),
    }

    # Top-3 as a table
    tbl = wandb.Table(columns=["rank", "id", "efficiency", "asr", "l2", "n_queries"])
    for rank, prog in enumerate(programs[:3], 1):
        pm = prog["metrics"]
        tbl.add_data(
            rank,
            prog["id"][:8],
            round(pm.get("efficiency", 0), 6),
            round(pm.get("asr", 0), 3),
            round(pm.get("l2", 0), 4),
            int(pm.get("n_queries", 0)),
        )
    log["monitor/top3"] = tbl

    wandb.log(log)


def fetch_programs(r: redis.Redis, phf: str) -> list[dict]:
    """Fetch and return all completed programs from Redis, unsorted."""
    pattern = f"{phf}:program:*"
    keys = list(r.scan_iter(pattern, count=1000))
    programs = []
    for key in keys:
        raw = r.get(key)
        if not raw:
            continue
        try:
            prog = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if prog.get("state") != "done" or not prog.get("code") or not prog.get("metrics"):
            continue
        programs.append({
            "id":      prog.get("id", "?"),
            "metrics": prog["metrics"],
            "code":    prog["code"],
        })
    return programs


def print_top(programs: list[dict], top: int, metric: str) -> None:
    sep = "─" * 72
    print(f"\nTotal evaluated: {len(programs)}  |  top-{min(top, len(programs))} by {metric}"
          f"  |  {time.strftime('%H:%M:%S')}")
    for rank, prog in enumerate(programs[:top], 1):
        m = prog["metrics"]
        print(f"\n{sep}")
        print(f"  #{rank}  id={prog['id'][:8]}…  "
              f"efficiency={m.get('efficiency', 0):.6f}  "
              f"asr={m.get('asr', 0):.2f}  "
              f"l2={m.get('l2', 0):.4f}  "
              f"queries={m.get('n_queries', 0):.0f}")
        print(sep)
        print(prog["code"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("phf", help="PHF name (phash, pdq, neuralhash)")
    p.add_argument("--top",    type=int, default=5,            help="How many programs to show")
    p.add_argument("--metric", default="efficiency",           help="Sort metric (default: efficiency)")
    p.add_argument("--db",     type=int, default=0,            help="Redis DB number")
    p.add_argument("--host",   default="localhost")
    p.add_argument("--port",   type=int, default=6379)
    p.add_argument("--save",   action="store_true",            help="Save code to out/best_{phf}/")
    p.add_argument("--watch",  nargs="?", const=30, type=int,  metavar="SECONDS",
                   help="Refresh every N seconds and log to W&B (default 30s)")
    p.add_argument("--no-wandb", action="store_true",          help="Disable W&B logging in --watch mode")
    args = p.parse_args()

    r = redis.Redis(host=args.host, port=args.port, db=args.db, decode_responses=True)
    reverse = args.metric not in ("l2",)

    if args.watch:
        interval = args.watch
        wb = None if args.no_wandb else _init_wandb(args.phf)
        print(f"Watching '{args.phf}' every {interval}s"
              + (" → W&B" if wb else "") + "  (Ctrl+C to stop)")
        try:
            while True:
                os.system("clear" if os.name == "posix" else "cls")
                programs = fetch_programs(r, args.phf)
                if programs:
                    programs.sort(key=lambda x: x["metrics"].get(args.metric, -1e9), reverse=reverse)
                    print_top(programs, args.top, args.metric)
                    _log_to_wandb(wb, programs, args.metric)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No completed programs yet…")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        programs = fetch_programs(r, args.phf)
        if not programs:
            print(f"No programs found for '{args.phf}' in Redis db={args.db}")
            sys.exit(1)
        programs.sort(key=lambda x: x["metrics"].get(args.metric, -1e9), reverse=reverse)
        print_top(programs, args.top, args.metric)

        if args.save:
            out_dir = Path("out") / f"best_{args.phf}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for rank, prog in enumerate(programs[: args.top], 1):
                fname = out_dir / f"rank{rank:02d}_{prog['id'][:8]}.py"
                fname.write_text(prog["code"])
            print(f"\nSaved {args.top} files to {out_dir}/")


if __name__ == "__main__":
    main()
