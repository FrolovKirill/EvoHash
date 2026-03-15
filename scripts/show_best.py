"""Dump the best evolved attack programs from Redis.

Usage:
    python scripts/show_best.py phash            # top-5 by efficiency
    python scripts/show_best.py pdq --top 10
    python scripts/show_best.py phash --metric asr
    python scripts/show_best.py phash --save     # save code to files
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import redis


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("phf", help="PHF name used as Redis key prefix (phash, pdq, neuralhash)")
    p.add_argument("--top",    type=int, default=5,          help="How many programs to show")
    p.add_argument("--metric", default="efficiency",         help="Sort metric (default: efficiency)")
    p.add_argument("--db",     type=int, default=0,          help="Redis DB number")
    p.add_argument("--host",   default="localhost")
    p.add_argument("--port",   type=int, default=6379)
    p.add_argument("--save",   action="store_true",          help="Save code to out/best_{phf}/")
    args = p.parse_args()

    r = redis.Redis(host=args.host, port=args.port, db=args.db, decode_responses=True)

    # Scan all program keys for this PHF
    pattern = f"{args.phf}:program:*"
    keys = list(r.scan_iter(pattern, count=1000))
    if not keys:
        print(f"No programs found for prefix '{args.phf}' in Redis db={args.db}")
        sys.exit(1)

    print(f"Found {len(keys)} programs in Redis (prefix={args.phf}, db={args.db})")

    # Load and parse
    programs = []
    for key in keys:
        raw = r.get(key)
        if not raw:
            continue
        try:
            prog = json.loads(raw)
        except json.JSONDecodeError:
            continue

        metrics = prog.get("metrics", {})
        state   = prog.get("state", "")
        code    = prog.get("code", "")

        if state != "done" or not code or not metrics:
            continue

        programs.append({
            "id":      prog.get("id", "?"),
            "metrics": metrics,
            "code":    code,
            "state":   state,
        })

    if not programs:
        print("No completed programs with metrics found.")
        sys.exit(1)

    # Sort
    reverse = args.metric not in ("l2",)  # lower is better for l2
    programs.sort(key=lambda x: x["metrics"].get(args.metric, -1e9), reverse=reverse)
    top = programs[: args.top]

    # Print
    sep = "─" * 72
    for rank, prog in enumerate(top, 1):
        m = prog["metrics"]
        print(f"\n{sep}")
        print(f"  #{rank}  id={prog['id'][:8]}…  "
              f"efficiency={m.get('efficiency', 0):.6f}  "
              f"asr={m.get('asr', 0):.2f}  "
              f"l2={m.get('l2', 0):.4f}  "
              f"queries={m.get('n_queries', 0):.0f}")
        print(sep)
        print(prog["code"])

    # Optionally save to files
    if args.save:
        out_dir = Path("out") / f"best_{args.phf}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for rank, prog in enumerate(top, 1):
            fname = out_dir / f"rank{rank:02d}_{prog['id'][:8]}.py"
            fname.write_text(prog["code"])
        print(f"\nSaved {len(top)} files to {out_dir}/")


if __name__ == "__main__":
    main()
