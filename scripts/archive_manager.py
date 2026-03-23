"""Dump / load the MAP-Elites archive from Redis to JSON files.

Usage:
    # Save current archive to a timestamped JSON
    python scripts/archive_manager.py save phash

    # Save to a specific file
    python scripts/archive_manager.py save phash --file snapshots/my_run.json

    # Auto-save every 5 minutes while evolution runs (Ctrl+C to stop)
    python scripts/archive_manager.py watch phash
    python scripts/archive_manager.py watch phash --interval 120

    # Load a snapshot back into Redis (merges by default)
    python scripts/archive_manager.py load phash --file snapshots/my_run.json

    # Load and overwrite (clears existing keys for this PHF first)
    python scripts/archive_manager.py load phash --file snapshots/my_run.json --overwrite

    # Show stats about a snapshot file (no Redis needed)
    python scripts/archive_manager.py info --file snapshots/my_run.json

    # Compare two snapshots (what appeared / changed / disappeared)
    python scripts/archive_manager.py diff --old snap1.json --new snap2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import redis


def save_archive(r: redis.Redis, phf: str, out_path: Path) -> None:
    """Dump all programs for a PHF from Redis to a JSON file."""
    pattern = f"{phf}:program:*"
    keys = list(r.scan_iter(pattern, count=1000))

    programs = {}
    for key in keys:
        raw = r.get(key)
        if raw:
            try:
                programs[key] = json.loads(raw)
            except json.JSONDecodeError:
                programs[key] = {"_raw": raw}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "phf": phf,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_keys": len(programs),
        "programs": programs,
    }
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    done = sum(1 for p in programs.values() if isinstance(p, dict) and p.get("state") == "done")
    print(f"Saved {len(programs)} keys ({done} done) -> {out_path}")


def load_archive(r: redis.Redis, phf: str, in_path: Path, overwrite: bool) -> None:
    """Load programs from a JSON snapshot back into Redis."""
    snapshot = json.loads(in_path.read_text(encoding="utf-8"))
    programs = snapshot.get("programs", {})

    if not programs:
        print("Snapshot is empty, nothing to load.")
        return

    if overwrite:
        existing = list(r.scan_iter(f"{phf}:program:*", count=1000))
        if existing:
            r.delete(*existing)
            print(f"Cleared {len(existing)} existing keys for '{phf}'")

    loaded = 0
    skipped = 0
    for key, prog in programs.items():
        if not overwrite and r.exists(key):
            skipped += 1
            continue
        value = prog if isinstance(prog, str) else json.dumps(prog, ensure_ascii=False)
        r.set(key, value)
        loaded += 1

    print(f"Loaded {loaded} keys into Redis (skipped {skipped} existing)")


def show_info(in_path: Path) -> None:
    """Print stats about a snapshot file without touching Redis."""
    snapshot = json.loads(in_path.read_text(encoding="utf-8"))
    programs = snapshot.get("programs", {})

    print(f"File:      {in_path}")
    print(f"PHF:       {snapshot.get('phf', '?')}")
    print(f"Saved at:  {snapshot.get('timestamp', '?')}")
    print(f"Total:     {len(programs)} programs")

    states = {}
    efficiencies = []
    for prog in programs.values():
        if not isinstance(prog, dict):
            continue
        st = prog.get("state", "unknown")
        states[st] = states.get(st, 0) + 1
        if st == "done" and prog.get("metrics"):
            eff = prog["metrics"].get("efficiency", -1000)
            if eff > -999:
                efficiencies.append(eff)

    print(f"States:    {states}")
    if efficiencies:
        efficiencies.sort(reverse=True)
        print(f"Done:      {len(efficiencies)} programs with metrics")
        print(f"Best eff:  {efficiencies[0]:.6f}")
        print(f"Top-5 eff: {[round(e, 6) for e in efficiencies[:5]]}")



# ---------------------------------------------------------------------------
# LLM pattern analysis (lightweight, for watch mode)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASELINE_NAMES = [
    "NES", "SimBa", "HSJA", "Prokos", "AtkScopes",
    "ZO-SignSGD", "RandomNoise",
]

def _make_analysis_prompt(code: str) -> str:
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


def _load_api_key() -> str | None:
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        for line in dotenv_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY=") and not line.endswith("..."):
                return line.split("=", 1)[1].strip()
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")


def _analyze_one(client, code: str, model: str) -> list[list]:
    """Ask LLM for top-3 baseline similarities. Returns [["Name", score], ...]."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _make_analysis_prompt(code)}],
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


def _fetch_done_programs(r: redis.Redis, phf: str) -> dict[str, dict]:
    """Fetch all done programs keyed by ID."""
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
                "key": key,
                "code": prog["code"],
                "metrics": prog["metrics"],
            }
    return programs


# ---------------------------------------------------------------------------
# Watch with analysis
# ---------------------------------------------------------------------------

def watch_archive(
    r: redis.Redis,
    phf: str,
    interval: int,
    snap_dir: Path,
    analyze: bool,
    model: str,
) -> None:
    """Periodically save snapshots + optionally run LLM analysis on new programs."""
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Persistent state across ticks
    prev_done_count = -1
    all_seen_ids: set[str] = set()          # every program ID ever seen
    analysis_cache: dict[str, list] = {}    # program_id -> top3 result
    history_path = snap_dir / "_history.json"

    # Load prior history if exists
    if history_path.exists():
        try:
            hist = json.loads(history_path.read_text(encoding="utf-8"))
            all_seen_ids = set(hist.get("all_seen_ids", []))
            analysis_cache = hist.get("analysis_cache", {})
            print(f"Loaded history: {len(all_seen_ids)} IDs seen, {len(analysis_cache)} analyzed")
        except Exception:
            pass

    # Init LLM client if analyzing
    client = None
    if analyze:
        api_key = _load_api_key()
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            print(f"LLM analysis enabled ({model})")
        else:
            print("WARNING: No API key found, analysis disabled.")
            analyze = False

    print(f"Watching '{phf}' every {interval}s -> {snap_dir}/  (Ctrl+C to stop)\n")

    try:
        while True:
            programs = _fetch_done_programs(r, phf)
            done_count = len(programs)
            current_ids = set(programs.keys())

            # Track history
            new_ids = current_ids - all_seen_ids
            all_seen_ids |= current_ids

            # Save snapshot if changed
            if done_count != prev_done_count:
                out = snap_dir / f"{phf}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                save_archive(r, phf, out)
                prev_done_count = done_count

            # Analyze programs: new, missing from cache, or previously failed
            if analyze and client:
                to_analyze = set()
                # Programs not yet in cache (new or cleared)
                for pid in current_ids:
                    if pid not in analysis_cache:
                        to_analyze.add(pid)
                # Previously failed analyses
                for pid, result in analysis_cache.items():
                    if result and result[0][0] == "error" and pid in current_ids:
                        to_analyze.add(pid)
                # Remove cached errors so they get retried
                for pid in to_analyze:
                    analysis_cache.pop(pid, None)

                if to_analyze:
                    print(f"\n  Analyzing {len(to_analyze)} program(s)...")
                    for pid in to_analyze:
                        prog = programs.get(pid)
                        if not prog:
                            continue
                        top3 = _analyze_one(client, prog["code"], model)
                        # Only cache successful results
                        if not (top3 and top3[0][0] == "error"):
                            analysis_cache[pid] = top3
                        else:
                            print(f"    {pid[:8]}: {top3[0][1]}")
                        time.sleep(0.5)

            # Print summary table
            _print_watch_summary(programs, analysis_cache, all_seen_ids)

            # Persist history
            history_path.write_text(json.dumps({
                "all_seen_ids": list(all_seen_ids),
                "analysis_cache": analysis_cache,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, ensure_ascii=False), encoding="utf-8")

            time.sleep(interval)

    except KeyboardInterrupt:
        # Final save
        history_path.write_text(json.dumps({
            "all_seen_ids": list(all_seen_ids),
            "analysis_cache": analysis_cache,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False), encoding="utf-8")
        print(f"\nStopped. History: {len(all_seen_ids)} total IDs seen, "
              f"{len(analysis_cache)} analyzed. Saved to {history_path}")


def _print_watch_summary(
    programs: dict[str, dict],
    analysis_cache: dict[str, list],
    all_seen_ids: set[str],
) -> None:
    """Print compact table: each cell's metrics + top-3 baselines."""
    now = time.strftime("%H:%M:%S")
    current = len(programs)
    total_ever = len(all_seen_ids)
    replaced = total_ever - current

    print(f"\n{'='*80}")
    print(f"  [{now}]  Cells: {current} active | {total_ever} total ever | {replaced} replaced")
    print(f"{'='*80}")

    # Sort by efficiency descending
    sorted_progs = sorted(
        programs.items(),
        key=lambda x: x[1]["metrics"].get("efficiency", -1e9),
        reverse=True,
    )

    for rank, (pid, prog) in enumerate(sorted_progs, 1):
        m = prog["metrics"]
        eff = m.get("efficiency", 0)
        asr = m.get("asr", 0)
        l2 = m.get("l2", 0)

        top3 = analysis_cache.get(pid)
        if top3 and not (len(top3) == 1 and top3[0][0] == "error"):
            patterns_str = "  ".join(
                f"{name}({score:.0%})" for name, score in top3[:3]
            )
        elif top3 and top3[0][0] == "error":
            patterns_str = "[analysis error]"
        else:
            patterns_str = "[pending]"

        print(f"  #{rank:2d}  {pid[:8]}  eff={eff:9.6f}  asr={asr:.2f}  l2={l2:7.4f}  | {patterns_str}")

    print()


def diff_snapshots(old_path: Path, new_path: Path) -> None:
    """Compare two snapshot files: new programs, removed, changed metrics."""
    old = json.loads(old_path.read_text(encoding="utf-8")).get("programs", {})
    new = json.loads(new_path.read_text(encoding="utf-8")).get("programs", {})

    old_keys = set(old.keys())
    new_keys = set(new.keys())

    added = new_keys - old_keys
    removed = old_keys - new_keys
    common = old_keys & new_keys

    changed = []
    for k in common:
        old_m = old[k].get("metrics", {}) if isinstance(old[k], dict) else {}
        new_m = new[k].get("metrics", {}) if isinstance(new[k], dict) else {}
        if old_m.get("efficiency") != new_m.get("efficiency"):
            changed.append((k, old_m.get("efficiency", 0), new_m.get("efficiency", 0)))

    print(f"Old: {old_path.name}  ({len(old_keys)} keys)")
    print(f"New: {new_path.name}  ({len(new_keys)} keys)")
    print(f"\nAdded:   {len(added)}")
    print(f"Removed: {len(removed)}")
    print(f"Changed: {len(changed)}")

    if added:
        print(f"\n--- New programs ---")
        for k in sorted(added):
            p = new[k]
            if isinstance(p, dict) and p.get("metrics"):
                eff = p["metrics"].get("efficiency", 0)
                asr = p["metrics"].get("asr", 0)
                print(f"  {k[-12:]}  eff={eff:.6f}  asr={asr:.2f}")

    if changed:
        print(f"\n--- Changed efficiency ---")
        for k, old_e, new_e in sorted(changed, key=lambda x: x[2] - x[1], reverse=True)[:10]:
            delta = new_e - old_e
            print(f"  {k[-12:]}  {old_e:.6f} -> {new_e:.6f}  ({delta:+.6f})")


def main() -> None:
    p = argparse.ArgumentParser(description="Save/load MAP-Elites archive snapshots")
    sub = p.add_subparsers(dest="command", required=True)

    # --- save ---
    s_save = sub.add_parser("save", help="Dump Redis archive to JSON")
    s_save.add_argument("phf", help="PHF name (phash, pdq, neuralhash)")
    s_save.add_argument("--file", type=Path, default=None, help="Output file path")
    s_save.add_argument("--db", type=int, default=0)
    s_save.add_argument("--host", default="localhost")
    s_save.add_argument("--port", type=int, default=6379)

    # --- watch ---
    s_watch = sub.add_parser("watch", help="Auto-save snapshots + LLM analysis")
    s_watch.add_argument("phf", help="PHF name (phash, pdq, neuralhash)")
    s_watch.add_argument("--run", default=None,
                         help="Run ID (default: auto-generated run_YYYYMMDD_HHMMSS)")
    s_watch.add_argument("--interval", type=int, default=300, metavar="SEC",
                         help="Seconds between snapshots (default: 300 = 5min)")
    s_watch.add_argument("--dir", type=Path, default=None,
                         help="Snapshots directory (overrides --run)")
    s_watch.add_argument("--analyze", action="store_true",
                         help="Enable LLM pattern analysis on new programs")
    s_watch.add_argument("--model", default="openai/gpt-oss-120b",
                         help="OpenRouter model for analysis")
    s_watch.add_argument("--db", type=int, default=0)
    s_watch.add_argument("--host", default="localhost")
    s_watch.add_argument("--port", type=int, default=6379)

    # --- load ---
    s_load = sub.add_parser("load", help="Load JSON snapshot into Redis")
    s_load.add_argument("phf", help="PHF name")
    s_load.add_argument("--file", type=Path, required=True, help="Snapshot file to load")
    s_load.add_argument("--overwrite", action="store_true", help="Clear existing keys first")
    s_load.add_argument("--db", type=int, default=0)
    s_load.add_argument("--host", default="localhost")
    s_load.add_argument("--port", type=int, default=6379)

    # --- info ---
    s_info = sub.add_parser("info", help="Show snapshot stats (no Redis needed)")
    s_info.add_argument("--file", type=Path, required=True)

    # --- diff ---
    s_diff = sub.add_parser("diff", help="Compare two snapshots")
    s_diff.add_argument("--old", type=Path, required=True)
    s_diff.add_argument("--new", type=Path, required=True)

    args = p.parse_args()

    if args.command == "info":
        show_info(args.file)
        return

    if args.command == "diff":
        diff_snapshots(args.old, args.new)
        return

    r = redis.Redis(host=args.host, port=args.port, db=args.db, decode_responses=True)

    if args.command == "save":
        out = args.file or Path("snapshots") / args.phf / f"{args.phf}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        save_archive(r, args.phf, out)

    elif args.command == "load":
        load_archive(r, args.phf, args.file, args.overwrite)

    elif args.command == "watch":
        if args.dir:
            snap_dir = args.dir
        else:
            run_id = args.run or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
            snap_dir = Path("snapshots") / args.phf / run_id
        watch_archive(r, args.phf, args.interval, snap_dir, args.analyze, args.model)


if __name__ == "__main__":
    main()
