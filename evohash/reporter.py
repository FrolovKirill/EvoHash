"""EvoHash W&B reporter.

Logs metrics and image grids to Weights & Biases after each validate() call.

W&B run is initialised once in problems/*/context.py (build_context).
validate.py calls log_iteration() which just does wandb.log().

Run structure in wandb:
  project : evohash
  group   : phash | pdq | neuralhash     (filter by PHF)
  name    : phash 2026-03-15 14:23       (human-readable)
  tags    : [phf_name]
  config  : phf, threshold, n_pairs_eval
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Thumbnail size for each cell in the image grid.
CELL = 128
# How many image pairs to show per iteration.
N_SHOW = 3
# Pixel gap between cells.
GAP = 6
# Height of the label strip below each image.
LABEL_H = 18


# ── Public API ────────────────────────────────────────────────────────────────

def _ensure_wandb_key() -> None:
    """Load WANDB_API_KEY from .env if not already in the environment."""
    import os
    if os.environ.get("WANDB_API_KEY"):
        return
    try:
        import evohash
        from pathlib import Path
        dotenv_path = Path(evohash.__file__).parent.parent / ".env"
        if not dotenv_path.exists():
            return
        with dotenv_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


def init_run(phf_name: str, threshold: int, n_pairs: int) -> str:
    """Initialise a W&B run.  Returns the run id.

    Call once from build_context().  Safe to call multiple times — subsequent
    calls are no-ops if a run is already active.
    """
    import time
    import wandb

    if wandb.run is not None:
        return wandb.run.id

    _ensure_wandb_key()

    import os
    ts = time.strftime("%Y-%m-%d %H:%M")
    project = os.environ.get("WANDB_PROJECT", "evohash")
    mode = "disabled" if not os.environ.get("WANDB_API_KEY") else None
    run = wandb.init(
        project=project,
        group=phf_name,
        name=f"{phf_name} {ts}",
        tags=[phf_name],
        config={
            "phf": phf_name,
            "threshold": threshold,
            "n_pairs_eval": n_pairs,
        },
        resume="allow",  # reconnect to WANDB_RUN_ID if already set by run_evohash.py
        mode=mode,
    )
    return run.id


def log_initial_programs(program_dir: Any) -> None:
    """Log all *.py files in program_dir/initial_programs/ as a W&B artifact.

    Call once from build_context() so all seed attack files are visible in W&B
    under the Artifacts tab.  Never raises.
    """
    try:
        import wandb
        from pathlib import Path

        if wandb.run is None:
            return

        initial_dir = Path(program_dir) / "initial_programs"
        if not initial_dir.exists():
            return

        py_files = sorted(initial_dir.glob("*.py"))
        if not py_files:
            return

        artifact = wandb.Artifact(
            name=f"initial_programs_{wandb.run.id}",
            type="code",
            description="Seed attack programs for EvoHash evolution",
        )
        for f in py_files:
            artifact.add_file(str(f), name=f.name)
        wandb.run.log_artifact(artifact)
    except Exception:
        pass


def log_iteration(
    report_dir: Any,          # kept for API compat, ignored
    phf_name: str,
    metrics: dict[str, float],
    context: dict,
    data: dict,
) -> None:
    """Log one iteration to W&B.  Never raises."""
    try:
        _log(phf_name, metrics, context, data)
    except Exception:
        pass


# ── Internal ──────────────────────────────────────────────────────────────────

def _log(phf_name: str, metrics: dict, context: dict, data: dict) -> None:
    import wandb

    if wandb.run is None:
        # Fallback: init a fresh run if context.py somehow didn't.
        init_run(phf_name, context.get("threshold", 0), len(context.get("source_images", [])))

    sources      = context.get("source_images", [])
    targets      = context.get("target_images", [])
    attacked_all = (data or {}).get("attacked_images", [])
    hash_fn      = context.get("hash_fn")
    t_hashes     = context.get("target_hashes", [])
    threshold    = context.get("threshold", 0)

    # ── Scalar metrics ───────────────────────────────────────────────────────
    log: dict[str, Any] = {
        "efficiency": metrics.get("efficiency", 0),
        "asr":        metrics.get("asr", 0),
        "l2":         metrics.get("l2", 0),
        "n_queries":  metrics.get("n_queries", 0),
    }

    # ── Program code panel ───────────────────────────────────────────────────
    program_code = (data or {}).get("program_code")
    if program_code:
        html = (
            "<html><body style='background:#0d1117;color:#c9d1d9'>"
            f"<pre style='font-size:12px;padding:16px'>{program_code}</pre>"
            "</body></html>"
        )
        log["program_code"] = wandb.Html(html)

    # ── Per-pair distances table ─────────────────────────────────────────────
    n_pairs = len(sources)
    if attacked_all and hash_fn:
        rows = []
        for i in range(min(n_pairs, len(attacked_all))):
            atk = _ensure_pil(attacked_all[i])
            try:
                dist = int(hash_fn.distance(hash_fn.compute(atk), t_hashes[i]))
                rows.append([i, dist, threshold, bool(dist <= threshold)])
            except Exception:
                rows.append([i, None, threshold, False])

        tbl = wandb.Table(columns=["pair", "hash_dist", "threshold", "success"])
        for row in rows:
            tbl.add_data(*row)
        log["pair_details"] = tbl

    # ── Image grid ───────────────────────────────────────────────────────────
    n_show = min(N_SHOW, len(sources), len(attacked_all) if attacked_all else 0)
    if n_show > 0:
        pair_info = _compute_pair_info(
            sources[:n_show], targets[:n_show], attacked_all[:n_show],
            hash_fn, t_hashes[:n_show], threshold,
        )
        grid = _make_grid(sources[:n_show], targets[:n_show], attacked_all[:n_show], pair_info)
        asr  = metrics.get("asr", 0)
        eff  = metrics.get("efficiency", 0)
        log["examples"] = wandb.Image(
            grid,
            caption=f"ASR={asr:.1%}  eff={eff:.4f}  threshold={threshold}",
        )

    wandb.log(log)

    # ── Save grid to disk for web UI ──────────────────────────────────────────
    if n_show > 0 and grid is not None:
        try:
            from pathlib import Path
            grid_dir = Path(__file__).parent.parent / "web" / "backend" / "static" / "grids"
            grid_dir.mkdir(parents=True, exist_ok=True)
            grid.save(str(grid_dir / f"{phf_name}_latest.png"))
            # Save best grid — track best efficiency in a file (survives subprocesses)
            eff = metrics.get("efficiency", -1e9)
            best_eff_file = grid_dir / f"{phf_name}_best_eff.txt"
            prev_best = -1e9
            try:
                if best_eff_file.exists():
                    prev_best = float(best_eff_file.read_text().strip())
            except Exception:
                pass
            if eff > prev_best:
                best_eff_file.write_text(str(eff))
                grid.save(str(grid_dir / f"{phf_name}_best.png"))
        except Exception:
            pass

    # ── Rolling best in run summary ──────────────────────────────────────────
    s = wandb.run.summary
    if metrics.get("efficiency", -1e9) > s.get("best_efficiency", -1e9):
        s["best_efficiency"] = metrics["efficiency"]
    if metrics.get("asr", 0.0) > s.get("best_asr", 0.0):
        s["best_asr"] = metrics["asr"]
    if 0 < metrics.get("l2", 1e9) < s.get("min_l2", 1e9):
        s["min_l2"] = metrics["l2"]



# ── Image helpers ─────────────────────────────────────────────────────────────

def _ensure_pil(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    arr = np.asarray(img)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _thumb(img: Image.Image) -> Image.Image:
    t = img.convert("RGB").copy()
    t.thumbnail((CELL, CELL), Image.LANCZOS)
    # Pad to exactly CELL×CELL
    out = Image.new("RGB", (CELL, CELL), (20, 20, 30))
    ox = (CELL - t.width) // 2
    oy = (CELL - t.height) // 2
    out.paste(t, (ox, oy))
    return out


def _diff_img(orig: Image.Image, attacked: Image.Image, amplify: float = 10.0) -> Image.Image:
    a = np.array(orig.convert("RGB")).astype(float)
    b = np.array(attacked.convert("RGB")).astype(float)
    diff = np.clip(np.abs(b - a) * amplify, 0, 255).astype(np.uint8)
    return Image.fromarray(diff)


def _compute_pair_info(
    sources, targets, attacked_all, hash_fn, t_hashes, threshold
) -> list[dict]:
    infos = []
    for i, (src, tgt, atk) in enumerate(zip(sources, targets, attacked_all)):
        atk_pil = _ensure_pil(atk)
        info: dict[str, Any] = {"dist": None, "success": False}
        if hash_fn and i < len(t_hashes):
            try:
                dist = int(hash_fn.distance(hash_fn.compute(atk_pil), t_hashes[i]))
                info["dist"]    = dist
                info["success"] = dist <= threshold
            except Exception:
                pass
        infos.append(info)
    return infos


def _add_label(cell: Image.Image, text: str, color: tuple) -> Image.Image:
    """Return a new image = cell + a coloured label strip at the bottom."""
    w, h = cell.size
    out = Image.new("RGB", (w, h + LABEL_H), (12, 12, 20))
    out.paste(cell, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # Centre the text in the label strip
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
    else:
        tw = len(text) * 6
    tx = max(0, (w - tw) // 2)
    draw.text((tx, h + 2), text, fill=color, font=font)
    return out


def _make_grid(sources, targets, attacked_all, pair_info: list[dict]) -> Image.Image:
    """Build an image grid: each row is one pair (source / target / attacked / diff)."""
    col_labels = ["Source", "Target", "Attacked", "Diff ×10"]
    n_cols = 4
    n_rows = len(sources)

    cell_h = CELL + LABEL_H
    total_w = n_cols * CELL + (n_cols - 1) * GAP
    total_h = n_rows * cell_h + (n_rows - 1) * GAP

    canvas = Image.new("RGB", (total_w, total_h), (8, 8, 15))

    for row, (src, tgt, atk, info) in enumerate(zip(sources, targets, attacked_all, pair_info)):
        src_pil = _ensure_pil(src)
        tgt_pil = _ensure_pil(tgt)
        atk_pil = _ensure_pil(atk)
        diff_pil = _diff_img(src_pil, atk_pil)

        success = info["success"]
        dist    = info["dist"]
        atk_label = (
            f"{'✓' if success else '✗'} dist={dist}"
            if dist is not None else "Attacked"
        )
        atk_color = (80, 220, 140) if success else (240, 100, 100)

        cells = [
            _add_label(_thumb(src_pil),  col_labels[0], (160, 160, 200)),
            _add_label(_thumb(tgt_pil),  col_labels[1], (160, 160, 200)),
            _add_label(_thumb(atk_pil),  atk_label,     atk_color),
            _add_label(_thumb(diff_pil), col_labels[3], (170, 130, 240)),
        ]

        # Optionally draw coloured border on attacked cell
        draw = ImageDraw.Draw(cells[2])
        border_color = (60, 200, 110) if success else (220, 80, 80)
        for bw in range(3):
            draw.rectangle([bw, bw, CELL - 1 - bw, cell_h - 1 - bw], outline=border_color)

        y = row * (cell_h + GAP)
        for col, cell in enumerate(cells):
            x = col * (CELL + GAP)
            canvas.paste(cell, (x, y))

    return canvas
