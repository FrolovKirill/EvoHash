"""
Task 1: Visualize best PDQ evolved attack (best L2, best LPIPS, median).
Task 2: Test transfer of best pHash/PDQ attacks across other hash functions.

Usage:
    python scripts/visualize_and_transfer.py
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evohash.dataset import load_image_pairs, image_to_array
from evohash.phf import get_phf
from evohash.evaluation import compute_lpips, compute_asr, compute_mean_l2, compute_efficiency

DATA_DIR = PROJECT_ROOT / "data" / "imagenet_val"
OUTPUT_DIR = PROJECT_ROOT / "results_visualization"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Grid drawing (replicating reporter.py UI style) ──────────────────────────

CELL = 160  # Slightly larger than reporter for better visibility
GAP = 6
LABEL_H = 22


def _thumb(img: Image.Image) -> Image.Image:
    t = img.convert("RGB").copy()
    t.thumbnail((CELL, CELL), Image.LANCZOS)
    out = Image.new("RGB", (CELL, CELL), (20, 20, 30))
    ox = (CELL - t.width) // 2
    oy = (CELL - t.height) // 2
    out.paste(t, (ox, oy))
    return out


def _diff_img(orig: Image.Image, attacked: Image.Image, amplify: float = 10.0) -> Image.Image:
    a = np.array(orig.convert("RGB").resize((CELL, CELL))).astype(float)
    b = np.array(attacked.convert("RGB").resize((CELL, CELL))).astype(float)
    diff = np.clip(np.abs(b - a) * amplify, 0, 255).astype(np.uint8)
    return Image.fromarray(diff)


def _add_label(cell: Image.Image, text: str, color: tuple) -> Image.Image:
    w, h = cell.size
    out = Image.new("RGB", (w, h + LABEL_H), (12, 12, 20))
    out.paste(cell, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
    else:
        tw = len(text) * 6
    tx = max(0, (w - tw) // 2)
    draw.text((tx, h + 3), text, fill=color, font=font)
    return out


def make_single_row(src: Image.Image, tgt: Image.Image, atk: Image.Image,
                    dist: int, success: bool, l2: float, lpips_val: float,
                    label_extra: str = "") -> Image.Image:
    """One row: Source | Target | Attacked | Diff x10."""
    cell_h = CELL + LABEL_H
    n_cols = 4
    total_w = n_cols * CELL + (n_cols - 1) * GAP
    canvas = Image.new("RGB", (total_w, cell_h), (8, 8, 15))

    diff_pil = _diff_img(src, atk)
    atk_label = f"{'✓' if success else '✗'} dist={dist} L2={l2:.1f}"
    if not np.isnan(lpips_val):
        atk_label += f" LPIPS={lpips_val:.3f}"
    atk_color = (80, 220, 140) if success else (240, 100, 100)

    cells = [
        _add_label(_thumb(src), "Source" + (f" {label_extra}" if label_extra else ""), (160, 160, 200)),
        _add_label(_thumb(tgt), "Target", (160, 160, 200)),
        _add_label(_thumb(atk), atk_label, atk_color),
        _add_label(_thumb(diff_pil), "Diff ×10", (170, 130, 240)),
    ]

    # Border on attacked cell
    draw = ImageDraw.Draw(cells[2])
    border_color = (60, 200, 110) if success else (220, 80, 80)
    for bw in range(3):
        draw.rectangle([bw, bw, CELL - 1 - bw, cell_h - 1 - bw], outline=border_color)

    for col, cell in enumerate(cells):
        x = col * (CELL + GAP)
        canvas.paste(cell, (x, 0))
    return canvas


def make_showcase_grid(rows: list[Image.Image], title: str = "") -> Image.Image:
    """Stack multiple row images vertically with optional title."""
    if not rows:
        return Image.new("RGB", (100, 100), (0, 0, 0))

    max_w = max(r.width for r in rows)
    title_h = 40 if title else 0
    total_h = title_h + sum(r.height for r in rows) + (len(rows) - 1) * GAP

    canvas = Image.new("RGB", (max_w, total_h), (8, 8, 15))

    if title:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((10, 10), title, fill=(220, 220, 255), font=font)

    y = title_h
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + GAP

    return canvas


# ── Program loader ────────────────────────────────────────────────────────────

def load_program(path: Path):
    spec = importlib.util.spec_from_file_location("attack_program", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.entrypoint


# ── Per-pair LPIPS ────────────────────────────────────────────────────────────

def compute_per_pair_lpips(originals: list[np.ndarray], attacked: list[np.ndarray]) -> list[float]:
    """Compute LPIPS for each pair individually."""
    try:
        import lpips as lpips_lib
        import torch
    except ImportError:
        return [float("nan")] * len(originals)

    loss_fn = lpips_lib.LPIPS(net="alex", verbose=False)
    scores = []
    for orig, atk in zip(originals, attacked):
        arr_o = orig.astype(np.float32) / 127.5 - 1.0
        arr_a = atk.astype(np.float32) / 127.5 - 1.0
        t_o = torch.from_numpy(arr_o).permute(2, 0, 1).unsqueeze(0)
        t_a = torch.from_numpy(arr_a).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            score = loss_fn(t_o, t_a).item()
        scores.append(score)
    return scores


# ── TASK 1: PDQ visualization ────────────────────────────────────────────────

def task1_pdq_visualization():
    print("=" * 70)
    print("TASK 1: PDQ Best Attack Visualization")
    print("=" * 70)

    best_pdq_path = PROJECT_ROOT / "snapshots" / "pdq" / "run_20260330_205147_last" / "bin_18" / "6fe64a77-0c19-41eb-acfd-bb75c216.py"
    print(f"Loading best PDQ program: {best_pdq_path.name}")
    attack_fn = load_program(best_pdq_path)

    phf = get_phf("pdq")
    n_pairs = 10
    pairs = load_image_pairs(DATA_DIR, n_pairs=n_pairs, seed=42)
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    target_hashes = [phf.compute(img) for img in targets]

    context = {
        "hash_fn": phf,
        "threshold": phf.threshold,
        "source_images": sources,
        "target_hashes": target_hashes,
        "target_images": targets,
    }

    print(f"Running attack on {n_pairs} pairs...")
    t0 = time.time()
    result = attack_fn(context)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    attacked_pil = result["attacked_images"]
    per_image = result["metrics"]

    # Compute per-pair metrics
    orig_arrs = [image_to_array(img) for img in sources]
    atk_arrs = [image_to_array(img) for img in attacked_pil]

    per_l2 = []
    for o, a in zip(orig_arrs, atk_arrs):
        diff = a.astype(float) - o.astype(float)
        per_l2.append(float(np.linalg.norm(diff.flatten()) / np.sqrt(o.size)))

    print("Computing per-pair LPIPS...")
    per_lpips = compute_per_pair_lpips(orig_arrs, atk_arrs)

    # Print all results
    print(f"\n{'Pair':>4} {'Success':>8} {'Dist':>6} {'L2':>8} {'LPIPS':>8} {'Queries':>8}")
    print("-" * 50)
    for i, (m, l2, lp) in enumerate(zip(per_image, per_l2, per_lpips)):
        print(f"{i:>4} {'✓' if m['success'] else '✗':>8} {m['final_dist']:>6.0f} {l2:>8.2f} {lp:>8.4f} {m['n_queries']:>8}")

    # Find best L2, best LPIPS, median
    successful = [(i, per_l2[i], per_lpips[i]) for i in range(n_pairs) if per_image[i]["success"]]

    if not successful:
        print("\nNo successful attacks! Cannot create visualization.")
        return

    # Best L2 (lowest)
    best_l2_idx = min(successful, key=lambda x: x[1])[0]
    # Best LPIPS (lowest)
    best_lpips_idx = min(successful, key=lambda x: x[2])[0]
    # Median by L2
    sorted_by_l2 = sorted(successful, key=lambda x: x[1])
    median_idx = sorted_by_l2[len(sorted_by_l2) // 2][0]

    # If best L2 and best LPIPS are the same, pick second-best LPIPS
    if best_lpips_idx == best_l2_idx and len(successful) > 1:
        sorted_by_lpips = sorted(successful, key=lambda x: x[2])
        best_lpips_idx = sorted_by_lpips[1][0]

    # If median is same as one of the above, shift it
    used = {best_l2_idx, best_lpips_idx}
    if median_idx in used:
        for entry in sorted_by_l2:
            if entry[0] not in used:
                median_idx = entry[0]
                break

    showcase_indices = [
        (best_l2_idx, "Best L2"),
        (best_lpips_idx, "Best LPIPS"),
        (median_idx, "Median"),
    ]

    print(f"\nShowcase pairs:")
    for idx, label in showcase_indices:
        print(f"  {label}: pair #{idx}, L2={per_l2[idx]:.2f}, LPIPS={per_lpips[idx]:.4f}, dist={per_image[idx]['final_dist']}")

    # Create visualization rows
    rows = []
    for idx, label in showcase_indices:
        dist = int(per_image[idx]["final_dist"])
        row = make_single_row(
            sources[idx], targets[idx], attacked_pil[idx],
            dist=dist, success=per_image[idx]["success"],
            l2=per_l2[idx], lpips_val=per_lpips[idx],
            label_extra=f"[{label}]",
        )
        rows.append(row)

    grid = make_showcase_grid(rows, title=f"PDQ Best Evolved Attack (eff={0.0321:.4f}) — {n_pairs} pairs, {sum(1 for s in successful)} successful")
    out_path = OUTPUT_DIR / "pdq_best_attack_showcase.png"
    grid.save(out_path)
    print(f"\nSaved: {out_path}")

    # Also save individual pairs for closer inspection
    for idx, label in showcase_indices:
        tag = label.lower().replace(" ", "_")
        # Save attacked image alone
        attacked_pil[idx].save(OUTPUT_DIR / f"pdq_{tag}_attacked.png")
        # Save source
        sources[idx].save(OUTPUT_DIR / f"pdq_{tag}_source.png")
        # Save target
        targets[idx].save(OUTPUT_DIR / f"pdq_{tag}_target.png")

    print(f"Individual images saved to {OUTPUT_DIR}")

    # Summary stats
    asr = compute_asr(per_image)
    mean_l2 = np.mean(per_l2)
    mean_lpips = np.mean(per_lpips)
    print(f"\nOverall: ASR={asr:.2f}, mean_L2={mean_l2:.2f}, mean_LPIPS={mean_lpips:.4f}")

    return {
        "attack_fn": attack_fn,
        "per_image": per_image,
        "per_l2": per_l2,
        "per_lpips": per_lpips,
    }


# ── Hash adapter for transfer compatibility ───────────────────────────────────

class HashAdapter:
    """Wraps a PHFWrapper to normalize hash objects for cross-PHF transfer.

    The pHash attack expects .hash attribute (ImageHash).
    The PDQ attack expects numpy arrays supporting ^ XOR.
    This adapter makes any PHF's hashes compatible with both.
    """

    def __init__(self, phf):
        self.phf = phf
        self.threshold = phf.threshold
        self.name = phf.name

    def compute(self, img):
        h = self.phf.compute(img)
        return AdaptedHash(h, self.phf)

    def distance(self, h1, h2):
        # Unwrap if adapted
        raw1 = h1._raw if isinstance(h1, AdaptedHash) else h1
        raw2 = h2._raw if isinstance(h2, AdaptedHash) else h2
        return self.phf.distance(raw1, raw2)

    def is_collision(self, h1, h2):
        return self.distance(h1, h2) <= self.threshold


class AdaptedHash:
    """Hash object that supports both .hash attribute and ^ XOR."""

    def __init__(self, raw_hash, phf):
        self._raw = raw_hash
        self._phf = phf

        # .hash attribute (for pHash-style attacks)
        if hasattr(raw_hash, 'hash'):
            self.hash = raw_hash.hash  # ImageHash → (8,8) bool
        elif isinstance(raw_hash, np.ndarray):
            # Reshape to (N, N) if possible for DCT-style access
            n = int(np.sqrt(len(raw_hash)))
            if n * n == len(raw_hash):
                self.hash = raw_hash.reshape(n, n).astype(bool)
            else:
                self.hash = raw_hash.astype(bool)
        else:
            self.hash = np.array(raw_hash, dtype=bool)

        # Flat binary array for XOR
        if hasattr(raw_hash, 'hash'):
            self._bits = raw_hash.hash.flatten().astype(np.uint8)
        elif isinstance(raw_hash, np.ndarray):
            self._bits = raw_hash.flatten().astype(np.uint8)
        else:
            self._bits = np.array(raw_hash, dtype=np.uint8).flatten()

    def __xor__(self, other):
        if isinstance(other, AdaptedHash):
            return self._bits ^ other._bits
        if hasattr(other, 'hash'):
            return self._bits ^ other.hash.flatten().astype(np.uint8)
        return self._bits ^ np.array(other, dtype=np.uint8).flatten()

    def __sub__(self, other):
        """Hamming distance (for ImageHash compatibility)."""
        xor = self ^ other
        return int(np.sum(xor))

    def __repr__(self):
        return f"AdaptedHash({self._bits.shape})"


# ── TASK 2: Transfer attacks ─────────────────────────────────────────────────

def test_transfer(attack_name: str, attack_fn, target_phf_name: str, n_pairs: int = 5,
                  seed: int = 42, hyperparams: dict | None = None):
    """Test an attack on a different hash function. Returns metrics dict."""
    raw_phf = get_phf(target_phf_name)
    phf = HashAdapter(raw_phf)
    pairs = load_image_pairs(DATA_DIR, n_pairs=n_pairs, seed=seed)
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    target_hashes = [phf.compute(img) for img in targets]

    context = {
        "hash_fn": phf,
        "threshold": phf.threshold,
        "source_images": sources,
        "target_hashes": target_hashes,
        "target_images": targets,
    }

    try:
        t0 = time.time()
        result = attack_fn(context)
        elapsed = time.time() - t0
    except Exception as e:
        return {
            "attack": attack_name,
            "target_phf": target_phf_name,
            "error": str(e),
            "asr": 0.0,
        }

    attacked_pil = result.get("attacked_images", [])
    per_image = result.get("metrics", [])

    orig_arrs = [image_to_array(img) for img in sources]
    atk_arrs = [image_to_array(img) for img in attacked_pil]

    asr = compute_asr(per_image)
    mean_l2 = compute_mean_l2(orig_arrs, atk_arrs) if atk_arrs else float("nan")
    dists = [m.get("final_dist", float("nan")) for m in per_image]
    mean_dist = float(np.mean(dists)) if dists else float("nan")

    return {
        "attack": attack_name,
        "target_phf": target_phf_name,
        "asr": asr,
        "mean_l2": mean_l2,
        "mean_dist": mean_dist,
        "threshold": phf.threshold,
        "n_pairs": n_pairs,
        "time": elapsed,
        "per_image": per_image,
        "sources": sources,
        "targets": targets,
        "attacked": attacked_pil,
    }


def make_robust_phash_attack(original_entrypoint):
    """Wrap the pHash analytical attack so that the DCT phase gracefully
    falls through to the generic boundary-walk when the target hash has
    a different shape (transfer scenario)."""

    def _robust_attack_single(img, target_hash, hash_fn, threshold,
                              max_iter=50, base_margin_factor=0.3,
                              target_img=None):
        orig_rgb = np.array(img).astype(np.float32)
        n_queries = 0
        best_rgb = orig_rgb.copy()
        best_dist = float(hash_fn.distance(hash_fn.compute(img), target_hash))
        best_l2 = float('inf')
        n_queries += 1

        # Try analytical DCT phase — only works for pHash-shaped hashes
        try:
            from scipy.fft import dct, idct
            from scipy.ndimage import zoom

            def _dct2(a):
                return dct(dct(a.astype(float), axis=0), axis=1)

            def _idct2(a):
                return idct(idct(a, axis=1), axis=0)

            H, W = orig_rgb.shape[:2]
            w_rgb = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            target_bits = target_hash.hash  # may fail if wrong shape

            gray32 = np.array(
                img.convert("L").resize((32, 32), Image.LANCZOS)
            ).astype(np.float64)
            current32 = gray32.copy()

            d = _dct2(current32)
            block = d[:8, :8].copy()
            # Shape check — abort analytical phase if hash isn't 8x8
            if target_bits.shape != (8, 8):
                raise ValueError(f"Hash shape {target_bits.shape} != (8,8)")

            base_margin = base_margin_factor * np.std(block)
            margin = max(base_margin, 0.3)
            no_improvement_count = 0
            prev_dist = best_dist

            for iteration in range(max_iter):
                if best_dist <= threshold:
                    break
                d = _dct2(current32)
                block = d[:8, :8].copy()
                mean = block.mean()
                current_bits = block > mean
                wrong_mask = current_bits != target_bits
                if not wrong_mask.any():
                    break

                wrong_indices = [(i, j, abs(block[i, j] - mean))
                                 for i in range(8) for j in range(8) if wrong_mask[i, j]]
                wrong_indices.sort(key=lambda x: x[2], reverse=True)
                new_block = block.copy()
                for i, j, _ in wrong_indices:
                    c = block[i, j]
                    if target_bits[i, j]:
                        new_block[i, j] = mean + (mean - c) / 63.0 + margin
                    else:
                        new_block[i, j] = mean - (c - mean) / 63.0 - margin

                still_wrong = (new_block > new_block.mean()) != target_bits
                if still_wrong.any():
                    for _ in range(4):
                        new_mean = new_block.mean()
                        still_wrong = (new_block > new_mean) != target_bits
                        if not still_wrong.any():
                            break
                        for i, j, _ in wrong_indices:
                            if not still_wrong[i, j]:
                                continue
                            c = new_block[i, j]
                            if target_bits[i, j]:
                                new_block[i, j] = new_mean + (new_mean - c) / 63.0 + margin
                            else:
                                new_block[i, j] = new_mean - (c - new_mean) / 63.0 - margin

                d_new = d.copy()
                d_new[:8, :8] = new_block
                new_gray32 = np.clip(_idct2(d_new), 0, 255)
                delta32 = new_gray32 - gray32
                delta_up = zoom(delta32, (H / 32.0, W / 32.0), order=1)
                delta_rgb = delta_up[:, :, None] * w_rgb[None, None, :]
                result_rgb = np.clip(orig_rgb + delta_rgb, 0, 255)

                def _nl2(o, p):
                    diff = p.astype(float) - o.astype(float)
                    return float(np.linalg.norm(diff.flatten()) / np.sqrt(o.size))

                current_l2 = _nl2(orig_rgb, result_rgb)
                dist = float(hash_fn.distance(hash_fn.compute(Image.fromarray(result_rgb.astype(np.uint8))), target_hash))
                n_queries += 1

                if dist <= threshold:
                    if current_l2 < best_l2:
                        best_dist = dist
                        best_l2 = current_l2
                        best_rgb = result_rgb
                        current32 = new_gray32
                    break

                if dist < best_dist or (dist == best_dist and current_l2 < best_l2):
                    best_dist = dist
                    best_l2 = current_l2
                    best_rgb = result_rgb
                    current32 = new_gray32
                    no_improvement_count = 0
                    margin *= 0.8 if prev_dist - dist > 2 else 0.7
                else:
                    no_improvement_count += 1
                    margin *= 0.7
                prev_dist = dist
                if no_improvement_count >= 3:
                    if margin < 0.1:
                        margin = 0.1
                    no_improvement_count = 0
                if margin > 50:
                    break

        except Exception as e:
            print(f"      [analytical DCT skipped: {e}]")

        # Generic boundary-walk fallback
        if best_dist > threshold and target_img is not None:
            def _nl2(o, p):
                diff = p.astype(float) - o.astype(float)
                return float(np.linalg.norm(diff.flatten()) / np.sqrt(o.size))

            src_arr = orig_rgb.copy()
            tgt_arr = np.array(target_img).astype(np.float32)
            lo, hi = 0.0, 1.0
            boundary_point = tgt_arr.copy()

            for _ in range(8):
                mid = (lo + hi) / 2.0
                cand = (1 - mid) * src_arr + mid * tgt_arr
                cand_img = Image.fromarray(np.clip(cand, 0, 255).astype(np.uint8))
                test_dist = float(hash_fn.distance(hash_fn.compute(cand_img), target_hash))
                n_queries += 1
                if test_dist <= threshold:
                    hi = mid
                    boundary_point = cand.copy()
                else:
                    lo = mid

            current = boundary_point.copy()
            for _ in range(30):
                direction = src_arr - current
                direction_norm = np.linalg.norm(direction.flatten())
                if direction_norm < 1e-8:
                    break
                direction = direction / direction_norm
                step_size = 3.0
                candidate = np.clip(current + step_size * direction, 0, 255)
                cand_img = Image.fromarray(candidate.astype(np.uint8))
                test_dist = float(hash_fn.distance(hash_fn.compute(cand_img), target_hash))
                n_queries += 1
                if test_dist <= threshold:
                    current = candidate
                    test_l2 = _nl2(src_arr, current)
                    if test_l2 < best_l2:
                        best_dist = test_dist
                        best_l2 = test_l2
                        best_rgb = current
                else:
                    step_size *= 0.5
                    if step_size < 0.5:
                        break

        final_pil = Image.fromarray(np.clip(best_rgb, 0, 255).astype(np.uint8))
        final_dist = float(hash_fn.distance(hash_fn.compute(final_pil), target_hash))
        n_queries += 1

        def _nl2(o, p):
            diff = p.astype(float) - o.astype(float)
            return float(np.linalg.norm(diff.flatten()) / np.sqrt(o.size))

        l2 = _nl2(orig_rgb, best_rgb)
        return final_pil, {
            "success": final_dist <= threshold,
            "l2": l2,
            "n_queries": n_queries,
            "final_dist": final_dist,
        }

    def robust_entrypoint(context: dict) -> dict:
        hash_fn = context["hash_fn"]
        threshold = context["threshold"]
        sources = context["source_images"]
        target_hashes = context["target_hashes"]
        target_images = context.get("target_images", [None] * len(sources))

        attacked_images, metrics = [], []
        for img, th, tgt in zip(sources, target_hashes, target_images):
            atk, m = _robust_attack_single(img, th, hash_fn, threshold,
                                           max_iter=50, base_margin_factor=0.3,
                                           target_img=tgt)
            attacked_images.append(atk)
            metrics.append(m)
        return {"attacked_images": attacked_images, "metrics": metrics}

    return robust_entrypoint


def task2_transfer_attacks():
    print("\n" + "=" * 70)
    print("TASK 2: Transfer Attack Testing")
    print("=" * 70)

    # Load best programs
    best_pdq_path = PROJECT_ROOT / "snapshots" / "pdq" / "run_20260330_205147_last" / "bin_18" / "6fe64a77-0c19-41eb-acfd-bb75c216.py"
    best_phash_path = PROJECT_ROOT / "snapshots" / "phash" / "best_programm.py"

    pdq_attack = load_program(best_pdq_path)
    phash_attack_raw = load_program(best_phash_path)
    phash_attack = make_robust_phash_attack(phash_attack_raw)

    # Available target PHFs (excluding the source PHF for each attack)
    # Try photodna only if available
    all_phfs = ["phash", "pdq", "neuralhash"]
    try:
        get_phf("photodna")
        all_phfs.append("photodna")
        print("PhotoDNA available")
    except Exception as e:
        print(f"PhotoDNA not available: {e}")

    n_pairs = 5
    results = []

    # Test PDQ attack on other PHFs
    print(f"\n--- Testing PDQ best attack (HSJA+DCT hybrid) on other PHFs ---")
    for phf_name in all_phfs:
        if phf_name == "pdq":
            continue
        print(f"\n  PDQ attack → {phf_name}...")
        r = test_transfer("PDQ_best", pdq_attack, phf_name, n_pairs=n_pairs)
        if "error" in r:
            print(f"    ERROR: {r['error']}")
        else:
            print(f"    ASR={r['asr']:.2f}, mean_L2={r['mean_l2']:.2f}, time={r['time']:.1f}s")
        results.append(r)

    # Test pHash attack on other PHFs
    print(f"\n--- Testing pHash best attack (analytical DCT + boundary walk) on other PHFs ---")
    for phf_name in all_phfs:
        if phf_name == "phash":
            continue
        print(f"\n  pHash attack → {phf_name}...")
        r = test_transfer("pHash_best", phash_attack, phf_name, n_pairs=n_pairs)
        if "error" in r:
            print(f"    ERROR: {r['error']}")
        else:
            print(f"    ASR={r['asr']:.2f}, mean_L2={r['mean_l2']:.2f}, time={r['time']:.1f}s")
        results.append(r)

    # Print summary table
    print(f"\n{'='*70}")
    print("TRANSFER ATTACK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Attack':<15} {'Target PHF':<12} {'ASR':>6} {'mean L2':>10} {'mean dist':>10} {'threshold':>10} {'Time(s)':>8}")
    print("-" * 75)
    for r in results:
        if "error" in r:
            print(f"{r['attack']:<15} {r['target_phf']:<12} {'ERR':>6} {'—':>10} {'—':>10} {'—':>10} {'—':>8}")
        else:
            print(f"{r['attack']:<15} {r['target_phf']:<12} {r['asr']:>6.2f} {r['mean_l2']:>10.2f} {r['mean_dist']:>10.1f} {r['threshold']:>10} {r['time']:>8.1f}")

    # Also print per-pair details
    print(f"\n{'='*70}")
    print("PER-PAIR DETAILS")
    print(f"{'='*70}")
    for r in results:
        if "error" in r:
            continue
        print(f"\n  {r['attack']} → {r['target_phf']} (threshold={r['threshold']}):")
        for i, m in enumerate(r['per_image']):
            print(f"    pair {i}: {'✓' if m['success'] else '✗'}  dist={m.get('final_dist', '?'):>8.1f}  L2={m.get('l2', '?'):>8.2f}  queries={m.get('n_queries', '?')}")

    # Save summary to CSV
    import csv
    csv_path = OUTPUT_DIR / "transfer_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack", "target_phf", "asr", "mean_l2", "n_pairs", "time_s"])
        for r in results:
            writer.writerow([
                r["attack"], r["target_phf"],
                r.get("asr", 0), r.get("mean_l2", ""),
                r.get("n_pairs", ""), r.get("time", ""),
            ])
    print(f"\nResults saved: {csv_path}")

    # Create visualization grids for successful transfers
    for r in results:
        if "error" in r or r["asr"] == 0:
            continue
        rows = []
        for i in range(min(3, len(r.get("attacked", [])))):
            m = r["per_image"][i]
            if not m.get("success"):
                continue
            orig_arr = image_to_array(r["sources"][i])
            atk_arr = image_to_array(r["attacked"][i])
            diff = atk_arr.astype(float) - orig_arr.astype(float)
            l2 = float(np.linalg.norm(diff.flatten()) / np.sqrt(orig_arr.size))
            row = make_single_row(
                r["sources"][i], r["targets"][i], r["attacked"][i],
                dist=int(m.get("final_dist", 0)), success=True,
                l2=l2, lpips_val=float("nan"),
                label_extra=f"pair#{i}",
            )
            rows.append(row)

        if rows:
            grid = make_showcase_grid(rows, title=f"Transfer: {r['attack']} → {r['target_phf']} (ASR={r['asr']:.2f})")
            out = OUTPUT_DIR / f"transfer_{r['attack']}_{r['target_phf']}.png"
            grid.save(out)
            print(f"Saved: {out}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def task3_pdq_no_superposition():
    """Visualize the best PDQ attack that does NOT use target image superposition."""
    print("\n" + "=" * 70)
    print("TASK 3: PDQ Non-Superposition Attack (NES)")
    print("=" * 70)

    nes_path = PROJECT_ROOT / "snapshots" / "pdq" / "run_20260330_205147_last" / "bin_unplaced" / "c6c0d2bd-48f9-4826-8409-cd9cbdfe.py"
    print(f"Loading NES attack: {nes_path.name}")
    attack_fn = load_program(nes_path)

    phf = get_phf("pdq")
    n_pairs = 10
    pairs = load_image_pairs(DATA_DIR, n_pairs=n_pairs, seed=42)
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    target_hashes = [phf.compute(img) for img in targets]

    context = {
        "hash_fn": phf,
        "threshold": phf.threshold,
        "source_images": sources,
        "target_hashes": target_hashes,
        "target_images": targets,
    }

    print(f"Running NES attack on {n_pairs} pairs (may be slow, ~500 iters × 20 samples)...")
    t0 = time.time()
    result = attack_fn(context)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    attacked_pil = result["attacked_images"]
    per_image = result["metrics"]

    orig_arrs = [image_to_array(img) for img in sources]
    atk_arrs = [image_to_array(img) for img in attacked_pil]

    per_l2 = []
    for o, a in zip(orig_arrs, atk_arrs):
        diff = a.astype(float) - o.astype(float)
        per_l2.append(float(np.linalg.norm(diff.flatten()) / np.sqrt(o.size)))

    print("Computing per-pair LPIPS...")
    per_lpips = compute_per_pair_lpips(orig_arrs, atk_arrs)

    print(f"\n{'Pair':>4} {'Success':>8} {'Dist':>6} {'L2':>8} {'LPIPS':>8} {'Queries':>8}")
    print("-" * 50)
    for i, (m, l2, lp) in enumerate(zip(per_image, per_l2, per_lpips)):
        print(f"{i:>4} {'✓' if m['success'] else '✗':>8} {m['final_dist']:>6.0f} {l2:>8.2f} {lp:>8.4f} {m['n_queries']:>8}")

    successful = [(i, per_l2[i], per_lpips[i]) for i in range(n_pairs) if per_image[i]["success"]]

    if not successful:
        print("\nNo successful attacks!")
        return

    best_l2_idx = min(successful, key=lambda x: x[1])[0]
    best_lpips_idx = min(successful, key=lambda x: x[2])[0]
    sorted_by_l2 = sorted(successful, key=lambda x: x[1])
    median_idx = sorted_by_l2[len(sorted_by_l2) // 2][0]

    if best_lpips_idx == best_l2_idx and len(successful) > 1:
        sorted_by_lpips = sorted(successful, key=lambda x: x[2])
        best_lpips_idx = sorted_by_lpips[1][0]
    used = {best_l2_idx, best_lpips_idx}
    if median_idx in used:
        for entry in sorted_by_l2:
            if entry[0] not in used:
                median_idx = entry[0]
                break

    showcase_indices = [
        (best_l2_idx, "Best L2"),
        (best_lpips_idx, "Best LPIPS"),
        (median_idx, "Median"),
    ]

    print(f"\nShowcase pairs:")
    for idx, label in showcase_indices:
        print(f"  {label}: pair #{idx}, L2={per_l2[idx]:.2f}, LPIPS={per_lpips[idx]:.4f}, dist={per_image[idx]['final_dist']}")

    rows = []
    for idx, label in showcase_indices:
        dist = int(per_image[idx]["final_dist"])
        row = make_single_row(
            sources[idx], targets[idx], attacked_pil[idx],
            dist=dist, success=per_image[idx]["success"],
            l2=per_l2[idx], lpips_val=per_lpips[idx],
            label_extra=f"[{label}]",
        )
        rows.append(row)

    n_succ = sum(1 for s in successful)
    asr = compute_asr(per_image)
    mean_l2_val = np.mean(per_l2)
    mean_lpips_val = np.mean(per_lpips)
    grid = make_showcase_grid(rows, title=f"PDQ NES Attack (no superposition, eff=0.0098) — {n_pairs} pairs, {n_succ} successful")
    out_path = OUTPUT_DIR / "pdq_nes_no_superposition_showcase.png"
    grid.save(out_path)
    print(f"\nSaved: {out_path}")

    for idx, label in showcase_indices:
        tag = label.lower().replace(" ", "_")
        attacked_pil[idx].save(OUTPUT_DIR / f"pdq_nes_{tag}_attacked.png")
        sources[idx].save(OUTPUT_DIR / f"pdq_nes_{tag}_source.png")
        targets[idx].save(OUTPUT_DIR / f"pdq_nes_{tag}_target.png")

    print(f"Individual images saved to {OUTPUT_DIR}")
    print(f"\nOverall: ASR={asr:.2f}, mean_L2={mean_l2_val:.2f}, mean_LPIPS={mean_lpips_val:.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["1", "2", "3", "all"], default="all")
    args = ap.parse_args()

    if args.task in ("1", "all"):
        task1_pdq_visualization()
    if args.task in ("2", "all"):
        task2_transfer_attacks()
    if args.task in ("3", "all"):
        task3_pdq_no_superposition()
    print(f"\nAll results in: {OUTPUT_DIR}")
