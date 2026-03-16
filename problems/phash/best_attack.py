import numpy as np
from PIL import Image
from scipy.fft import dct, idct
from scipy.ndimage import zoom

def entrypoint(context: dict) -> dict:
    """Optimised analytical pHash attack.
    Reduces L2 by tightening per‑coefficient margins, using true luminance
    weighting, and preserving float precision during up‑sampling.  A binary‑
    search scaling stage finalises the minimal perturbation while keeping the
    hash within the collision threshold.
    """
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources = context["source_images"]
    target_hashes = context["target_hashes"]

    attacked_images = []
    metrics = []
    for src_img, tgt_hash in zip(sources, target_hashes):
        adv_img, metric = _attack(src_img, tgt_hash, hash_fn, threshold)
        attacked_images.append(adv_img)
        metrics.append(metric)
    return {"attacked_images": attacked_images, "metrics": metrics}

# ---------------------------------------------------------------------------
# Helper functions – all operate in the 32×32 grayscale DCT domain
# ---------------------------------------------------------------------------

def _dct2(arr: np.ndarray) -> np.ndarray:
    """2‑D orthogonal DCT (type‑II, norm='ortho')."""
    return dct(dct(arr, axis=0, norm='ortho'), axis=1, norm='ortho')

def _idct2(arr: np.ndarray) -> np.ndarray:
    """Inverse orthogonal DCT (type‑III, norm='ortho')."""
    return idct(idct(arr, axis=1, norm='ortho'), axis=0, norm='ortho')

def _attack(
    img: Image.Image,
    target_hash,
    hash_fn,
    threshold: int,
    max_iter: int = 20,
    base_margin: float = 0.05,
    margin_growth: float = 1.02,
    max_margin: float = 0.5,
) -> tuple[Image.Image, dict]:
    """Core analytical attack.
    * Uses a tiny per‑coefficient margin (base_margin) to avoid overshoot.
    * Recomputes the block mean after each coefficient change for stability.
    * Applies true luminance weighting when projecting the grayscale delta to RGB.
    * Upsamples the perturbation with float‑precision zoom (order=0 – nearest‑neighbor).
    * After a successful collision a binary‑search scaling step reduces L2.
    """
    # Original RGB as float32
    orig_rgb = np.array(img).astype(np.float32)
    H, W = orig_rgb.shape[:2]

    # Luminance weighting – projects a grayscale delta to minimal‑norm RGB delta
    w_rgb = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    w_norm_sq = (w_rgb ** 2).sum()
    w_scaled = w_rgb / w_norm_sq  # ensures ||delta_rgb||_2 = ||delta_gray||_2

    # Target bits (8×8 bool array)
    target_bits = np.array(target_hash.hash, dtype=bool)

    # Grayscale 32×32 source (exact pHash pipeline)
    gray32 = np.array(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    current_gray = gray32.copy()

    # Initial distance and query count
    best_dist = float(hash_fn.distance(hash_fn.compute(img), target_hash))
    n_queries = 1
    best_rgb = orig_rgb.copy()
    margin = base_margin

    for _ in range(max_iter):
        if best_dist <= threshold:
            break
        # DCT of current grayscale
        dct_full = _dct2(current_gray)
        block = dct_full[:8, :8]
        mean_val = block.mean()
        cur_bits = block > mean_val
        wrong = cur_bits != target_bits
        if not np.any(wrong):
            break
        # Update each mismatched coefficient, recomputing mean after each change
        new_block = block.copy()
        for i, j in zip(*np.where(wrong)):
            if target_bits[i, j]:
                new_block[i, j] = mean_val + margin
            else:
                new_block[i, j] = mean_val - margin
            mean_val = new_block.mean()
        # Optional second drift‑correction pass (max 1 additional pass)
        cur_bits = new_block > mean_val
        if np.any(cur_bits != target_bits):
            # One extra correction pass
            for i, j in zip(*np.where(cur_bits != target_bits)):
                if target_bits[i, j]:
                    new_block[i, j] = mean_val + margin
                else:
                    new_block[i, j] = mean_val - margin
                mean_val = new_block.mean()
        # Re‑assemble and inverse DCT
        dct_mod = dct_full.copy()
        dct_mod[:8, :8] = new_block
        new_gray32 = np.clip(_idct2(dct_mod), 0, 255)

        # Compute float‑precision delta and upsample (nearest‑neighbor to keep magnitude)
        delta32 = new_gray32 - gray32
        delta_up = zoom(delta32, (H / 32.0, W / 32.0), order=0)
        delta_rgb = delta_up[:, :, None] * w_scaled[None, None, :]
        candidate_rgb = np.clip(orig_rgb + delta_rgb, 0, 255)
        candidate_img = Image.fromarray(candidate_rgb.astype(np.uint8))

        dist = float(hash_fn.distance(hash_fn.compute(candidate_img), target_hash))
        n_queries += 1
        if dist < best_dist:
            best_dist = dist
            best_rgb = candidate_rgb
            current_gray = new_gray32
        # Grow margin modestly, respecting max_margin
        margin = min(margin * margin_growth, max_margin)

    # ---------------------------------------------------------------------
    # Scaling‑down phase – binary search for smallest scalar preserving collision
    # ---------------------------------------------------------------------
    if best_dist <= threshold:
        best_rgb, scale_q = _scale_down(orig_rgb, best_rgb, target_hash, hash_fn, threshold)
        n_queries += scale_q
        final_dist = float(hash_fn.distance(hash_fn.compute(Image.fromarray(best_rgb.astype(np.uint8))), target_hash))
    else:
        final_dist = best_dist

    final_img = Image.fromarray(np.clip(best_rgb, 0, 255).astype(np.uint8))
    n_queries += 1
    l2 = float(np.linalg.norm((best_rgb - orig_rgb).ravel()) / np.sqrt(orig_rgb.size))
    metric = {
        "success": final_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": final_dist,
    }
    return final_img, metric

def _scale_down(
    orig_rgb: np.ndarray,
    pert_rgb: np.ndarray,
    target_hash,
    hash_fn,
    threshold: int,
    max_iters: int = 20,
) -> tuple[np.ndarray, int]:
    """Binary‑search the minimal scalar α∈[0,1] that keeps the hash collision.
    Returns the scaled RGB image and the number of additional hash queries.
    """
    delta = pert_rgb - orig_rgb
    lo, hi = 0.0, 1.0
    best_rgb = pert_rgb.copy()
    queries = 0
    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        cand = np.clip(orig_rgb + mid * delta, 0, 255)
        cand_img = Image.fromarray(cand.astype(np.uint8))
        dist = float(hash_fn.distance(hash_fn.compute(cand_img), target_hash))
        queries += 1
        if dist <= threshold:
            best_rgb = cand
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    return best_rgb, queries
