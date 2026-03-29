"""ATKScopes -- multiresolution adversarial perturbation attack.

Three scales: pixel (direct pixel coordinate descent), global (full-image
DCT coefficient descent), mid (patch-based local DCT descent). Each iteration
picks one random coordinate, estimates gradient via symmetric finite-difference,
and updates with per-coordinate Adam optimizer.

Reference: Zhang et al., USENIX Security 2025.
"""

import numpy as np
from PIL import Image
from scipy.fft import dct, idct


def normalised_l2(original, perturbed):
    diff = perturbed.astype(float) - original.astype(float)
    return float(np.linalg.norm(diff.flatten()) / np.sqrt(original.size))


def query_distance(arr, target_hash, hash_fn):
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return float(hash_fn.distance(hash_fn.compute(img), target_hash))


def _dct2(block):
    return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def _idct2(block):
    return idct(idct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def _patch_coords(h, w, k):
    """Non-overlapping grid of (r0, r1, c0, c1) tuples."""
    coords = []
    for r in range(0, h - k + 1, k):
        for c in range(0, w - k + 1, k):
            coords.append((r, r + k, c, c + k))
    return coords


class _Adam1D:
    __slots__ = ("m", "v", "t", "beta1", "beta2", "eps")

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = 0.0
        self.v = 0.0
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self, g, lr):
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g * g)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return -lr * m_hat / (v_hat ** 0.5 + self.eps)


class _PixelState:
    """Pixel-scale: one pixel-channel coordinate per step."""

    def __init__(self, shape):
        self.delta = np.zeros(shape, dtype=np.float32)

    def make_probes(self, x0, rng, a):
        H, W, C = x0.shape
        i, j, ch = int(rng.integers(H)), int(rng.integers(W)), int(rng.integers(C))
        key = ("px", i, j, ch)

        x_cur = np.clip(x0 + self.delta, 0, 255)
        e = np.zeros_like(self.delta)
        e[i, j, ch] = a

        plus_img = np.clip(x_cur + e, 0, 255)
        minus_img = np.clip(x_cur - e, 0, 255)

        def apply_step(step_val):
            self.delta[i, j, ch] += step_val
            result = np.clip(x0 + self.delta, 0.0, 255.0)
            self.delta[:] = result - x0
            return result

        return key, plus_img, minus_img, apply_step


class _GlobalDCTState:
    """Global DCT: one DCT coefficient per step over full image."""

    def __init__(self, x0):
        H, W, C = x0.shape
        self.coeffs = [_dct2(x0[:, :, ch]) for ch in range(C)]
        self.delta = [np.zeros((H, W), dtype=np.float32) for _ in range(C)]
        self.C = C

    def _build(self):
        chs = [np.clip(_idct2(self.coeffs[k] + self.delta[k]), 0.0, 255.0)
               for k in range(self.C)]
        return np.stack(chs, axis=2).astype(np.float32)

    def make_probes(self, x0, rng, a):
        H, W, C = x0.shape
        fi, fj, ch = int(rng.integers(H)), int(rng.integers(W)), int(rng.integers(C))
        key = ("dctG", fi, fj, ch)

        old = float(self.delta[ch][fi, fj])

        self.delta[ch][fi, fj] = old + a
        plus_img = self._build()
        self.delta[ch][fi, fj] = old - a
        minus_img = self._build()
        self.delta[ch][fi, fj] = old

        def apply_step(step_val):
            self.delta[ch][fi, fj] += step_val
            return self._build()

        return key, plus_img, minus_img, apply_step


class _MidDCTState:
    """Mid-scale local DCT: one coefficient in a random patch per step."""

    def __init__(self, x0, k):
        H, W, C = x0.shape
        self.k = k
        self.patches = _patch_coords(H, W, k)
        if not self.patches:
            self.patches = [(0, H, 0, W)]
        self.delta = np.zeros_like(x0, dtype=np.float32)

    def make_probes(self, x0, rng, a):
        H, W, C = x0.shape
        pidx = int(rng.integers(len(self.patches)))
        r0, r1, c0, c1 = self.patches[pidx]
        ph, pw = r1 - r0, c1 - c0
        fi, fj, ch = int(rng.integers(ph)), int(rng.integers(pw)), int(rng.integers(C))
        key = ("dctM", pidx, fi, fj, ch)

        coeff = np.zeros((ph, pw), dtype=np.float32)
        coeff[fi, fj] = 1.0
        basis = _idct2(coeff).astype(np.float32)
        nrm = float(np.linalg.norm(basis.ravel()))
        if nrm > 1e-12:
            basis /= nrm

        x_cur = np.clip(x0 + self.delta, 0, 255)

        plus_img = x_cur.copy()
        plus_img[r0:r1, c0:c1, ch] = np.clip(
            plus_img[r0:r1, c0:c1, ch] + a * basis, 0, 255)

        minus_img = x_cur.copy()
        minus_img[r0:r1, c0:c1, ch] = np.clip(
            minus_img[r0:r1, c0:c1, ch] - a * basis, 0, 255)

        def apply_step(step_val):
            self.delta[r0:r1, c0:c1, ch] += step_val * basis
            result = np.clip(x0 + self.delta, 0.0, 255.0)
            self.delta[:] = result - x0
            return result

        return key, plus_img, minus_img, apply_step


_SCALE_DEFAULTS = {
    "pixel":  {"a": 2.0,  "n_iter": 1000},
    "global": {"a": 50.0, "n_iter": 2000},
    "mid":    {"a": 5.0,  "n_iter": 800},
}

def _attack_single(img, target_hash, hash_fn, threshold,
                   scale="global", n_iter=None, lr=0.05, a=None,
                   patch_size=None, beta1=0.9, beta2=0.999, eps=1e-8):
    """Attack one image using ATKScopes coordinate descent with Adam."""
    defaults = _SCALE_DEFAULTS[scale]
    if n_iter is None:
        n_iter = defaults["n_iter"]
    if a is None:
        a = defaults["a"]
    orig = np.array(img).astype(np.float32)
    H, W, C = orig.shape
    k = patch_size or max(H // 4, 8)

    rng = np.random.default_rng()

    if scale == "pixel":
        state = _PixelState(orig.shape)
    elif scale == "global":
        state = _GlobalDCTState(orig)
    elif scale == "mid":
        state = _MidDCTState(orig, k)
    else:
        raise ValueError(f"Unknown scale: {scale}")

    adam_states = {}
    best = orig.copy()
    best_dist = query_distance(orig, target_hash, hash_fn)
    n_queries = 1

    for _ in range(n_iter):
        if best_dist <= threshold:
            break

        key, plus_img, minus_img, apply_step = state.make_probes(orig, rng, a)

        fp = query_distance(plus_img, target_hash, hash_fn)
        fn = query_distance(minus_img, target_hash, hash_fn)
        n_queries += 2

        g = (fp - fn) / (2.0 * a)

        opt = adam_states.get(key)
        if opt is None:
            opt = _Adam1D(beta1, beta2, eps)
            adam_states[key] = opt

        step_val = opt.step(g, lr)
        x_new = apply_step(step_val)

        dist_new = query_distance(x_new, target_hash, hash_fn)
        n_queries += 1

        if dist_new < best_dist:
            best_dist = dist_new
            best = x_new.copy()

    l2 = normalised_l2(orig, best)
    return Image.fromarray(np.clip(best, 0, 255).astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }


def entrypoint(context: dict) -> dict:
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources = context["source_images"]
    target_hashes = context["target_hashes"]

    attacked_images, metrics = [], []
    for img, th in zip(sources, target_hashes):
        atk, m = _attack_single(img, th, hash_fn, threshold, scale="global")
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}