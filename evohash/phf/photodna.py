"""PhotoDNA wrapper — cross-platform implementation via DLL.

Platform behaviour
------------------
Windows  : DLL loaded directly via ctypes — fast (~1-5 ms per image).
Linux/mac: DLL called via a Wine Python subprocess (~0.3-1 s per image).

Digest   : np.ndarray uint8, shape (144,)
Distance : L1 norm
Threshold: 3855

Setup
-----
Windows:
  1. Extract PhotoDNAx64.dll from FTK ISO (7-Zip).
  2. Set PHOTODNA_DLL env var or pass dll_path to constructor.

Linux:
  from evohash.phf.photodna import setup_photodna
  setup_photodna()  # downloads FTK ISO, extracts DLL, fetches Wine Python

"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from .base import PHFWrapper

# ---------------------------------------------------------------------------
# Platform detection & defaults
# ---------------------------------------------------------------------------

_IS_WINDOWS = platform.system() == "Windows"

_DEFAULT_WORK_DIR = (
    os.environ.get("PHOTODNA_WORK_DIR")
    or os.path.join(os.path.expanduser("~"), ".cache", "photodna")
)
# DLL bundled in repo at data/photodna/PhotoDNAx64.dll
_BUNDLED_DLL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
    "data", "photodna", "PhotoDNAx64.dll",
)
_DEFAULT_DLL_PATH = (
    os.environ.get("PHOTODNA_DLL", "")
    or (_BUNDLED_DLL if os.path.isfile(_BUNDLED_DLL) else "")
)
_WINE_PYTHON_SUBPATH = "python-3.9.12-embed-amd64/python.exe"
_WINE_CMD = shutil.which("wine64") or shutil.which("wine") or "wine64"

# ---------------------------------------------------------------------------
# generateHashes.py — Wine subprocess helper
# ---------------------------------------------------------------------------

_GENERATE_HASHES_SRC = r'''"""PhotoDNA hash generator — runs inside Wine Python."""
import sys, os, ctypes
from ctypes import c_char_p, c_int, c_ubyte, POINTER
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

DLL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PhotoDNAx64.dll")
_lib = ctypes.cdll.LoadLibrary(DLL_PATH)
_fn  = _lib.ComputeRobustHash
_fn.argtypes = [c_char_p, c_int, c_int, c_int, POINTER(c_ubyte), c_int]
_fn.restype  = c_ubyte

def compute(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    buf = (c_ubyte * 144)()
    _fn(c_char_p(img.tobytes()), img.width, img.height, 0, buf, 0)
    return ",".join(str(v) for v in buf)

if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(compute(path))
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_pil_rgb(image) -> Image.Image:
    """Convert various inputs to a PIL RGB Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    # numpy array
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr).convert("RGB")


def _parse_hash_line(line: str) -> np.ndarray:
    values = [int(v.strip()) for v in line.strip().split(",") if v.strip()]
    if len(values) != 144:
        raise RuntimeError(f"Expected 144 hash values, got {len(values)}")
    return np.array(values, dtype=np.uint8)


# ===========================================================================
# Windows backend: direct ctypes
# ===========================================================================


class _WindowsBackend:
    def __init__(self, dll_path: str) -> None:
        if not os.path.isfile(dll_path):
            raise FileNotFoundError(
                f"[PhotoDNA] DLL not found: {dll_path}\n"
                "Set PHOTODNA_DLL env var or pass dll_path= to PhotoDNAWrapper()."
            )
        lib = ctypes.CDLL(dll_path)
        fn = lib.ComputeRobustHash
        fn.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
        ]
        fn.restype = ctypes.c_ubyte
        self._fn = fn

    def compute(self, pil_img: Image.Image) -> np.ndarray:
        buf = (ctypes.c_ubyte * 144)()
        self._fn(
            ctypes.c_char_p(pil_img.tobytes()),
            pil_img.width, pil_img.height,
            0, buf, 0,
        )
        return np.array(list(buf), dtype=np.uint8)

    def compute_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        return [self.compute(img) for img in images]


# ===========================================================================
# Linux/macOS backend: Wine subprocess
# ===========================================================================


class _WineBackend:
    def __init__(self, work_dir: str) -> None:
        wine_python = os.path.join(work_dir, _WINE_PYTHON_SUBPATH)
        script = os.path.join(work_dir, "vendor", "generateHashes.py")
        if not os.path.isfile(wine_python) or not os.path.isfile(script):
            raise RuntimeError(
                "[PhotoDNA] Not set up. Run:\n"
                "  from evohash.phf.photodna import setup_photodna\n"
                "  setup_photodna()"
            )
        self._wine_python = wine_python
        self._script = script

    def _run(self, args: list[str], timeout: int = 120) -> str:
        result = subprocess.run(
            [_WINE_CMD, self._wine_python, self._script, *args],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "WINEDEBUG": "-all"},
        )
        if result.returncode != 0:
            raise RuntimeError(f"PhotoDNA Wine failed: {result.stderr}")
        return result.stdout

    def compute(self, pil_img: Image.Image) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
            pil_img.save(tmp)
        try:
            stdout = self._run([tmp])
        finally:
            if os.path.isfile(tmp):
                os.remove(tmp)
        return _parse_hash_line(stdout.splitlines()[0])

    def compute_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Compute hashes for multiple images in one Wine subprocess."""
        tmps: list[str] = []
        try:
            for img in images:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    tmps.append(f.name)
                    img.save(f.name)
            stdout = self._run(tmps)
        finally:
            for t in tmps:
                if os.path.isfile(t):
                    os.remove(t)
        lines = [ln for ln in stdout.splitlines() if ln.strip()]
        if len(lines) != len(images):
            raise RuntimeError(
                f"Expected {len(images)} hash lines, got {len(lines)}"
            )
        return [_parse_hash_line(ln) for ln in lines]


# ===========================================================================
# Setup helper
# ===========================================================================


def setup_photodna(work_dir: str = _DEFAULT_WORK_DIR, force: bool = False) -> str:
    """One-time setup. Returns path to the DLL (Windows) or Wine Python (Linux)."""
    if _IS_WINDOWS:
        dll = _DEFAULT_DLL_PATH or os.path.join(work_dir, "PhotoDNAx64.dll")
        if os.path.isfile(dll):
            print(f"[PhotoDNA] Windows — DLL found: {dll}")
            return dll
        print(
            "[PhotoDNA] Windows setup:\n"
            "  1. Extract PhotoDNAx64.dll from FTK ISO:\n"
            "     https://d1kpmuwb7gvu1i.cloudfront.net/AD_FTK_7.0.0.iso\n"
            f"  2. Place it in: {work_dir}\n"
            "     or set PHOTODNA_DLL env var"
        )
        return ""

    # Linux / macOS
    wine_python = os.path.join(work_dir, _WINE_PYTHON_SUBPATH)
    dll_path = os.path.join(work_dir, "PhotoDNAx64.dll")

    if os.path.isfile(wine_python) and os.path.isfile(dll_path) and not force:
        print(f"[PhotoDNA] Already set up at {work_dir}")
        _ensure_vendor_script(work_dir)
        return wine_python

    os.makedirs(work_dir, exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir(work_dir)
    try:
        # Install dependencies
        for pkg in ("wine64", "cabextract", "genisoimage", "curl"):
            if not shutil.which(pkg):
                subprocess.run(f"apt-get update -qq && apt-get install -y -q {pkg}",
                               shell=True, check=True, text=True)

        if not os.path.isfile(dll_path):
            print("[PhotoDNA] Downloading FTK ISO (~3.3 GB)...")
            subprocess.run(
                "curl -L --progress-bar -o AD_FTK_7.0.0.iso "
                "https://d1kpmuwb7gvu1i.cloudfront.net/AD_FTK_7.0.0.iso",
                shell=True, check=True)
            subprocess.run(
                "isoinfo -i AD_FTK_7.0.0.iso -x /FTK/FTK/X64/_8A89F09/DATA1.CAB > Data1.cab",
                shell=True, check=True)
            os.makedirs("tmp_cab", exist_ok=True)
            subprocess.run("cabextract -d tmp_cab -q Data1.cab", shell=True, check=True)

            found = None
            for root, _, files in os.walk("tmp_cab"):
                for fname in files:
                    if "photodna" in fname.lower() and fname.endswith(".dll"):
                        found = os.path.join(root, fname)
                        break
            if not found:
                raise RuntimeError("PhotoDNAx64.dll not found in FTK ISO")
            shutil.copy(found, dll_path)

            for f in ("AD_FTK_7.0.0.iso", "Data1.cab"):
                if os.path.isfile(f):
                    os.remove(f)
            shutil.rmtree("tmp_cab", ignore_errors=True)

        if not os.path.isfile(wine_python):
            print("[PhotoDNA] Downloading Wine Python 3.9...")
            subprocess.run(
                "curl -L --progress-bar -o wine_python_39.tar.gz "
                "https://github.com/jankais3r/pyPhotoDNA/releases/download/"
                "wine_python_39/wine_python_39.tar.gz",
                shell=True, check=True)
            subprocess.run("tar -xf wine_python_39.tar.gz", shell=True, check=True)
            os.remove("wine_python_39.tar.gz")

        _ensure_vendor_script(work_dir)
        print("[PhotoDNA] Setup complete")
        return wine_python
    finally:
        os.chdir(orig_dir)


def _ensure_vendor_script(work_dir: str) -> None:
    vendor_dir = os.path.join(work_dir, "vendor")
    os.makedirs(vendor_dir, exist_ok=True)
    script_path = os.path.join(vendor_dir, "generateHashes.py")
    with open(script_path, "w") as f:
        f.write(_GENERATE_HASHES_SRC)
    dll_src = os.path.join(work_dir, "PhotoDNAx64.dll")
    dll_lnk = os.path.join(vendor_dir, "PhotoDNAx64.dll")
    if os.path.isfile(dll_src) and not os.path.exists(dll_lnk):
        os.symlink(dll_src, dll_lnk)


# ===========================================================================
# Public wrapper — implements PHFWrapper ABC
# ===========================================================================


class PhotoDNAWrapper(PHFWrapper):
    """Microsoft PhotoDNA perceptual hash (Windows + Linux/Wine).

    Backend is selected automatically and resolved lazily on first compute().
    """

    threshold: int = 3855

    def __init__(
        self,
        dll_path: str = _DEFAULT_DLL_PATH,
        work_dir: str = _DEFAULT_WORK_DIR,
    ) -> None:
        self._dll_path = dll_path
        self._work_dir = work_dir
        self._backend: _WindowsBackend | _WineBackend | None = None

    def _resolve(self) -> None:
        if self._backend is not None:
            return
        if _IS_WINDOWS:
            dll = (
                self._dll_path
                or os.environ.get("PHOTODNA_DLL", "")
                or os.path.join(self._work_dir, "PhotoDNAx64.dll")
            )
            self._backend = _WindowsBackend(dll)
        else:
            self._backend = _WineBackend(self._work_dir)

    @property
    def name(self) -> str:
        return "PhotoDNA"

    def compute(self, image: Image.Image) -> np.ndarray:
        """Compute PhotoDNA hash. Returns np.ndarray uint8, shape (144,)."""
        self._resolve()
        pil_img = _to_pil_rgb(image)
        return self._backend.compute(pil_img)

    def compute_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Compute hashes for multiple images. On Linux/Wine, uses a single subprocess."""
        self._resolve()
        pil_imgs = [_to_pil_rgb(img) for img in images]
        return self._backend.compute_batch(pil_imgs)

    def distance(self, h1: Any, h2: Any) -> float:
        """L1 distance between two 144-byte PhotoDNA digests."""
        a = np.asarray(h1, dtype=np.int32)
        b = np.asarray(h2, dtype=np.int32)
        return float(np.abs(a - b).sum())
