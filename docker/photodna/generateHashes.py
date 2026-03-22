"""PhotoDNA stdin/stdout server — runs inside Wine Python in Docker."""
import sys, io, base64, ctypes
from ctypes import c_char_p, c_int, c_ubyte, POINTER
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

_lib = ctypes.cdll.LoadLibrary(r"Z:\app\PhotoDNAx64.dll")
_fn  = _lib.ComputeRobustHash
_fn.argtypes = [c_char_p, c_int, c_int, c_int, POINTER(c_ubyte), c_int]
_fn.restype  = c_ubyte

def compute_b64(b64line: str) -> str:
    img_bytes = base64.b64decode(b64line.strip())
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = (c_ubyte * 144)()
    _fn(c_char_p(img.tobytes()), img.width, img.height, 0, buf, 0)
    return ",".join(str(v) for v in buf)

if __name__ == "__main__":
    # Signal readiness so the host knows Wine+DLL initialised OK.
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            result = compute_b64(line)
        except Exception as e:
            result = f"ERROR:{e}"
        sys.stdout.write(result + "\n")
        sys.stdout.flush()
