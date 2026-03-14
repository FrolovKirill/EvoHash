# NeuralHash Implementation Plan

## Situation

Apple's NeuralHash model files are present on this Mac at:
`/System/Library/Frameworks/Vision.framework/Resources/`

Files available:
- `NeuralHashv3b_fp16-current.espresso.{net,shape,weights}` — compiled CoreML model (fp16)
- `neuralhash_128x96_seed1.dat` — seed matrix for final hash computation

Already done: decoded and copied all 4 files to `data/neuralhash_model/`.

## Problem

The model is in Apple's proprietary **espresso** format (compiled CoreML).
Standard conversion tools (TNN coreml2onnx) fail on the fp16 variant.

## Plan

### Step 1 — Convert espresso → CoreML .mlpackage
Use macOS `coremlcompiler` CLI tool (ships with Xcode) to decompile the
compiled espresso back to a `.mlpackage` or `.mlmodel`:
```bash
xcrun coremlcompiler decompile \
  data/neuralhash_model/model.espresso.net \
  data/neuralhash_model/
```

### Step 2 — Run inference via CoreML Python API (no ONNX needed)
If decompile works → load with `coremltools` and run inference directly.
If not → use macOS `Vision` framework via `PyObjC` (already on macOS):
```python
import Vision, CoreML
# Run NeuralHashv3b model via native macOS Vision API
```

### Step 3 — Implement NeuralHashWrapper
Fill in `evohash/phf/neuralhash.py`:
- `compute(image)` → 96-bit hash (numpy array of bits)
- `distance(h1, h2)` → Hamming distance
- Input: resize to 360×360, normalize to [-1, 1]
- Output: `seed_matrix @ model_output` → sign → 96 bits

### Step 4 — Implement problem definition
- Fill `problems/neuralhash/context.py` — load image pairs + NeuralHashWrapper
- Fill `problems/neuralhash/validate.py` — same fitness logic as phash/pdq
- Add seed attack programs: `random_noise.py`, `nes_attack.py`, `simba_attack.py`

### Step 5 — Test & commit
- Smoke test: hash two images, verify distance is integer in [0, 96]
- Verify threshold=17 works correctly
- Add `data/neuralhash_model/` to `.gitignore` (model weights are Apple-proprietary)
- Commit working implementation

## Fallback
If CoreML/PyObjC approach fails:
- Use macOS `swift` one-liner to call Vision framework and pipe hash as JSON
- Wrap as subprocess call in Python
