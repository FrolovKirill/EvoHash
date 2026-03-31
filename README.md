# EvoHash

**LLM-guided evolutionary discovery of collision attacks on perceptual hash functions.**

EvoHash uses [GigaEvo](https://github.com/FusionBrainLab/gigaevo-core) — an evolutionary programming framework powered by LLMs — to automatically discover black-box collision attacks against perceptual hash functions (PHFs) used in content moderation and digital rights management.

## Overview

Perceptual hash functions (pHash, PDQ, NeuralHash, PhotoDNA) map visually similar images to identical or near-identical digests. A **collision attack** finds a minimally-perturbed image whose hash matches a target hash. EvoHash evolves Python attack strategies that maximise:

```
efficiency = ASR / (distortion + ε)
```

where **ASR** is the fraction of successful collisions and **distortion** combines L2 pixel distance and LPIPS perceptual similarity (configurable via `.env` — see `EFFICIENCY_MODE`). Default mode is `weighted_sum`: `d = L2 + 50·LPIPS`. Higher efficiency = more successful attacks with less visible perturbation.

## Supported PHFs

| PHF        | Hash size   | Collision threshold | Status                                      |
|------------|-------------|---------------------|---------------------------------------------|
| pHash      | 64 bits     | Hamming ≤ 12        | ✅ Implemented                              |
| PDQ        | 256 bits    | Hamming ≤ 92        | ✅ Implemented                              |
| NeuralHash | 96 bits     | Hamming ≤ 17        | ✅ Implemented (cross-platform, ONNX)       |
| PhotoDNA   | 144 floats  | L1 ≤ 3855           | ✅ Implemented (Windows DLL / Linux Wine)    |

## Baselines

16 seed attack strategies are implemented for each PHF (pHash, PDQ, NeuralHash, PhotoDNA):

| Attack | Description |
|--------|-------------|
| **random_noise** | i.i.d. Gaussian perturbations, keep the best sample |
| **NES** | Natural Evolution Strategy with antithetic sampling |
| **SimBa** | Simple Black-box Adversarial attacks in DCT/pixel basis |
| **ZO-Sign-SGD** | Zeroth-order sign gradient descent |
| **ZO-Sign-SGD v2** | ZO-Sign-SGD with tuned hyperparameters |
| **HSJA** | HopSkipJump — decision-boundary walk from target image |
| **HSJA v2** | HSJA with boundary-based sign updates |
| **NES+HSJA** | HSJA initialisation → NES refinement |
| **NES+JSHA** | NES initialisation → HSJA refinement |
| **SimBa+HSJA** | HSJA initialisation → SimBa DCT-basis refinement |
| **SimBa+JSHA** | SimBa initialisation → HSJA refinement |
| **ZO-Sign-SGD+HSJA** | HSJA initialisation → ZO-Sign-SGD refinement |
| **ZO-Sign-SGD+JSHA** | ZO-Sign-SGD initialisation → HSJA refinement |
| **ZO-Sign-SGD v2+HSJA** | HSJA init → ZO-Sign-SGD v2 refinement |
| **Prokos** | Frequency-targeted DCT-domain gradient attack |
| **AtkScopes** | Multi-scale patch sensitivity attack |

## Project Structure

```
EvoHash/
├── run_evohash.py              # CLI entry point
├── requirements.txt
├── web/                        # Web UI (FastAPI + React)
│   ├── run_web.py              # Single entry point: python web/run_web.py
│   ├── backend/
│   │   ├── app.py              # FastAPI app (REST + WebSocket)
│   │   ├── runner.py           # Subprocess manager (start/stop/stream)
│   │   └── redis_bridge.py     # Reads metrics/programs from Redis
│   └── frontend/               # React + Vite + TypeScript + Tailwind
│       └── src/pages/          # Dashboard / Monitor / Results / Baselines
├── evohash/
│   ├── phf/                    # PHF wrapper library
│   │   ├── base.py             # Abstract PHFWrapper interface
│   │   ├── phash.py            # pHash (imagehash)
│   │   ├── pdq.py              # PDQ (pdqhash)
│   │   ├── neuralhash.py       # NeuralHash (cross-platform ONNX)
│   │   └── photodna.py         # PhotoDNA (Windows DLL / Linux Wine)
│   ├── attacks/                # Unified hash-agnostic attack library
│   │   ├── random_noise.py     # Gaussian noise baseline
│   │   ├── nes.py              # Natural Evolution Strategies
│   │   ├── prokos.py           # Frequency-targeted DCT gradient attack
│   │   ├── simba.py            # Simple Black-box (DCT + block bases)
│   │   ├── zo_signsgd.py       # Zeroth-order sign gradient descent
│   │   ├── atkscopes.py        # Multi-scale patch sensitivity
│   │   ├── hsja.py             # HopSkipJump decision-boundary walk
│   │   ├── hybrid.py           # Two-stage hybrids (HSJA + refinement)
│   │   ├── analytical_dct.py   # pHash-specific DCT attack
│   │   └── utils.py            # Shared helpers (L2, clamp, query)
│   ├── dataset.py              # ImageNet Val image pair loader
│   ├── evaluation.py           # ASR, L2, LPIPS, Efficiency, Transferability
│   └── reporter.py             # W&B experiment tracking (metrics + image grids)
├── tests/                      # pytest test suite
│   ├── test_phf_wrappers.py    # PHF wrapper tests (pHash, PDQ, NeuralHash)
│   ├── test_attacks.py         # Attack interface contract tests
│   └── test_attack_utils.py    # Attack utility function tests
├── problems/
│   ├── phash/                  # gigaevo problem definition for pHash
│   │   ├── context.py          # Runtime context (images + hash fn)
│   │   ├── validate.py         # Fitness evaluator
│   │   ├── metrics.yaml        # Metric specs (efficiency as primary)
│   │   ├── task_description.txt # LLM prompt for mutation
│   │   └── initial_programs/   # 16 seed attack strategies (thin wrappers)
│   ├── pdq/                    # Same structure for PDQ
│   ├── neuralhash/             # Same structure for NeuralHash
│   └── photodna/               # Same structure for PhotoDNA
├── proxy/
│   └── openrouter_proxy.py         # Structured-output fixup proxy for OpenRouter
├── config/
│   ├── models.yaml                    # Single source of truth for available LLM models
│   └── llm/
│       └── openrouter.yaml            # Hydra template for OpenRouter (model injected at runtime)
├── scripts/
│   ├── download_dataset.py         # Download ImageNet Val subset
│   ├── evaluate.py                 # Full benchmark evaluation
│   ├── hyperparam_sweep.py         # Hyperparameter grid search for attacks
│   ├── show_best.py                # Dump best evolved programs from Redis
│   ├── archive_manager.py          # Snapshot save/load & LLM analysis
│   ├── analyze_patterns.py         # LLM-based program classification
│   └── visualize_and_transfer.py   # Visualize attacks & test cross-PHF transfer
└── data/imagenet_val/          # Images (gitignored, populated by download script)
```

## How It Works

1. **Problem definition** — each PHF gets a `problems/<phf>/` directory with a fitness evaluator (`validate.py`) and seed attack programs (`initial_programs/`).
2. **Evolution** — GigaEvo runs a MAP-Elites evolutionary loop: seed programs are mutated by an LLM (configurable via `config/models.yaml` — GPT-OSS 120B, GLM-4.7 Flash, Qwen 3.5 35B/122B, Mistral Small 2603, etc.), evaluated by running them against the PHF, and the best strategies are stored and selected for future mutations. A local proxy (`proxy/openrouter_proxy.py`) automatically fixes structured-output format mismatches between OpenRouter models and GigaEvo's pydantic schemas.
3. **Fitness** — the evaluator re-verifies attack success via hash queries (not trusting the program's self-report) and returns `efficiency = ASR / (distortion + ε)`, where distortion combines L2 and LPIPS.
4. **Output** — evolved attack programs are stored in Redis; metrics and image grids are logged to [Weights & Biases](https://wandb.ai/) in real time. Use `scripts/show_best.py` to export top programs from Redis.

### OpenRouter Proxy

EvoHash communicates with LLMs via [OpenRouter](https://openrouter.ai/). Many open-source models on OpenRouter return malformed structured output (e.g. a bare JSON list instead of an object, or truncated responses). The local proxy (`proxy/openrouter_proxy.py`) transparently fixes these issues:

- **JSON shape fixup** — wraps bare `[...]` into `{"field": [...]}` when the schema expects an object
- **Truncation retry** — retries with doubled `max_tokens` when `finish_reason=length`
- **Syntax validation** — retries if the generated `code` field has a Python `SyntaxError`

The proxy runs on `localhost:8100` (FastAPI + uvicorn, no Docker needed) and is started automatically by both `run_evohash.py` and the web UI. No manual setup required.

## Quick Start

### Web UI (recommended)

The easiest way to run EvoHash is via the built-in web interface.
Requires Node.js 18+ (for the React frontend build).

```bash
# Install dependencies
pip install -r requirements.txt -r web/backend/requirements.txt

# Clone GigaEvo and apply required patches
git clone https://github.com/FusionBrainLab/gigaevo-core gigaevo-core
pip install -e gigaevo-core/
sed -i.bak 's/primary_resolution: 150/primary_resolution: 20/' gigaevo-core/config/constants/islands.yaml && rm gigaevo-core/config/constants/islands.yaml.bak
sed -i.bak 's/max_length=5/max_length=8/' gigaevo-core/gigaevo/llm/agents/lineage.py && rm gigaevo-core/gigaevo/llm/agents/lineage.py.bak

# Configure and launch
cp .env.example .env  # add OPENROUTER_API_KEY
redis-server --daemonize yes
python web/run_web.py
# → opens http://localhost:8765
```

> **gigaevo-core patches:** EvoHash requires two changes in gigaevo-core after cloning:
> 1. `config/constants/islands.yaml` — reduce `primary_resolution` from 150 to 20 (MAP-Elites grid granularity)
> 2. `gigaevo/llm/agents/lineage.py` — change `max_length=5` to `max_length=8` (prevents LLM validation errors)

The web UI handles dataset download, proxy startup, live log streaming, and metrics charts.
Flags: `--port PORT`, `--no-browser`, `--dev` (hot-reload).

### CLI

See [QUICKSTART.md](QUICKSTART.md) for the full manual setup.

### Hyperparameter Sweep

Search for working hyperparameter configurations for gradient-based attacks:

```bash
# Sweep all attacks on pHash with 3 image pairs (fast, ~10 min)
python scripts/hyperparam_sweep.py --phf phash --n-pairs 3

# Sweep on PDQ with 10 pairs (slower, more reliable)
python scripts/hyperparam_sweep.py --phf pdq --n-pairs 10
```

Results are saved incrementally to `results_hyperparam_sweep.csv` (append mode, safe to interrupt) with detailed per-pair logs in `hyperparam_sweep.log`. The sweep covers NES, SimBa, ZO-Sign-SGD, Prokos, and ATKScopes with multiple configurations per attack.

### Baseline Evaluation

Evaluate all seed attack programs for a given PHF:

```bash
# Evaluate all seeds for pHash, save results for the web UI
python scripts/evaluate.py --phf phash --all-seeds --csv results_baselines.csv

# Evaluate a single program
python scripts/evaluate.py --phf pdq --program problems/pdq/initial_programs/hsja_attack.py --n-pairs 10
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ASR** | Attack Success Rate — fraction of image pairs that achieved a hash collision |
| **mean L2** | Mean normalised L2 pixel distortion: `‖δ‖₂ / √(H·W·C)` |
| **Efficiency** | `ASR / (distortion + ε)` — primary fitness metric (distortion = L2 + LPIPS, configurable) |
| **LPIPS** | Learned Perceptual Image Patch Similarity (AlexNet) |
| **Queries** | Mean number of hash evaluations per attack |
| **Time** | Mean wall-clock seconds per attack |
| **Transferability** | Success rate of attacks evolved for one PHF applied to another |

## Results

Full presentation: [EvoHash.pdf](EvoHash.pdf)

### Experimental setup

- **Models:** GPT-OSS 120B, GLM-4.7 Flash, **Qwen 3.5 122B-A10B** (via OpenRouter)
- **Dataset:** 100 random image pairs from ImageNet Val
- **Evolution:** 20 generations, 10 pairs per evaluation

### pHash evolution (20 generations, Qwen 3.5 122B)

| Stage | Efficiency | L2 | LPIPS | Queries |
|-------|----------:|----:|------:|--------:|
| Best baseline (Analytical DCT) | 0.022 | 32.9 | 0.268 | 3 |
| **Evolved attack** (gen 5, DCT + NES+HSJA cross) | **0.032** | **22.1** | **0.183** | **2** |

Efficiency **+45%**, L2 **-33%**, LPIPS **-32%**. The evolved program adds adaptive margin, sorted bit correction, and a boundary walk phase to the analytical DCT baseline.

### PDQ evolution (20 generations, Qwen 3.5 122B)

| Stage | Efficiency | L2 | LPIPS | Queries |
|-------|----------:|----:|------:|--------:|
| Best baseline (NES+HSJA) | 0.0225 | 30.0 | 0.289 | 3025 |
| **Evolved attack** (gen 7, HSJA + analytical DCT) | **0.0321** | **23.9** | **0.146** | **119** |

Efficiency **+43%**, L2 **-20%**, LPIPS **-49%**, queries **96% fewer**. The LLM independently invented an analytical DCT approach on generation 2, then crossed it with HSJA-based attacks by generation 7.

### Attack transferability

Evolved attacks transfer to all four PHFs with **100% ASR** without retraining (30 image pairs):

| Evolved for ↓ / Tested on → | PDQ | pHash | NeuralHash | PhotoDNA |
|------------------------------|-----|-------|------------|----------|
| **PDQ best** | 1.00 / L2=30.5 | 1.00 / L2=56.8 | 1.00 / L2=71.2 | 1.00 / L2=54.9 |
| **pHash best** | 1.00 / L2=31.0 | 1.00 / L2=22.1 | 1.00 / L2=71.1 | 1.00 / L2=54.8 |

## Limitations

- **PhotoDNA**: On Windows, requires `PhotoDNAx64.dll` (bundled in `data/photodna/`). On macOS, uses a Docker container with Wine — Docker Desktop must be running (image builds automatically on first use). On Linux with Wine set up, uses Wine directly.
- **GigaChat-2-Max**: Not yet integrated (not available on OpenRouter).
- **Dataset**: ImageNet-1k requires a HuggingFace account and acceptance of the dataset licence. The download script falls back to Tiny-ImageNet or synthetic images.

## Team

Kirill Frolov, Daniil Koblov, Amir Sadreev, Grigorii Gulii, Yakupova Valeria

601 RUSKOLKEK

## License

MIT
