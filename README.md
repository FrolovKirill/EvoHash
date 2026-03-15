# EvoHash

**LLM-guided evolutionary discovery of collision attacks on perceptual hash functions.**

EvoHash uses [GigaEvo](https://github.com/FusionBrainLab/gigaevo-core) — an evolutionary programming framework powered by LLMs — to automatically discover black-box collision attacks against perceptual hash functions (PHFs) used in content moderation and digital rights management.

## Overview

Perceptual hash functions (pHash, PDQ, NeuralHash, PhotoDNA) map visually similar images to identical or near-identical digests. A **collision attack** finds a minimally-perturbed image whose hash matches a target hash. EvoHash evolves Python attack strategies that maximise:

```
efficiency = ASR / (mean_L2 + ε)
```

where **ASR** is the fraction of successful collisions and **mean_L2** is the average pixel-space distortion. Higher efficiency = more successful attacks with less visible perturbation.

## Supported PHFs

| PHF        | Hash size   | Collision threshold | Status                                      |
|------------|-------------|---------------------|---------------------------------------------|
| pHash      | 64 bits     | Hamming ≤ 12        | ✅ Implemented                              |
| PDQ        | 256 bits    | Hamming ≤ 92        | ✅ Implemented                              |
| NeuralHash | 96 bits     | Hamming ≤ 17        | ✅ Implemented (macOS via Vision framework) |
| PhotoDNA   | 144 floats  | L1 ≤ 3855           | 🚧 Stub (Microsoft proprietary API)         |

## Baselines

All 8 baselines from the paper are implemented for pHash, PDQ, and NeuralHash:

| Attack | Description |
|--------|-------------|
| **random_noise** | i.i.d. Gaussian perturbations, keep the best sample |
| **NES** | Natural Evolution Strategy with antithetic sampling |
| **SimBa** | Simple Black-box Adversarial attacks in DCT/pixel basis |
| **ZO-Sign-SGD** | Zeroth-order sign gradient descent |
| **HSJA** | HopSkipJump — decision-boundary walk from target image |
| **NES+HSJA** | HSJA initialisation → NES refinement |
| **SimBa+HSJA** | HSJA initialisation → SimBa DCT-basis refinement |
| **ZO-Sign-SGD+HSJA** | HSJA initialisation → ZO-Sign-SGD refinement |
| **Prokos** | Frequency-targeted DCT-domain gradient attack |
| **AtkScopes** | Multi-scale patch sensitivity attack |

## Project Structure

```
EvoHash/
├── run_evohash.py              # Main entry point
├── requirements.txt
├── evohash/
│   ├── phf/                    # PHF wrapper library
│   │   ├── base.py             # Abstract PHFWrapper interface
│   │   ├── phash.py            # pHash (imagehash)
│   │   ├── pdq.py              # PDQ (pdqhash)
│   │   ├── neuralhash.py       # NeuralHash (macOS Vision via PyObjC)
│   │   └── photodna.py         # TODO stub (Microsoft API)
│   ├── dataset.py              # ImageNet Val image pair loader
│   ├── evaluation.py           # ASR, L2, LPIPS, Efficiency, Transferability
│   └── reporter.py             # W&B experiment tracking (metrics + image grids)
├── problems/
│   ├── phash/                  # gigaevo problem definition for pHash
│   │   ├── context.py          # Runtime context (images + hash fn)
│   │   ├── validate.py         # Fitness evaluator
│   │   ├── metrics.yaml        # Metric specs (efficiency as primary)
│   │   ├── task_description.txt # LLM prompt for mutation
│   │   ├── helper.py           # Shared attack utilities
│   │   └── initial_programs/   # 10 seed attack strategies
│   ├── pdq/                    # Same structure for PDQ
│   └── neuralhash/             # Same structure for NeuralHash (macOS only)
├── config/
│   └── llm/
│       └── openrouter_gpt_oss.yaml  # GPT-OSS 120B via OpenRouter
├── scripts/
│   ├── download_dataset.py     # Download ImageNet Val subset
│   ├── evaluate.py             # Full benchmark evaluation
│   └── show_best.py            # Dump best evolved programs from Redis
└── data/imagenet_val/          # Images (gitignored, populated by download script)
```

## How It Works

1. **Problem definition** — each PHF gets a `problems/<phf>/` directory with a fitness evaluator (`validate.py`) and seed attack programs (`initial_programs/`).
2. **Evolution** — GigaEvo runs a MAP-Elites evolutionary loop: seed programs are mutated by an LLM (GPT OSS 120B via OpenRouter), evaluated by running them against the PHF, and the best strategies are stored and selected for future mutations.
3. **Fitness** — the evaluator re-verifies attack success via hash queries (not trusting the program's self-report) and returns `efficiency = ASR / (mean_L2 + ε)`.
4. **Output** — evolved attack programs are stored in Redis; metrics and image grids are logged to [Weights & Biases](https://wandb.ai/) in real time. Use `scripts/show_best.py` to export top programs from Redis.

## Quick Start

See [QUICKSTART.md](QUICKSTART.md).

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ASR** | Attack Success Rate — fraction of image pairs that achieved a hash collision |
| **mean L2** | Mean normalised L2 pixel distortion: `‖δ‖₂ / √(H·W·C)` |
| **Efficiency** | `ASR / (mean_L2 + ε)` — primary fitness metric |
| **LPIPS** | Learned Perceptual Image Patch Similarity (AlexNet) |
| **Queries** | Mean number of hash evaluations per attack |
| **Time** | Mean wall-clock seconds per attack |
| **Transferability** | Success rate of attacks evolved for one PHF applied to another |

## Limitations

- **PhotoDNA**: Requires a Microsoft API licence. See `evohash/phf/photodna.py`.
- **NeuralHash**: macOS only (uses Apple Vision framework via PyObjC). Requires `pip install pyobjc-framework-Vision pyobjc-core` and copying `neuralhash_128x96_seed1.dat` from `/System/Library/Frameworks/Vision.framework/Resources/`.
- **GigaChat-2-Max**: Not yet integrated (not available on OpenRouter).
- **Dataset**: ImageNet-1k requires a HuggingFace account and acceptance of the dataset licence. The download script falls back to Tiny-ImageNet or synthetic images.

## License

MIT
