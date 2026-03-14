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

| PHF        | Hash size | Collision threshold | Status          |
|------------|-----------|---------------------|-----------------|
| pHash      | 64 bits   | Hamming ≤ 12        | ✅ Implemented  |
| PDQ        | 256 bits  | Hamming ≤ 92        | ✅ Implemented  |
| NeuralHash | 96 bits   | Hamming ≤ 17        | 🚧 Stub (Apple proprietary) |
| PhotoDNA   | 144 floats | L1 ≤ 3855          | 🚧 Stub (Microsoft proprietary) |

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
│   │   ├── neuralhash.py       # TODO stub
│   │   └── photodna.py         # TODO stub
│   ├── dataset.py              # ImageNet Val image pair loader
│   └── evaluation.py           # ASR, L2, LPIPS, efficiency metrics
├── problems/
│   ├── phash/                  # gigaevo problem definition for pHash
│   │   ├── context.py          # Runtime context (images + hash fn)
│   │   ├── validate.py         # Fitness evaluator
│   │   ├── metrics.yaml        # Metric specs (efficiency as primary)
│   │   ├── task_description.txt # LLM prompt for mutation
│   │   ├── helper.py           # Shared attack utilities
│   │   └── initial_programs/   # Seed strategies
│   │       ├── random_noise.py # Gaussian noise baseline
│   │       ├── nes_attack.py   # Natural Evolution Strategy
│   │       └── simba_attack.py # SimBa (DCT-basis coordinate descent)
│   └── pdq/                    # Same structure for PDQ
├── scripts/
│   ├── download_dataset.py     # Download ImageNet Val subset
│   └── evaluate.py             # Full benchmark evaluation
└── data/imagenet_val/          # Images (gitignored, populated by download script)
```

## How It Works

1. **Problem definition** — each PHF gets a `problems/<phf>/` directory with a fitness evaluator (`validate.py`) and seed attack programs (`initial_programs/`).
2. **Evolution** — GigaEvo runs a MAP-Elites evolutionary loop: seed programs are mutated by an LLM (OpenRouter OSS models), evaluated by running them against the PHF, and the best strategies are stored and selected for future mutations.
3. **Fitness** — the evaluator re-verifies attack success via hash queries (not trusting the program's self-report) and returns `efficiency = ASR / (mean_L2 + ε)`.
4. **Output** — evolved attack programs are stored in Redis; GigaEvo logs to `gigaevo-core/outputs/`.

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
| **Transferability** | Success rate of pHash-evolved attacks applied to PDQ (and vice versa) |

## Baselines

The initial seed programs implement classical gradient-free attacks:
- **Random noise** — i.i.d. Gaussian perturbations, keep the best sample
- **NES** — Natural Evolution Strategy with antithetic sampling
- **SimBa** — Simple Black-box Adversarial attacks in DCT/pixel-block basis

## Limitations

- **NeuralHash**: Apple's model weights are not publicly available. See `evohash/phf/neuralhash.py`.
- **PhotoDNA**: Requires a Microsoft API licence. See `evohash/phf/photodna.py`.
- **GigaChat-2-Max**: Not yet integrated (not available on OpenRouter).
- **Dataset**: ImageNet-1k requires a HuggingFace account and acceptance of the dataset licence. The download script falls back to Tiny-ImageNet or synthetic images.

## License

MIT
