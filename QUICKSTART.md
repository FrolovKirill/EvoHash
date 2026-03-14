# Quick Start

## Prerequisites

- Python 3.12+
- Redis server
<<<<<<< HEAD
- An [OpenRouter](https://openrouter.ai/) API key (for GPT OSS mutations)
=======
- An [OpenRouter](https://openrouter.ai/) API key (for OSS LLM mutations)
- **macOS** (required only for NeuralHash)
>>>>>>> 7cf9802633783b6613d5b3bdb7f010afae946c78

## 1. Clone and install

```bash
# Clone EvoHash
git clone <your-repo-url> EvoHash
cd EvoHash

# Clone GigaEvo into gigaevo-core/ and install as editable
# (editable install is required: run_evohash.py calls gigaevo-core/run.py at runtime;
#  pip install gigaevo alone is not enough because run.py and config/ are not published to PyPI)
git clone https://github.com/FusionBrainLab/gigaevo-core gigaevo-core
pip install -e gigaevo-core/

# Install EvoHash dependencies
pip install imagehash pdqhash pillow numpy scipy tabulate
```

For LPIPS (optional, used in full evaluation):
```bash
pip install torch torchvision lpips
```

For **NeuralHash** (macOS only):
```bash
pip install pyobjc-framework-Vision pyobjc-core

# Copy seed matrix from macOS system
mkdir -p data/neuralhash_model
cp /System/Library/Frameworks/Vision.framework/Resources/neuralhash_128x96_seed1.dat \
   data/neuralhash_model/seed1.dat
```

## 2. Configure environment

```bash
# Copy and edit the example env file
cp .env.example .env   # or create .env manually
```

`.env` contents:
```
OPENROUTER_API_KEY=sk-or-...
```

> **Note:** `run_evohash.py` automatically maps `OPENROUTER_API_KEY → OPENAI_API_KEY`
> so GigaEvo's LLM configs work without modification.

## 3. Download dataset

```bash
# ~200 images (tries ImageNet → Tiny-ImageNet → synthetic fallback)
python scripts/download_dataset.py --n-images 200
```

If you have a HuggingFace account with ImageNet-1k access, log in first:
```bash
huggingface-cli login
```

For a quick smoke test without any downloads:
```bash
python scripts/download_dataset.py --synthetic --n-images 30
```

## 4. Start Redis

```bash
redis-server --daemonize yes
```

Verify it's running:
```bash
redis-cli ping   # should print PONG
```

## 5. Run evolution

```bash
# Evolve pHash attacks for 50 generations
python run_evohash.py phash

<<<<<<< HEAD
# More options
python run_evohash.py phash --max-generations 100
python run_evohash.py pdq   --max-generations 50  --redis-db 1
=======
# Evolve PDQ attacks
python run_evohash.py pdq --max-generations 100 --llm openrouter_bandit

# Evolve NeuralHash attacks (macOS only)
python run_evohash.py neuralhash --max-generations 50

# Use a different Redis DB to run multiple PHFs in parallel
python run_evohash.py phash   --redis-db 0
python run_evohash.py pdq     --redis-db 1
python run_evohash.py neuralhash --redis-db 2
>>>>>>> 7cf9802633783b6613d5b3bdb7f010afae946c78

# Resume a previous run
python run_evohash.py phash --resume
```

Outputs are written to `gigaevo-core/outputs/YYYY-MM-DD/HH-MM-SS/`.

## 6. Evaluate results

Benchmark all seed strategies (no evolution required):
```bash
python scripts/evaluate.py --phf phash --all-seeds
python scripts/evaluate.py --phf pdq --all-seeds
python scripts/evaluate.py --phf neuralhash --all-seeds   # macOS only
```

Evaluate a specific program:
```bash
python scripts/evaluate.py --phf phash \
    --program problems/phash/initial_programs/nes_attack.py \
    --n-pairs 100
```

Save results to CSV:
```bash
python scripts/evaluate.py --phf phash --all-seeds --output-csv results_phash.csv
```

Expected output format:
```
| Program              | PHF   |  ASR | mean L2 | Efficiency | Queries | LPIPS | Time (s) |
|----------------------|-------|------|---------|------------|---------|-------|----------|
| random_noise         | phash | 0.00 |    3.21 |     0.0000 |     201 | 0.042 |     1.23 |
| nes_attack           | phash | 0.10 |    8.47 |     0.0118 |    3001 | 0.089 |    12.50 |
| simba_attack         | phash | 0.20 |   12.31 |     0.0163 |    2401 | 0.127 |    14.80 |
| zo_signsgd_attack    | phash | 0.15 |    7.20 |     0.0208 |    6151 | 0.073 |    18.40 |
| hsja_attack          | phash | 0.80 |   36.20 |     0.0221 |    1615 | 0.312 |     9.80 |
| nes_hsja_attack      | phash | 0.85 |   36.75 |     0.0231 |    3358 | 0.320 |    22.10 |
| simba_hsja_attack    | phash | 0.80 |   39.22 |     0.0204 |    1123 | 0.335 |    15.60 |
| prokos_attack        | phash | 0.05 |    0.34 |     0.1471 |    6121 | 0.012 |    31.20 |
| atkscopes_attack     | phash | 0.02 |    0.11 |     0.1818 |   61673 | 0.004 |   190.50 |
```

## Typical Workflow

```
download data → start Redis → run evolution (hours) → evaluate evolved programs
                                    ↓
                          gigaevo-core/outputs/
                          (best evolved programs stored in Redis)
```

After evolution, export programs from Redis to files using GigaEvo's built-in tools:
```bash
cd gigaevo-core
python tools/redis2pd.py   # export all programs to CSV/Parquet
```

## Troubleshooting

**`FileNotFoundError: data/imagenet_val/ not found`**
→ Run `python scripts/download_dataset.py` first.

**`FileNotFoundError: data/neuralhash_model/seed1.dat`**
→ Copy the seed file from macOS Vision framework (see Step 1, NeuralHash section).

**`ImportError: No module named 'Vision'`**
→ Install PyObjC: `pip install pyobjc-framework-Vision pyobjc-core` (macOS only).

**`Redis database is not empty`**
→ Either flush it (`redis-cli -n 0 FLUSHDB`) or use `--resume` to continue, or use a different DB (`--redis-db 1`).

**`OPENAI_API_KEY not set` warning**
→ Set `OPENROUTER_API_KEY` in your environment or `.env` file.

**`gigaevo-core not found`**
→ Clone the repo into `gigaevo-core/` (see Step 1).
