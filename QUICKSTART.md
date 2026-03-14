# Quick Start

## Prerequisites

- Python 3.12+
- Redis server
- An [OpenRouter](https://openrouter.ai/) API key (for OSS LLM mutations)

## 1. Clone and install

```bash
# Clone EvoHash
git clone <your-repo-url> EvoHash
cd EvoHash

# Clone GigaEvo (required at runtime)
git clone https://github.com/gigaevo/gigaevo.git gigaevo-core

# Install GigaEvo and EvoHash dependencies
pip install -e gigaevo-core/
pip install imagehash pdqhash pillow numpy scipy tabulate
```

For LPIPS (optional, used in full evaluation):
```bash
pip install torch torchvision lpips
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

# More options
python run_evohash.py phash --max-generations 100 --llm openrouter_bandit
python run_evohash.py pdq   --max-generations 50  --redis-db 1

# Resume a previous run
python run_evohash.py phash --resume
```

Outputs are written to `gigaevo-core/outputs/YYYY-MM-DD/HH-MM-SS/`.

## 6. Evaluate results

Benchmark all seed strategies (no evolution required):
```bash
python scripts/evaluate.py --phf phash --all-seeds
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
| Program      | PHF   |  ASR | mean L2 | Efficiency | Queries | LPIPS | Time (s) |
|--------------|-------|------|---------|------------|---------|-------|----------|
| random_noise | phash | 0.00 |    3.21 |     0.0000 |     201 | 0.042 |     1.23 |
| nes_attack   | phash | 0.10 |    8.47 |     0.0118 |    3001 | 0.089 |    12.50 |
| simba_attack | phash | 0.20 |   12.31 |     0.0163 |    2401 | 0.127 |    14.80 |
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

**`Redis database is not empty`**
→ Either flush it (`redis-cli -n 0 FLUSHDB`) or use `--resume` to continue, or use a different DB (`--redis-db 1`).

**`OPENAI_API_KEY not set` warning**
→ Set `OPENROUTER_API_KEY` in your environment or `.env` file.

**`gigaevo-core not found`**
→ Clone the repo into `gigaevo-core/` (see Step 1).
