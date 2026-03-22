# Quick Start

## Option A — Web UI (recommended)

The web interface runs the entire pipeline with one button press: it handles dataset download, streams live logs and metrics, and shows results in a browser.

**Prerequisites:** Python 3.12+, Redis, Node.js 18+ with npm (for the React frontend).

```bash
# 1. Install all dependencies
pip install -r requirements.txt
pip install -r web/backend/requirements.txt

# 2. Clone GigaEvo (required)
git clone https://github.com/FusionBrainLab/gigaevo-core gigaevo-core
pip install -e gigaevo-core/

# 3. Copy and fill in your API key
cp .env.example .env
# edit .env → set OPENROUTER_API_KEY

# 4. Start Redis
redis-server --daemonize yes

# 5. Launch
python web/run_web.py
# → opens http://localhost:8765
```

> `run_web.py` auto-installs backend pip deps and runs `npm install && npm run build` on first launch.
> Use `--port PORT` to change the default port, `--no-browser` to suppress auto-open, `--dev` for hot-reload.

The web UI has four pages:
- **Запуск** — choose PHF and LLM model (loaded from `config/models.yaml`), configure generations/pairs, start/stop the run. Includes "reset to defaults" and "clear data" buttons, and shows last run parameters.
- **Монитор** — live colorized logs, efficiency/ASR chart per generation, separate metric cards and image grid tabs for best and latest programs (source/target/attacked/diff images updated every 2s).
- **Результаты** — table of evolved programs with code viewer and .py download
- **Бейзлайны** — run `evaluate.py` on all baseline attacks with one click

---

## Option B — CLI (manual setup)

### Prerequisites

- Python 3.12+
- Redis server
- An [OpenRouter](https://openrouter.ai/) API key (for LLM mutations — model list in `config/models.yaml`)
- A [Weights & Biases](https://wandb.ai/) API key (for experiment tracking)
- Node.js 18+ with npm (only for Web UI)
- **onnxruntime** (for NeuralHash, cross-platform)

### 1. Clone and install

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
pip install -r requirements.txt
```

For LPIPS (optional, used in full evaluation):
```bash
pip install torch torchvision lpips
```

For **NeuralHash** (cross-platform):
```bash
pip install onnxruntime
# Model files (model.onnx + seed1.dat) are included in data/neuralhash_model/
```

### 2. Configure environment

```bash
# Copy and edit the example env file
cp .env.example .env
```

`.env` contents:
```
OPENROUTER_API_KEY=sk-or-...   # https://openrouter.ai/
WANDB_API_KEY=                 # https://wandb.ai/authorize
WANDB_PROJECT=evohash
```

> **Note:** `run_evohash.py` automatically maps `OPENROUTER_API_KEY → OPENAI_API_KEY`
> so GigaEvo's LLM configs work without modification.
> All keys are read from `.env` automatically — no `wandb login` needed.

### 3. Download dataset

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

### 4. Start Redis

> Redis is required for both CLI and Web UI — the evolution engine stores programs and metrics in Redis.

```bash
redis-server --daemonize yes
```

Verify it's running:
```bash
redis-cli ping   # should print PONG
```

### 5. Run evolution

```bash
# Evolve pHash attacks for 50 generations
python run_evohash.py phash

# More options
python run_evohash.py phash --max-generations 100
python run_evohash.py pdq   --max-generations 50  --redis-db 1

# Evolve NeuralHash attacks (cross-platform via ONNX)
python run_evohash.py neuralhash --max-generations 50

# Use a different Redis DB to run multiple PHFs in parallel
python run_evohash.py phash      --redis-db 0
python run_evohash.py pdq        --redis-db 1
python run_evohash.py neuralhash --redis-db 2

# Resume a previous run
python run_evohash.py phash --resume
```

W&B logs metrics and image grids in real time — a run URL is printed at startup.
Outputs are also written to `gigaevo-core/outputs/YYYY-MM-DD/HH-MM-SS/`.

> **Note:** `run_evohash.py` automatically starts a local proxy (`proxy/openrouter_proxy.py`)
> that fixes structured-output JSON responses from OpenRouter. The proxy runs on port 8100
> and stops when evolution finishes. No manual setup is needed.

### 6. Evaluate results

Benchmark all seed strategies (no evolution required):
```bash
python scripts/evaluate.py --phf phash --all-seeds
python scripts/evaluate.py --phf pdq --all-seeds
python scripts/evaluate.py --phf neuralhash --all-seeds
python scripts/evaluate.py --phf photodna --all-seeds     # requires DLL setup
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

### Typical Workflow

```
download data → start Redis → run evolution (hours) → evaluate evolved programs
                                    ↓
                          gigaevo-core/outputs/
                          (best evolved programs stored in Redis)
```

After evolution, view the best programs directly from Redis:
```bash
# print top-5 by efficiency
python scripts/show_best.py phash

# save top-10 code to out/best_phash/*.py
python scripts/show_best.py phash --top 10 --save

# sort by ASR instead
python scripts/show_best.py phash --metric asr
```

#### Live monitoring during evolution

To watch the best programs update in real time while evolution is running,
open a second terminal and run:

```bash
# refresh every 30s (default) — prints to terminal AND logs to W&B
python scripts/show_best.py phash --watch

# refresh every 10s
python scripts/show_best.py phash --watch 10

# terminal only, no W&B
python scripts/show_best.py phash --watch 30 --no-wandb
```

So the typical two-terminal setup during a run is:

```
Terminal 1:  python run_evohash.py phash --max-generations 100
Terminal 2:  python scripts/show_best.py phash --watch 30
```

In W&B you will see a separate **monitor** run (grouped under the same PHF)
with the following panels updated every refresh cycle:

| W&B key | What it shows |
|---|---|
| `monitor/best_program` | Code of the current best program (HTML) |
| `monitor/best_efficiency` | Efficiency of the best program over time |
| `monitor/best_asr` | ASR of the best program over time |
| `monitor/best_l2` | L2 of the best program over time |
| `monitor/top3` | Table with top-3 programs and their metrics |

### Notebooks

| Notebook | What it does |
|---|---|
| [notebooks/baselines_evaluation.ipynb](notebooks/baselines_evaluation.ipynb) | Runs all baseline attacks and displays metrics (ASR, L2, Efficiency, LPIPS, Time) with bar charts, scatter plots, and heatmaps |

```bash
pip install jupyter matplotlib seaborn pandas
jupyter notebook notebooks/baselines_evaluation.ipynb
```

Set `N_PAIRS = 100` in the first cell for the full benchmark, or keep it at `20` for a quick test.
The notebook is independent of GigaEvo and Redis — it can run in parallel with evolution.

### Troubleshooting

**`FileNotFoundError: data/imagenet_val/ not found`**
→ Run `python scripts/download_dataset.py` first.

**`FileNotFoundError: data/neuralhash_model/seed1.dat`**
→ Model files should be in `data/neuralhash_model/`. They are included in the repo.

**PhotoDNA on macOS: `Docker run failed`**
→ Make sure Docker Desktop is running. The image `evohash-photodna` is built automatically on first compute() call (~1 min). Subsequent calls are fast.

**PhotoDNA on Linux: `[PhotoDNA] Not set up`**
→ Either install Wine and run `setup_photodna()`, or install Docker — the Docker backend is used automatically if Wine Python is not set up.

**`ImportError: No module named 'onnxruntime'`**
→ Install onnxruntime: `pip install onnxruntime` (or `onnxruntime-gpu` for CUDA).

**`Redis database is not empty`**
→ Either flush it (`redis-cli -n 0 FLUSHDB`) or use `--resume` to continue, or use a different DB (`--redis-db 1`).

**`OPENAI_API_KEY not set` warning**
→ Set `OPENROUTER_API_KEY` in your `.env` file.

**W&B run not appearing**
→ Check `WANDB_API_KEY` is set in `.env`. Get a key at https://wandb.ai/authorize.

**`gigaevo-core not found`**
→ Clone the repo into `gigaevo-core/` (see Step 1).

**Web UI doesn't open in the browser**
→ Navigate to `http://localhost:8765` manually. Use `--no-browser` flag to suppress auto-open.

**Web UI: "Frontend ещё не собран"**
→ Run `cd web/frontend && npm install && npm run build` to build the frontend manually.
