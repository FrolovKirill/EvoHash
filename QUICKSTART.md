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

### Web UI pages

**Запуск** — configuration and launch page.
- Select target PHF (pHash, PDQ, NeuralHash, PhotoDNA) and LLM model from dropdown (models loaded from `config/models.yaml`)
- Set number of generations and image pairs per evaluation
- Start / stop evolution with a single button
- "Clear data" button flushes the Redis DB and resets logs
- Shows last run parameters for reference
- Runs `/api/check-env` on load to verify that gigaevo-core, dataset, Redis, and API key are all set up correctly

**Монитор** — live dashboard during evolution.
- Real-time colorized log stream via WebSocket (`/ws/logs`)
- Efficiency and ASR chart updated each generation
- Metric cards for best and latest programs (efficiency, ASR, L2, queries)
- Image grid tabs: source / target / attacked / diff images, auto-refreshed every 2 seconds
- Status bar showing current generation progress

**Результаты** — evolved programs browser.
- Table of all programs stored in Redis, sorted by efficiency
- Click any row to view the full attack code
- Download individual programs as `.py` files

**Бейзлайны** — baseline evaluation.
- Run `scripts/evaluate.py --all-seeds` on all baseline attacks with one click
- Results displayed in a table (ASR, L2, Efficiency, Queries, Time)
- Results saved to `results_baselines.csv` for further analysis

### Web UI API

The backend exposes a REST API at `http://localhost:8765/api/`. Interactive docs available at `/docs` (Swagger UI).

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Current run status, generation progress |
| `/api/run` | POST | Start evolution (accepts PHF, generations, model, etc.) |
| `/api/stop` | POST | Stop running evolution |
| `/api/clear-data` | POST | Flush Redis DB and reset state |
| `/api/programs` | GET | List evolved programs from Redis |
| `/api/metrics` | GET | Best program metrics |
| `/api/metrics-latest` | GET | Latest evaluated program metrics |
| `/api/metrics-history` | GET | Metrics over time (for charts) |
| `/api/grid-image` | GET | Best/latest image grid (PNG) |
| `/api/models` | GET | Available LLM models from `config/models.yaml` |
| `/api/check-env` | GET | Verify prerequisites (Redis, data, API key, gigaevo) |
| `/api/baselines` | GET | Baseline evaluation results |
| `/api/run-baselines` | POST | Run baseline evaluation |
| `/ws/logs` | WebSocket | Live log stream + generation status updates |

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

#### Archive snapshots and LLM analysis

`scripts/archive_manager.py` saves periodic snapshots of the MAP-Elites archive from Redis
to JSON files and optionally classifies each evolved program against known baseline attack
patterns using an LLM (via OpenRouter API).

```bash
# Save a one-off snapshot
python scripts/archive_manager.py save phash

# Auto-save every 5 minutes while evolution runs (Ctrl+C to stop)
python scripts/archive_manager.py watch phash --interval 300

# Auto-save + LLM pattern analysis (classifies code against NES, SimBa, HSJA, etc.)
python scripts/archive_manager.py watch phash --analyze --interval 300

# Use a different model for analysis (default: openai/gpt-oss-120b)
python scripts/archive_manager.py watch phash --analyze --model qwen/qwen3.5-122b-a10b

# Load a snapshot back into Redis
python scripts/archive_manager.py load phash --file snapshots/phash/my_run.json

# Load and overwrite (clears existing keys for this PHF first)
python scripts/archive_manager.py load phash --file snapshots/phash/my_run.json --overwrite

# Show stats about a snapshot (no Redis needed)
python scripts/archive_manager.py info --file snapshots/phash/my_run.json

# Compare two snapshots (added / removed / changed programs)
python scripts/archive_manager.py diff --old snap1.json --new snap2.json
```

The `watch` mode prints a live summary table:
```
================================================================================
  [21:03:06]  Cells: 8 active | 10 total ever | 2 replaced
================================================================================
  # 1  985c01b1  eff= 0.023423  asr=1.00  l2=42.6925  | HSJA(95%)  NES(45%)  SimBa(20%)
  # 2  0e47c465  eff= 0.023074  asr=1.00  l2=43.3394  | SimBa(62%)  HSJA(48%)  NES(15%)
  ...
```

- **Cells active** — programs currently in the MAP-Elites archive
- **Total ever** — every program ID ever observed (tracks replacements)
- **Top-3 baselines** — LLM-assigned similarity to known attack patterns (0-100%)

Snapshots are saved to `snapshots/{phf}/` (gitignored). Analysis results and history
are cached in `snapshots/{phf}/_history.json` to avoid redundant API calls.

For standalone analysis without snapshots, use `scripts/analyze_patterns.py`:

```bash
python scripts/analyze_patterns.py phash
python scripts/analyze_patterns.py pdq --model qwen/qwen3.5-122b-a10b
```

Typical three-terminal setup with analysis:

```
Terminal 1:  python run_evohash.py phash --max-generations 100
Terminal 2:  python scripts/archive_manager.py watch phash --analyze --interval 300
Terminal 3:  python scripts/show_best.py phash --watch 30
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
