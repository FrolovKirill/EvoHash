"""LLM-based pattern analyzer for evolved attack programs.

Fetches evolved programs from the Redis MAP-Elites archive, sends each
program's code to an LLM, and classifies how similar it is to known
baseline attack patterns.

Usage:
    python scripts/analyze_patterns.py phash
    python scripts/analyze_patterns.py phash --top 20
    python scripts/analyze_patterns.py pdq --db 1 --model openai/gpt-oss-120b
    python scripts/analyze_patterns.py phash --save          # save JSON report
    python scripts/analyze_patterns.py phash --program-id abc123  # analyze one
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import redis
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Baseline pattern descriptions (extracted from initial_programs/)
# ---------------------------------------------------------------------------

BASELINE_PATTERNS = {
    "NES": (
        "Natural Evolution Strategy. Estimates pseudo-gradient of Hamming distance "
        "using antithetic (±) random Gaussian perturbations in pixel space, then "
        "takes a gradient descent step. Key signatures: antithetic sampling "
        "(x + σv, x − σv), gradient = Σ(d_pos − d_neg)·v, pixel-space updates."
    ),
    "SimBa": (
        "Simple Black-box Attack using DCT basis vectors. Coordinate descent in "
        "DCT domain: randomly samples DCT basis directions, evaluates ±step, "
        "applies whichever reduces hash distance. Key signatures: explicit DCT "
        "basis construction, per-direction greedy accept/reject."
    ),
    "HSJA": (
        "HopSkipJump Attack. Decision-based attack starting from a known colliding "
        "point (target image). Binary search to find boundary, then estimates "
        "boundary gradient via Monte-Carlo sampling of orthogonal random vectors. "
        "Key signatures: binary_search along source→colliding line, boundary "
        "gradient estimation, starts from target image."
    ),
    "Prokos": (
        "Frequency-targeted iterative attack (Prokos et al., USENIX 2023). "
        "Perturbs in DCT coefficient space with a frequency weight mask favoring "
        "low-to-mid frequencies. Key signatures: DCT/IDCT transforms, "
        "frequency_weight_mask with cutoff_ratio, gradient estimation in DCT domain."
    ),
    "AtkScopes": (
        "Multi-scale patch-based attack. Divides image into overlapping patches "
        "at multiple resolutions, estimates per-patch hash sensitivity, then "
        "concentrates NES-style gradient attacks on most sensitive patches. "
        "Key signatures: patch_slices at multiple scales, sensitivity estimation, "
        "top-k patch selection."
    ),
    "ZO-SignSGD": (
        "Zeroth-Order Sign SGD. Estimates gradient via forward finite differences "
        "with random Gaussian directions, then applies sign(gradient) * lr as "
        "fixed-magnitude update. Key signatures: forward-only (not antithetic) "
        "gradient estimation, np.sign(grad) update step, L-inf style perturbation."
    ),
    "RandomNoise": (
        "Random Gaussian noise baseline. Samples i.i.d. Gaussian noise, adds to "
        "original image, keeps best sample. No gradient estimation at all. "
        "Key signatures: simple noise sampling loop, no gradient computation, "
        "pure random search."
    ),
}

# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert in adversarial attacks on perceptual hash functions.
You will be given:
1. The source code of an evolved attack program.
2. Descriptions of known baseline attack patterns.

Your task: analyze the evolved code and determine which INDIVIDUAL baseline
pattern(s) it most resembles. Consider the core algorithmic approach, not
superficial code style differences.
Score each baseline INDEPENDENTLY. Do NOT combine baselines (no "NES+HSJA" etc).
If the code uses ideas from multiple baselines, list each one separately.

Respond with valid JSON only (no markdown fences), using this schema:
{
  "primary_pattern": "<baseline name or 'Novel'>",
  "primary_similarity": <0.0 to 1.0>,
  "secondary_pattern": "<baseline name or 'None'>",
  "secondary_similarity": <0.0 to 1.0>,
  "novel_elements": ["<list of techniques not found in any baseline>"],
  "summary": "<1-2 sentence description of the attack strategy>"
}

Similarity scale:
- 1.0 = essentially identical algorithm
- 0.7-0.9 = same core algorithm with modifications
- 0.4-0.6 = borrows key ideas but significantly different
- 0.1-0.3 = shares only superficial similarities
- 0.0 = completely unrelated
"""


def _build_user_prompt(code: str) -> str:
    baselines_text = "\n\n".join(
        f"**{name}:** {desc}" for name, desc in BASELINE_PATTERNS.items()
    )
    return (
        f"## Known Baseline Patterns\n\n{baselines_text}\n\n"
        f"## Evolved Attack Code\n\n```python\n{code}\n```\n\n"
        f"Analyze this evolved code and classify it against the baselines above."
    )


def _load_api_key() -> str:
    """Load OpenRouter API key from .env or environment."""
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        for line in dotenv_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY=") and not line.endswith("..."):
                return line.split("=", 1)[1].strip()
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Error: set OPENROUTER_API_KEY in .env or environment", file=sys.stderr)
        sys.exit(1)
    return key


def analyze_program(
    client: OpenAI,
    code: str,
    model: str,
) -> dict:
    """Send a single program to the LLM for pattern classification."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(code)},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    content = resp.choices[0].message.content
    if not content:
        return {"raw_response": "", "error": "Empty response from LLM"}
    content = content.strip()
    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_response": content, "error": "Failed to parse JSON"}


# ---------------------------------------------------------------------------
# Redis + main
# ---------------------------------------------------------------------------

def fetch_programs(r: redis.Redis, phf: str) -> list[dict]:
    """Fetch all completed programs from Redis."""
    pattern = f"{phf}:program:*"
    keys = list(r.scan_iter(pattern, count=1000))
    programs = []
    for key in keys:
        raw = r.get(key)
        if not raw:
            continue
        try:
            prog = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if prog.get("state") != "done" or not prog.get("code") or not prog.get("metrics"):
            continue
        programs.append({
            "id": prog.get("id", "?"),
            "metrics": prog["metrics"],
            "code": prog["code"],
        })
    return programs


def print_result(rank: int, prog: dict, analysis: dict) -> None:
    sep = "-" * 72
    m = prog["metrics"]
    print(f"\n{sep}")
    print(
        f"  #{rank}  id={prog['id'][:8]}...  "
        f"eff={m.get('efficiency', 0):.6f}  "
        f"asr={m.get('asr', 0):.2f}  "
        f"l2={m.get('l2', 0):.4f}"
    )
    if "error" in analysis:
        print(f"  [LLM parse error] {analysis.get('raw_response', '')[:200]}")
    else:
        pri = analysis.get("primary_pattern", "?")
        pri_sim = analysis.get("primary_similarity", 0)
        sec = analysis.get("secondary_pattern", "None")
        sec_sim = analysis.get("secondary_similarity", 0)
        summary = analysis.get("summary", "")
        novel = analysis.get("novel_elements", [])
        print(f"  Pattern:  {pri} ({pri_sim:.0%})")
        if sec and sec != "None":
            print(f"  Also:     {sec} ({sec_sim:.0%})")
        if novel:
            print(f"  Novel:    {', '.join(novel)}")
        print(f"  Summary:  {summary}")
    print(sep)


def main() -> None:
    p = argparse.ArgumentParser(description="LLM-based pattern analyzer for evolved attacks")
    p.add_argument("phf", help="PHF name (phash, pdq, neuralhash)")
    p.add_argument("--top", type=int, default=10, help="How many top programs to analyze")
    p.add_argument("--metric", default="efficiency", help="Sort metric")
    p.add_argument("--db", type=int, default=0, help="Redis DB number")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=6379)
    p.add_argument("--model", default="openai/gpt-oss-120b", help="OpenRouter model ID")
    p.add_argument("--save", action="store_true", help="Save JSON report to out/analysis_{phf}/")
    p.add_argument("--program-id", default=None, help="Analyze a specific program by ID prefix")
    args = p.parse_args()

    api_key = _load_api_key()
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    r = redis.Redis(host=args.host, port=args.port, db=args.db, decode_responses=True)
    programs = fetch_programs(r, args.phf)
    if not programs:
        print(f"No programs found for '{args.phf}' in Redis db={args.db}")
        sys.exit(1)

    # Filter by ID if specified
    if args.program_id:
        programs = [p for p in programs if p["id"].startswith(args.program_id)]
        if not programs:
            print(f"No program found with ID prefix '{args.program_id}'")
            sys.exit(1)

    # Sort
    reverse = args.metric not in ("l2",)
    programs.sort(key=lambda x: x["metrics"].get(args.metric, -1e9), reverse=reverse)
    programs = programs[: args.top]

    print(f"Analyzing {len(programs)} programs for '{args.phf}' using {args.model}...")

    results = []
    for rank, prog in enumerate(programs, 1):
        print(f"\n[{rank}/{len(programs)}] Analyzing {prog['id'][:8]}...", end="", flush=True)
        analysis = analyze_program(client, prog["code"], args.model)
        print(" done.")
        print_result(rank, prog, analysis)
        results.append({
            "id": prog["id"],
            "metrics": prog["metrics"],
            "analysis": analysis,
        })
        # Small delay to avoid rate limiting
        if rank < len(programs):
            time.sleep(1)

    # Summary table
    print(f"\n{'='*72}")
    print(f"  PATTERN DISTRIBUTION (top-{len(results)} by {args.metric})")
    print(f"{'='*72}")
    pattern_counts: dict[str, list] = {}
    for res in results:
        a = res["analysis"]
        pat = a.get("primary_pattern", "Unknown")
        pattern_counts.setdefault(pat, []).append(res["metrics"].get("efficiency", 0))

    for pat, effs in sorted(pattern_counts.items(), key=lambda x: -len(x[1])):
        avg_eff = sum(effs) / len(effs) if effs else 0
        print(f"  {pat:20s}  count={len(effs):3d}  avg_efficiency={avg_eff:.6f}")

    if args.save:
        out_dir = Path("out") / f"analysis_{args.phf}"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
