"""EvoHash runner — launches gigaevo evolution for a chosen PHF.

Usage::

    # Run pHash evolution for 50 generations
    python run_evohash.py phash

    # Run PDQ evolution with custom generation count and LLM
    python run_evohash.py pdq --max-generations 100 --llm openrouter_bandit

    # Pass additional Hydra overrides at the end
    python run_evohash.py phash --max-generations 20 -- redis.db=3 temperature=0.9

OPENROUTER_API_KEY must be set in the environment (or in a .env file in the
project root).

How it works
────────────
gigaevo's ``run.py`` is a Hydra application that reads configs from
``gigaevo-core/config/``.  This script:

  1. Resolves the absolute path of our problem directory.
  2. Sets ``OPENAI_API_KEY = OPENROUTER_API_KEY`` because gigaevo's LLM configs
     read ``OPENAI_API_KEY`` — OpenRouter accepts the same key name.
  3. Launches ``python gigaevo-core/run.py`` as a subprocess with the correct
     Hydra overrides so gigaevo finds our problem and uses OpenRouter.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()
GIGAEVO_DIR = PROJECT_ROOT / "gigaevo-core"
PROBLEMS_DIR = PROJECT_ROOT / "problems"

# ── Supported PHFs ────────────────────────────────────────────────────────────

SUPPORTED_PHFS = ["phash", "pdq", "neuralhash", "photodna"]

# Default LLM config (gigaevo ships openrouter_ensemble.yaml and openrouter_bandit.yaml)
DEFAULT_LLM = "openrouter_gpt_oss"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run EvoHash evolutionary attack discovery for a chosen PHF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evohash.py phash
  python run_evohash.py pdq --max-generations 100 --llm openrouter_bandit
  python run_evohash.py phash -- redis.db=2 temperature=0.8
""",
    )
    p.add_argument(
        "phf",
        choices=SUPPORTED_PHFS,
        help="Target perceptual hash function.",
    )
    p.add_argument(
        "--max-generations",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of evolution generations (default: 50).",
    )
    p.add_argument(
        "--llm",
        default=DEFAULT_LLM,
        metavar="CONFIG",
        help=(
            f"LLM config name from gigaevo-core/config/llm/ "
            f"(default: {DEFAULT_LLM})."
        ),
    )
    p.add_argument(
        "--redis-db",
        type=int,
        default=0,
        metavar="N",
        help="Redis database number (default: 0).  Use different numbers for concurrent runs.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing Redis data instead of starting fresh.",
    )
    return p


def _check_prerequisites(phf: str) -> None:
    """Raise early with a helpful message if prerequisites are missing."""
    if not GIGAEVO_DIR.exists():
        raise SystemExit(
            f"gigaevo-core not found at {GIGAEVO_DIR}.\n"
            "Clone it with:\n"
            "  git clone https://github.com/gigaevo/gigaevo.git gigaevo-core"
        )

    problem_dir = PROBLEMS_DIR / phf
    if not problem_dir.exists():
        raise SystemExit(f"Problem directory not found: {problem_dir}")

    if phf == "photodna":
        raise SystemExit(
            f"'photodna' is not yet implemented.  "
            f"See evohash/phf/photodna.py for details."
        )

    if phf == "neuralhash":
        import platform
        if platform.system() != "Darwin":
            raise SystemExit(
                "NeuralHash requires macOS (uses Apple Vision framework).\n"
                "Install PyObjC: pip install pyobjc-framework-Vision pyobjc-core"
            )


def _build_env(base_env: dict[str, str]) -> dict[str, str]:
    """
    Build the subprocess environment.

    Maps OPENROUTER_API_KEY → OPENAI_API_KEY so gigaevo's LLM configs
    (which use ``${oc.env:OPENAI_API_KEY}``) transparently pick up the
    OpenRouter key.
    """
    env = base_env.copy()

    # Load .env if present (same as gigaevo's run.py does via dotenv)
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        _load_dotenv(dotenv_path, env)

    # Resolve OPENROUTER_API_KEY: .env value takes priority, then os.environ fallback.
    # A value of "None" or empty string in .env is treated as unset.
    openrouter_key = _nonempty(env.get("OPENROUTER_API_KEY"))
    if openrouter_key is None:
        openrouter_key = _nonempty(os.environ.get("OPENROUTER_API_KEY"))
        if openrouter_key:
            env["OPENROUTER_API_KEY"] = openrouter_key

    if openrouter_key and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = openrouter_key
    elif not env.get("OPENAI_API_KEY"):
        print(
            "WARNING: neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set.\n"
            "LLM calls will fail.  Set OPENROUTER_API_KEY in your environment "
            "or in a .env file.",
            file=sys.stderr,
        )

    # Ensure the EvoHash project root is in PYTHONPATH so that gigaevo's
    # exec_runner subprocesses (which inherit this env) can import the
    # evohash package (evohash/phf/, evohash/dataset.py, etc.).
    existing_pp = env.get("PYTHONPATH", "")
    project_root_str = str(PROJECT_ROOT)
    if existing_pp:
        env["PYTHONPATH"] = project_root_str + os.pathsep + existing_pp
    else:
        env["PYTHONPATH"] = project_root_str

    return env


def _nonempty(value: str | None) -> str | None:
    """Return *value* if it is a non-empty string that is not literally 'None', else None."""
    if not value or value.strip().lower() == "none":
        return None
    return value


def _load_dotenv(path: Path, env: dict[str, str]) -> None:
    """Minimal .env loader (no external dependency required)."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in env:  # don't overwrite existing env vars
                env[key] = value


def run(
    phf: str,
    max_generations: int = 50,
    llm: str = DEFAULT_LLM,
    redis_db: int = 0,
    resume: bool = False,
    extra_overrides: list[str] | None = None,
) -> int:
    """
    Launch gigaevo's ``run.py`` for the given PHF.

    Returns the process exit code.
    """
    _check_prerequisites(phf)

    problem_dir = str(PROBLEMS_DIR / phf)

    overrides = list(extra_overrides or [])

    # EvoHash-specific LLM configs live in PROJECT_ROOT/config/llm/.
    # Hydra's --config-dir prepends this path to the config search path,
    # so our configs are found before gigaevo's built-in ones.
    custom_config_dir = PROJECT_ROOT / "config"

    cmd = [
        sys.executable,
        str(GIGAEVO_DIR / "run.py"),
        "--config-dir", str(custom_config_dir),
        f"problem.name={phf}",
        f"problem.dir={problem_dir}",
        f"llm={llm}",
        f"max_generations={max_generations}",
        f"redis.db={redis_db}",
        f"pipeline=with_context",
        f"redis.resume={str(resume).lower()}",
    ] + overrides

    env = _build_env(dict(os.environ))

    print(f"EvoHash: running evolution for '{phf}'")
    print(f"  Problem dir : {problem_dir}")
    print(f"  LLM config  : {llm}")
    print(f"  Generations : {max_generations}")
    print(f"  Redis DB    : {redis_db}")
    print(f"  Resume      : {resume}")
    if overrides:
        print(f"  Extra       : {overrides}")
    print()

    result = subprocess.run(cmd, cwd=str(GIGAEVO_DIR), env=env)
    return result.returncode


def main() -> None:
    parser = build_parser()
    # parse_known_args so that bare Hydra overrides (e.g. temperature=0.9) are
    # passed through as extra_overrides without confusing argparse.
    args, extra = parser.parse_known_args()

    exit_code = run(
        phf=args.phf,
        max_generations=args.max_generations,
        llm=args.llm,
        redis_db=args.redis_db,
        resume=args.resume,
        extra_overrides=extra,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
