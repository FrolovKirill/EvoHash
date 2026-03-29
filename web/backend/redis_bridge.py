"""Read metrics and programs from Redis."""
import json
from typing import Any

_redis_client = None


def get_client(host: str = "localhost", port: int = 6379, db: int = 0):
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    import redis
    client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=2)
    client.ping()
    _redis_client = client
    return client


def reset_client():
    global _redis_client
    _redis_client = None


def _flatten_program(data: dict) -> dict:
    """Flatten nested gigaevo program structure for the frontend.

    gigaevo stores: {"id", "code", "metrics": {"efficiency": ..., "asr": ...}, "state", ...}
    Frontend expects metrics at the top level.
    """
    flat = {
        "id": data.get("id", ""),
        "code": data.get("code", ""),
        "name": data.get("name", ""),
        "state": data.get("state", ""),
        "atomic_counter": data.get("atomic_counter", 0),
    }
    metrics = data.get("metrics", {})
    if isinstance(metrics, dict):
        flat.update(metrics)
    lineage = data.get("lineage", {})
    if isinstance(lineage, dict):
        flat["generation"] = lineage.get("generation")
        parents = lineage.get("parents", [])
        flat["parent_id"] = parents[0] if parents else None
    return flat


def get_programs(phf: str, redis_port: int = 6379, top_n: int = 20) -> list[dict[str, Any]]:
    try:
        client = get_client(port=redis_port)
    except Exception:
        return []
    try:
        pattern = f"{phf}:program:*"
        keys = client.keys(pattern)
        programs = []
        for key in keys:
            try:
                raw = client.get(key)
                if raw:
                    data = json.loads(raw)
                    programs.append(_flatten_program(data))
            except Exception:
                continue
        programs.sort(key=lambda x: float(x.get("efficiency", -1)), reverse=True)
        return programs[:top_n]
    except Exception:
        return []


def get_metrics_history(phf: str, redis_port: int = 6379) -> list[dict[str, Any]]:
    try:
        client = get_client(port=redis_port)
    except Exception:
        return []
    try:
        key = f"{phf}:metrics_history"
        raw = client.get(key)
        if raw:
            return json.loads(raw)
        return []
    except Exception:
        return []


def get_best_metrics(phf: str, redis_port: int = 6379) -> dict[str, Any]:
    programs = get_programs(phf, redis_port, top_n=1)
    if not programs:
        return {}
    best = programs[0]
    return {
        "efficiency": best.get("efficiency"),
        "asr": best.get("asr"),
        "l2": best.get("l2"),
        "n_queries": best.get("n_queries"),
    }


def get_latest_metrics(phf: str, redis_port: int = 6379) -> dict[str, Any]:
    """Return metrics of the most recently evaluated program (highest atomic_counter)."""
    try:
        client = get_client(port=redis_port)
    except Exception:
        return {}
    try:
        pattern = f"{phf}:program:*"
        keys = client.keys(pattern)
        latest = None
        latest_counter = -1
        for key in keys:
            try:
                raw = client.get(key)
                if raw:
                    data = json.loads(raw)
                    # Skip programs that haven't been evaluated yet
                    metrics = data.get("metrics")
                    if not isinstance(metrics, dict) or metrics.get("efficiency") is None:
                        continue
                    counter = data.get("atomic_counter", 0)
                    if counter > latest_counter:
                        latest_counter = counter
                        latest = _flatten_program(data)
            except Exception:
                continue
        if not latest:
            return {}
        return {
            "efficiency": latest.get("efficiency"),
            "asr": latest.get("asr"),
            "l2": latest.get("l2"),
            "n_queries": latest.get("n_queries"),
        }
    except Exception:
        return {}


def build_metrics_history(phf: str, redis_port: int = 6379) -> list[dict[str, Any]]:
    """Build chart data from all programs in Redis, grouped by generation (running best)."""
    programs = get_programs(phf, redis_port, top_n=10000)
    if not programs:
        return []
    by_gen: dict[int, dict] = {}
    for p in programs:
        gen = p.get("generation")
        if gen is None:
            continue
        gen = int(gen)
        eff = float(p.get("efficiency", -1))
        if gen not in by_gen or eff > float(by_gen[gen].get("efficiency", -1)):
            by_gen[gen] = p
    history = []
    best_eff = -1e9
    best_asr = 0.0
    for gen in sorted(by_gen.keys()):
        p = by_gen[gen]
        eff = float(p.get("efficiency", 0))
        asr = float(p.get("asr", 0))
        if eff > best_eff:
            best_eff = eff
            best_asr = asr
        history.append({"gen": gen, "efficiency": best_eff, "asr": best_asr})
    return history


