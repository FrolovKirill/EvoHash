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
        "efficiency": best.get("efficiency", 0),
        "asr": best.get("asr", 0),
        "l2": best.get("l2", 0),
        "n_queries": best.get("n_queries", 0),
    }
