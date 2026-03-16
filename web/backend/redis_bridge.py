"""Read metrics and programs from Redis (real or fake)."""
import json
from typing import Any

_redis_client = None
_fake_client = None


def get_client(host: str = "localhost", port: int = 6380, db: int = 0):
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=1)
        client.ping()
        _redis_client = client
        return client
    except Exception:
        return get_fake_client()


def get_fake_client():
    global _fake_client
    if _fake_client is None:
        try:
            import fakeredis
            _fake_client = fakeredis.FakeRedis()
        except ImportError:
            return None
    return _fake_client


def reset_client():
    global _redis_client
    _redis_client = None


def get_programs(phf: str, redis_port: int = 6380, top_n: int = 20) -> list[dict[str, Any]]:
    client = get_client(port=redis_port)
    if client is None:
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
                    programs.append(data)
            except Exception:
                continue
        programs.sort(key=lambda x: float(x.get("efficiency", -1)), reverse=True)
        return programs[:top_n]
    except Exception:
        return []


def get_metrics_history(phf: str, redis_port: int = 6380) -> list[dict[str, Any]]:
    client = get_client(port=redis_port)
    if client is None:
        return []
    try:
        key = f"{phf}:metrics_history"
        raw = client.get(key)
        if raw:
            return json.loads(raw)
        return []
    except Exception:
        return []


def get_best_metrics(phf: str, redis_port: int = 6380) -> dict[str, Any]:
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
