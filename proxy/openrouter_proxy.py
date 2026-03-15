"""Lightweight fixup proxy between gigaevo and OpenRouter.

Some models (e.g. GPT-OSS-120b) return a bare JSON **list** when gigaevo
asks for structured output with a pydantic schema that expects an **object**.

This proxy:
  1. Forwards every request to OpenRouter unchanged.
  2. On the way back, inspects chat-completion responses that used
     ``response_format`` with a JSON schema.
  3. If the model returned ``[...]`` where an object was expected, wraps
     the list into ``{"<field>": [...]}`` using the schema to pick the
     right field name.
  4. Everything else passes through untouched.

Usage::

    python proxy/openrouter_proxy.py              # default :8100
    python proxy/openrouter_proxy.py --port 9000  # custom port

Then point your LLM config to ``base_url: http://localhost:8100/api/v1``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from aiohttp import ClientSession, web

# ── Config ────────────────────────────────────────────────────────────────────

UPSTREAM = "https://openrouter.ai"
DEFAULT_PORT = 8100

log = logging.getLogger("openrouter-proxy")


# ── Schema helpers ────────────────────────────────────────────────────────────

def _find_list_field(schema: dict) -> str | None:
    """Return the name of the single root-level list field, or None.

    If the schema has exactly **one** property whose type is ``array``, that
    field is the obvious wrapper target (e.g. ``insights`` for
    ``ProgramInsights``).  If there are zero or multiple array fields we
    cannot safely guess, so we return None and skip the fix.
    """
    props = schema.get("properties", {})
    list_fields: list[str] = []
    for name, spec in props.items():
        # Direct "type": "array"
        if spec.get("type") == "array":
            list_fields.append(name)
            continue
        # anyOf / oneOf containing array (pydantic Optional[list[...]])
        for variant in spec.get("anyOf", []) + spec.get("oneOf", []):
            if variant.get("type") == "array":
                list_fields.append(name)
                break

    if len(list_fields) == 1:
        return list_fields[0]
    return None


def _extract_schema(request_body: dict) -> dict | None:
    """Pull the JSON schema from the request's ``response_format``, if any."""
    rf = request_body.get("response_format")
    if not rf or rf.get("type") != "json_schema":
        return None
    return rf.get("json_schema", {}).get("schema")


def _fix_content(content: str, schema: dict) -> str | None:
    """If *content* is a bare JSON list and the schema has one list field, wrap it.

    Returns the fixed string, or None if no fix was needed/possible.
    """
    stripped = content.strip()
    if not stripped.startswith("["):
        return None  # already an object (or not JSON)

    field = _find_list_field(schema)
    if field is None:
        return None  # can't determine wrapper field

    # Validate it's actually parseable JSON before wrapping
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None  # paranoia check

    wrapped = json.dumps({field: parsed}, ensure_ascii=False)
    log.info("Fixed bare list → wrapped into {%s: [...]}", field)
    return wrapped


# ── Proxy handler ─────────────────────────────────────────────────────────────

async def _proxy_handler(request: web.Request) -> web.Response:
    """Forward request to OpenRouter, optionally fix the response."""
    session: ClientSession = request.app["session"]

    upstream_url = UPSTREAM + request.path_qs
    body = await request.read()

    # Build headers, forwarding auth etc.
    headers = {}
    for hdr in ("Authorization", "Content-Type", "HTTP-Referer",
                "X-Title", "Accept", "Accept-Encoding"):
        val = request.headers.get(hdr)
        if val:
            headers[hdr] = val

    # Parse request body to check for response_format (only for POST)
    schema = None
    request_body = None
    if request.method == "POST" and body:
        try:
            request_body = json.loads(body)
            schema = _extract_schema(request_body)
        except json.JSONDecodeError:
            pass

    # Forward to upstream
    async with session.request(
        request.method,
        upstream_url,
        headers=headers,
        data=body,
    ) as upstream_resp:
        resp_body = await upstream_resp.read()
        resp_headers = {
            k: v for k, v in upstream_resp.headers.items()
            if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")
        }

        # Log and fix chat completion responses
        is_chat = request.path.rstrip("/").endswith("/chat/completions")
        if is_chat and upstream_resp.status == 200:
            resp_body = _log_and_fix_response(resp_body, schema)

        return web.Response(
            status=upstream_resp.status,
            headers=resp_headers,
            body=resp_body,
        )


def _log_and_fix_response(resp_body: bytes, schema: dict | None) -> bytes:
    """Log finish_reason / usage, then fix structured-output content if needed."""
    try:
        data = json.loads(resp_body)
    except json.JSONDecodeError:
        return resp_body

    # Log model, finish_reason, token usage for every chat completion
    model = data.get("model", "?")
    usage = data.get("usage", {})
    choices = data.get("choices", [])

    for choice in choices:
        reason = choice.get("finish_reason", "?")
        if reason == "length":
            log.warning(
                "TRUNCATED response (finish_reason=length) model=%s "
                "prompt_tokens=%s completion_tokens=%s",
                model,
                usage.get("prompt_tokens", "?"),
                usage.get("completion_tokens", "?"),
            )
        else:
            log.debug(
                "model=%s finish_reason=%s completion_tokens=%s",
                model, reason, usage.get("completion_tokens", "?"),
            )

    # Fix structured-output only if a schema was in the request
    if schema is None:
        return resp_body

    modified = False
    for choice in choices:
        msg = choice.get("message", {})
        content = msg.get("content")
        if not content:
            continue

        fixed = _fix_content(content, schema)
        if fixed is not None:
            msg["content"] = fixed
            modified = True

    if modified:
        return json.dumps(data, ensure_ascii=False).encode()
    return resp_body


# ── App setup ─────────────────────────────────────────────────────────────────

async def _on_startup(app: web.Application) -> None:
    app["session"] = ClientSession()
    log.info("Proxy started → upstream %s", UPSTREAM)


async def _on_cleanup(app: web.Application) -> None:
    await app["session"].close()


def create_app() -> web.Application:
    app = web.Application()
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    # Catch-all route — forward everything
    app.router.add_route("*", "/{path:.*}", _proxy_handler)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenRouter fixup proxy")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        stream=sys.stderr,
    )

    web.run_app(create_app(), port=args.port, print=lambda msg: log.info(msg))


if __name__ == "__main__":
    main()
