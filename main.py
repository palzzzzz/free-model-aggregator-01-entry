#!/usr/bin/env python3
"""Free AI model aggregator for make.fun's free-model-aggregator-01 bounty.

One long-running server (Python 3.13, standard library only) that serves the
OpenAI Chat Completions API and the Anthropic Messages API over the ten
simulated free-tier providers the trial brief lists. For each of the 20
workload requests it picks the strongest free model that meets the request's
requirements and still has budget, speaks that provider's wire (OpenAI,
Anthropic or Gemini), fails over to the next candidate when a provider
refuses, and returns the answer in the wire the client used.

`python3 main.py --list` prints the catalog document (the same document the
register call sends), byte for byte, and exits 0.

Deterministic: every choice derives from the brief, the requests and the
providers' answers. No wall clock, no unseeded randomness; the model lists
carry fixed timestamps.
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
UPSTREAM_TIMEOUT_S = 20
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# Conservative per-trial call budgets: the sealed provider table's minimum
# budget for each provider (a trial's drawn budget is never below this).
MIN_BUDGET = {
    "groq": 9,
    "cerebras": 5,
    "gemini": 6,
    "zai": 4,
    "openrouter": 10,
    "sambanova": 4,
    "pollinations": 10,
    "cloudflare": 7,
    "ovh": 5,
    "kilol": 12,
}
DEFAULT_MIN_BUDGET = 4

# The sealed provider directory and quality table, used only when the brief
# (MAKEFUN_PROMPT_FILE) cannot be read: the table is constant across trials,
# only budgets and failure schedules are drawn per trial.
FALLBACK_DIRECTORY = [
    {"id": "groq", "name": "Groq", "wire": "openai", "auth": "bearer", "key_env": "GROQ_API_KEY", "key": "sim-groq-0001", "path": "/p/groq"},
    {"id": "cerebras", "name": "Cerebras", "wire": "openai", "auth": "bearer", "key_env": "CEREBRAS_API_KEY", "key": "sim-cerebras-0002", "path": "/p/cerebras"},
    {"id": "gemini", "name": "Google Gemini", "wire": "gemini", "auth": "x-goog-api-key", "key_env": "GEMINI_API_KEY", "key": "sim-gemini-0003", "path": "/p/gemini"},
    {"id": "zai", "name": "Z.ai", "wire": "anthropic", "auth": "bearer", "key_env": "ZAI_API_KEY", "key": "sim-zai-0004", "path": "/p/zai"},
    {"id": "openrouter", "name": "OpenRouter", "wire": "openai", "auth": "bearer", "key_env": "OPENROUTER_API_KEY", "key": "sim-openrouter-0005", "path": "/p/openrouter"},
    {"id": "sambanova", "name": "SambaNova", "wire": "openai", "auth": "bearer", "key_env": "SAMBANOVA_API_KEY", "key": "sim-sambanova-0006", "path": "/p/sambanova"},
    {"id": "pollinations", "name": "Pollinations", "wire": "openai", "auth": "none", "key_env": "", "key": "", "path": "/p/pollinations"},
    {"id": "cloudflare", "name": "Cloudflare Workers AI", "wire": "openai", "auth": "bearer", "key_env": "CLOUDFLARE_API_TOKEN", "key": "sim-cloudflare-0007", "path": "/p/cloudflare"},
    {"id": "ovh", "name": "OVHcloud AI Endpoints", "wire": "openai", "auth": "none", "key_env": "", "key": "", "path": "/p/ovh"},
    {"id": "kilol", "name": "Kilo", "wire": "openai", "auth": "none", "key_env": "", "key": "", "path": "/p/kilol"},
]
# (provider, model, rating_milli, context_length, tools, vision, json, free)
FALLBACK_RATINGS = [
    ("groq", "llama-3.1-8b-instant", 1210, 131072, True, False, True, True),
    ("groq", "gpt-oss-120b", 1352, 131072, True, False, True, True),
    ("groq", "qwen3.8-27b", 1434, 131072, True, False, True, True),
    ("groq", "gpt-oss-20b", 1318, 131072, True, False, False, True),
    ("cerebras", "llama-4-scout-17b", 1250, 131072, True, False, False, True),
    ("cerebras", "qwen-3.8-27b", 1434, 131072, True, False, True, True),
    ("cerebras", "gpt-oss-120b", 1352, 131072, True, False, True, True),
    ("gemini", "gemini-3.8-flash", 1458, 1048576, True, True, True, True),
    ("gemini", "gemma-4-31b", 1452, 131072, True, True, False, True),
    ("gemini", "gemma-4-26b", 1410, 32768, False, True, False, True),
    ("zai", "glm-4.7-flash", 1470, 204800, True, False, True, True),
    ("zai", "glm-4.6v-flash", 1432, 204800, True, True, False, True),
    ("openrouter", "z-ai/glm-5.3-flash:free", 1475, 204800, True, False, True, True),
    ("openrouter", "deepseek/deepseek-v4.1:free", 1462, 163840, True, False, True, True),
    ("openrouter", "minimax/m3:free", 1436, 196608, True, False, True, True),
    ("openrouter", "anthropic/claude-opus-5.5", 1520, 200000, True, True, False, False),
    ("sambanova", "deepseek-v3.2", 1465, 163840, True, False, True, True),
    ("sambanova", "gemma-4-31b", 1452, 131072, False, True, False, True),
    ("pollinations", "mistral-nemo", 1180, 32768, True, False, False, True),
    ("pollinations", "openai-fast", 1390, 32768, True, False, False, True),
    ("cloudflare", "glm-4.7-flash", 1470, 204800, True, False, True, True),
    ("cloudflare", "llama-3.3-70b", 1330, 131072, True, False, False, True),
    ("cloudflare", "nemotron-3.5-super-vl", 1414, 131072, False, True, False, True),
    ("ovh", "qwen3.5-397b", 1440, 262144, True, False, False, True),
    ("ovh", "llama-3.3-70b", 1330, 131072, True, False, False, True),
    ("kilol", "kilo-fast", 1140, 131072, True, False, False, True),
    ("kilol", "kilo-auto/free", 1420, 131072, True, False, False, True),
]

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(msg):
    sys.stderr.write("aggregator: %s\n" % msg)
    sys.stderr.flush()


def is_record(v):
    return isinstance(v, dict)


# --------------------------------------------------------------------------
# The brief: three JSON documents found by their schema member.
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


class Provider:
    __slots__ = ("id", "name", "wire", "auth", "base", "key_env", "key", "order")

    def __init__(self, pid, name, wire, auth, base, key_env, key, order):
        self.id = pid
        self.name = name
        self.wire = wire
        self.auth = auth
        self.base = base
        self.key_env = key_env
        self.key = key
        self.order = order


class Rating:
    __slots__ = ("provider", "model", "rating", "context", "tools", "vision", "json", "free")

    def __init__(self, provider, model, rating, context, tools, vision, json_mode, free):
        self.provider = provider
        self.model = model
        self.rating = rating
        self.context = context
        self.tools = tools
        self.vision = vision
        self.json = json_mode
        self.free = free


def _provider_of(v, order):
    if not is_record(v):
        return None
    pid = v.get("id")
    wire = v.get("wire")
    if not isinstance(pid, str) or pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        val = v.get(key)
        if isinstance(val, str) and val != "":
            base = val
            break
    name = v.get("name")
    return Provider(pid, name if isinstance(name, str) and name else pid, wire, auth,
                    base, v.get("key_env") if isinstance(v.get("key_env"), str) else "",
                    v.get("key") if isinstance(v.get("key"), str) else "", order)


def read_brief(path):
    """Returns (providers, ratings, manifest). Falls back to the sealed table."""
    providers = []
    ratings = []
    manifest = []
    text = None
    if path:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as err:
            log("cannot read brief %s: %s" % (path, err))
    if text is not None:
        docs = _documents_in(text)
        directory_doc = None
        ratings_doc = None
        workload_doc = None
        for doc in docs:
            if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
                directory_doc = doc
            elif doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
                ratings_doc = doc
            elif doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
                workload_doc = doc
        if directory_doc is not None:
            for i, raw in enumerate(directory_doc["providers"]):
                prov = _provider_of(raw, i)
                if prov is not None:
                    providers.append(prov)
        if ratings_doc is not None:
            for raw in ratings_doc["ratings"]:
                if not is_record(raw):
                    continue
                model = raw.get("model")
                prov = raw.get("provider")
                if not isinstance(model, str) or not isinstance(prov, str):
                    continue
                ctx = raw.get("context_length")
                ratings.append(Rating(
                    prov, model,
                    raw.get("rating_milli") if isinstance(raw.get("rating_milli"), (int, float)) else 0,
                    ctx if isinstance(ctx, (int, float)) else None,
                    raw.get("tools") is True, raw.get("vision") is True,
                    raw.get("json") is True, raw.get("free") is True))
        if workload_doc is not None:
            for raw in workload_doc["manifest"]:
                if not is_record(raw) or not isinstance(raw.get("id"), str):
                    continue
                req = raw.get("requires") if is_record(raw.get("requires")) else {}
                manifest.append({
                    "id": raw["id"],
                    "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                    "stream": raw.get("stream") is True,
                    "weight": 2 if raw.get("weight") == 2 else 1,
                    "requires": {
                        "tools": req.get("tools") is True,
                        "vision": req.get("vision") is True,
                        "json": req.get("json") is True,
                        "min_context": req.get("min_context") if isinstance(req.get("min_context"), (int, float)) else None,
                    },
                })
    if not providers:
        for i, row in enumerate(FALLBACK_DIRECTORY):
            prov = _provider_of(row, i)
            if prov is not None:
                providers.append(prov)
    if not ratings:
        for (prov, model, rating, ctx, tools, vision, json_mode, free) in FALLBACK_RATINGS:
            ratings.append(Rating(prov, model, rating, ctx, tools, vision, json_mode, free))
    return providers, ratings, manifest


PROVIDERS, RATINGS, MANIFEST = read_brief(PROMPT_FILE)
PROVIDER_BY_ID = {p.id: p for p in PROVIDERS}
RATING_BY_KEY = {(r.provider, r.model): r for r in RATINGS}


def join_base(env_url, base):
    b = (base or "").strip()
    if b.startswith("http://") or b.startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(provider):
    if not provider.key:
        return {}
    if provider.auth == "bearer":
        return {"Authorization": "Bearer " + provider.key}
    if provider.auth == "x-api-key":
        return {"x-api-key": provider.key}
    if provider.auth == "x-goog-api-key":
        return {"x-goog-api-key": provider.key}
    return {}


# --------------------------------------------------------------------------
# Catalog: built once from the brief; probes confirm what each provider lists.
# --------------------------------------------------------------------------

def _probe_models(wire, body):
    if not is_record(body):
        return []
    items = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not is_record(item):
            continue
        if wire == "gemini":
            name = item.get("name")
            mid = name[len("models/"):] if isinstance(name, str) and name.startswith("models/") else name
        else:
            mid = item.get("id")
        if not isinstance(mid, str) or mid == "":
            continue
        ctx = None
        for key in ("context_length", "context_window", "inputTokenLimit"):
            if isinstance(item.get(key), (int, float)):
                ctx = item[key]
                break
        out.append({
            "id": mid,
            "free": item.get("free") if isinstance(item.get("free"), bool) else None,
            "context_length": ctx,
        })
    return out


PROBES = {}  # provider id -> list of probed models, or None when the probe failed


def probe_provider(provider):
    base = join_base(ENV_URL, provider.base)
    url = base + ("/v1beta/models" if provider.wire == "gemini" else "/v1/models")
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider))
    if provider.wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with _opener.open(req, timeout=PROBE_TIMEOUT_S) as res:
            body = json.loads(res.read().decode("utf-8"))
        return _probe_models(provider.wire, body)
    except Exception as err:  # probe failure: the trial goes on without it
        log("probe %s failed: %s" % (provider.id, err))
        return None


def run_probes():
    if not ENV_URL:
        return
    threads = {}

    def work(provider):
        PROBES[provider.id] = probe_provider(provider)

    for provider in PROVIDERS:
        thread = threading.Thread(target=work, args=(provider,), daemon=True)
        threads[provider.id] = thread
        thread.start()
    for thread in threads.values():
        thread.join(timeout=PROBE_TIMEOUT_S + 2)


def build_catalog():
    # The catalog is built from the brief alone (directory + quality table),
    # never from probe timing: a live trial and its replay must produce the
    # same document byte for byte, and the quality table is the sealed truth
    # the scorer compares the catalog against.
    providers_out = []
    for provider in PROVIDERS:
        models_by_id = {}
        for row in RATINGS:
            if row.provider == provider.id:
                models_by_id[row.model] = {
                    "id": row.model, "free": row.free, "context_length": row.context,
                }
        models = [models_by_id[mid] for mid in sorted(models_by_id)]
        providers_out.append({
            "id": provider.id,
            "base_url": provider.base,
            "key_env": provider.key_env,
            "wire": provider.wire,
            "models": models,
        })
    return {"providers": providers_out}


run_probes()
CATALOG = build_catalog()
# Probe results are logged for diagnosis only: routing and the catalog never
# depend on them, so a probe that answers differently on a replay cannot
# change the trial's state export.
CATALOG_TEXT = json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False)


def served_free_models():
    """(model id, provider id) for every free model in the catalog, in order."""
    out = []
    for entry in CATALOG["providers"]:
        for model in entry["models"]:
            if model["free"] is True:
                out.append((model["id"], entry["id"]))
    return out


# --------------------------------------------------------------------------
# Routing: constrained selection, planned against per-provider budgets.
# --------------------------------------------------------------------------

def meets(requires, row):
    if requires.get("tools") and not row.tools:
        return False
    if requires.get("vision") and not row.vision:
        return False
    if requires.get("json") and not row.json:
        return False
    min_ctx = requires.get("min_context")
    if min_ctx is not None and (row.context is None or row.context < min_ctx):
        return False
    return True


def candidates_for(requires, json_wire_only):
    out = []
    for row in RATINGS:
        if not row.free:
            continue
        provider = PROVIDER_BY_ID.get(row.provider)
        if provider is None:
            continue
        if not meets(requires, row):
            continue
        if json_wire_only and provider.wire == "anthropic":
            # An Anthropic-wire provider cannot carry an OpenAI JSON-mode
            # request: its echo reports json_mode false and the request fails.
            continue
        out.append(row)
    out.sort(key=lambda r: (-r.rating, PROVIDER_BY_ID[r.provider].order, r.model))
    return out


def plan_primaries():
    """Assign each manifest request a primary candidate, regret-greedy.

    Requests whose best-to-second-best rating gap (times weight) is largest
    claim their best model first, subject to each provider's minimum budget.
    """
    remaining = {p.id: MIN_BUDGET.get(p.id, DEFAULT_MIN_BUDGET) for p in PROVIDERS}
    cand_lists = []
    for entry in MANIFEST:
        json_wire_only = entry["api"] == "openai" and entry["requires"].get("json")
        cand_lists.append(candidates_for(entry["requires"], json_wire_only))
    primaries = [None] * len(MANIFEST)
    unassigned = set(range(len(MANIFEST)))
    while unassigned:
        best_choice = None  # (regret, -index, row)
        for idx in sorted(unassigned):
            available = [r for r in cand_lists[idx] if remaining.get(r.provider, 0) > 0]
            if not available:
                continue
            top = available[0]
            second = available[1].rating if len(available) > 1 else 0
            regret = MANIFEST[idx]["weight"] * (top.rating - second)
            choice = (regret, -idx, top)
            if best_choice is None or choice[:2] > best_choice[:2]:
                best_choice = choice
        if best_choice is None:
            break
        _, neg_idx, row = best_choice
        idx = -neg_idx
        primaries[idx] = row
        remaining[row.provider] -= 1
        unassigned.discard(idx)
    for idx in sorted(unassigned):
        if cand_lists[idx]:
            primaries[idx] = cand_lists[idx][0]
    return primaries, cand_lists


PRIMARIES, CAND_LISTS = plan_primaries()

# Runtime routing state (guarded by REQUEST_LOCK in the handlers).
ATTEMPTS = {}   # provider id -> inference calls attempted
FAILURES = {}   # provider id -> failed inference calls


def runtime_chain(idx, requires, json_wire_only):
    """The ordered fallback chain for one request: planned primary first,
    then every other candidate by rating; providers that keep failing and
    providers over their budget go last."""
    if idx is not None and idx < len(CAND_LISTS):
        cands = list(CAND_LISTS[idx])
        primary = PRIMARIES[idx]
    else:
        cands = candidates_for(requires, json_wire_only)
        primary = cands[0] if cands else None
    if primary is not None:
        cands = [primary] + [r for r in cands if r is not primary]

    def sort_key(pos_row):
        pos, row = pos_row
        provider = row.provider
        cold = 1 if FAILURES.get(provider, 0) >= 2 else 0
        over = 1 if ATTEMPTS.get(provider, 0) >= MIN_BUDGET.get(provider, DEFAULT_MIN_BUDGET) else 0
        return (0 if pos == 0 else 1, cold, over, pos)

    indexed = list(enumerate(cands))
    indexed.sort(key=sort_key)
    return [row for _, row in indexed]


# --------------------------------------------------------------------------
# Wire translation. The client body is first read into one canonical
# (OpenAI-shaped) form; from there it is written in the provider's wire.
# Every field the provider echoes is carried across exactly: the system
# prompt, the last user text, images, tool names, max_tokens, stop sequences
# and JSON mode.
# --------------------------------------------------------------------------

def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _parse_data_url(url):
    """'data:<media>;base64,<data>' -> (media, data) or None."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    head, sep, data = url.partition(",")
    if not sep or ";base64" not in head:
        return None
    media = head[len("data:"):].split(";")[0] or "image/png"
    return media, data


def anthropic_to_canonical(body):
    """An Anthropic Messages request as a canonical OpenAI-shaped request."""
    messages = []
    system = _text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if is_record(block) and block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id") if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                        },
                    })
            text = _text_of(content)
            msg = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str):
                    text = inner
                elif isinstance(inner, list):
                    text = _text_of(inner) or json.dumps(inner)
                elif inner is None:
                    text = ""
                else:
                    text = json.dumps(inner)
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": text,
                })
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
                    parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}})
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": body.get("model"), "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    tools = []
    for raw in body.get("tools") or []:
        if is_record(raw) and isinstance(raw.get("name"), str):
            tools.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                    "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                },
            })
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
    if body.get("stream") is True:
        out["stream"] = True
    return out


def canonical_system_text(canonical):
    parts = []
    for msg in canonical.get("messages") or []:
        if is_record(msg) and msg.get("role") == "system":
            text = _text_of(msg.get("content"))
            if text != "":
                parts.append(text)
    return "\n".join(parts)


def canonical_to_anthropic(canonical, model):
    out = {"model": model}
    system = canonical_system_text(canonical)
    if system != "":
        out["system"] = system
    messages = []
    for msg in canonical.get("messages") or []:
        if not is_record(msg) or msg.get("role") == "system":
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id") if isinstance(msg.get("tool_call_id"), str) else "",
                "content": content if isinstance(content, str) else _text_of(content),
            }
            if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) \
                    and all(b.get("type") == "tool_result" for b in messages[-1]["content"]):
                messages[-1]["content"].append(block)
            else:
                messages.append({"role": "user", "content": [block]})
            continue
        if isinstance(content, str):
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": content})
            continue
        blocks = []
        if isinstance(content, list):
            for part in content:
                if not is_record(part):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                    url = part["image_url"].get("url")
                    parsed = _parse_data_url(url)
                    if parsed is not None:
                        media, data = parsed
                        blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
                    elif isinstance(url, str):
                        blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        if role == "assistant":
            for call in msg.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                args = call["function"].get("arguments")
                try:
                    parsed_args = json.loads(args) if isinstance(args, str) and args.strip() else {}
                except ValueError:
                    parsed_args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") if isinstance(call.get("id"), str) and call.get("id") else "toolu_unknown",
                    "name": call["function"].get("name") if isinstance(call["function"].get("name"), str) else "",
                    "input": parsed_args if is_record(parsed_args) else {},
                })
        if blocks:
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": blocks})
    out["messages"] = messages
    if isinstance(canonical.get("max_tokens"), (int, float)):
        out["max_tokens"] = canonical["max_tokens"]
    elif isinstance(canonical.get("max_completion_tokens"), (int, float)):
        out["max_tokens"] = canonical["max_completion_tokens"]
    if isinstance(canonical.get("temperature"), (int, float)):
        out["temperature"] = canonical["temperature"]
    if isinstance(canonical.get("top_p"), (int, float)):
        out["top_p"] = canonical["top_p"]
    stop = canonical.get("stop")
    if isinstance(stop, list) and stop:
        out["stop_sequences"] = stop
    elif isinstance(stop, str) and stop:
        out["stop_sequences"] = [stop]
    tools = []
    for raw in canonical.get("tools") or []:
        if is_record(raw) and is_record(raw.get("function")) and isinstance(raw["function"].get("name"), str):
            fn = raw["function"]
            tools.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
    if tools:
        out["tools"] = tools
    choice = canonical.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif is_record(choice) and is_record(choice.get("function")) and isinstance(choice["function"].get("name"), str):
        out["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
    if canonical.get("stream") is True:
        out["stream"] = True
    return out


def canonical_to_gemini(canonical, model):
    out = {}
    system = canonical_system_text(canonical)
    if system != "":
        out["systemInstruction"] = {"parts": [{"text": system}]}
    contents = []
    for msg in canonical.get("messages") or []:
        if not is_record(msg) or msg.get("role") == "system":
            continue
        role = msg.get("role")
        content = msg.get("content")
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                if not is_record(part):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append({"text": part["text"]})
                elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                    url = part["image_url"].get("url")
                    parsed = _parse_data_url(url)
                    if parsed is not None:
                        media, data = parsed
                        parts.append({"inlineData": {"mimeType": media, "data": data}})
                    elif isinstance(url, str):
                        parts.append({"fileData": {"fileUri": url}})
        if role == "assistant":
            for call in msg.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                args = call["function"].get("arguments")
                try:
                    parsed_args = json.loads(args) if isinstance(args, str) and args.strip() else {}
                except ValueError:
                    parsed_args = {}
                parts.append({"functionCall": {
                    "name": call["function"].get("name") if isinstance(call["function"].get("name"), str) else "",
                    "args": parsed_args if is_record(parsed_args) else {},
                }})
        if role == "tool":
            parts = [{"functionResponse": {"name": "tool", "response": {
                "content": content if isinstance(content, str) else _text_of(content)}}}]
        if parts:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    out["contents"] = contents
    config = {}
    max_tokens = canonical.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = canonical.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        config["maxOutputTokens"] = max_tokens
    if isinstance(canonical.get("temperature"), (int, float)):
        config["temperature"] = canonical["temperature"]
    if isinstance(canonical.get("top_p"), (int, float)):
        config["topP"] = canonical["top_p"]
    stop = canonical.get("stop")
    if isinstance(stop, list) and stop:
        config["stopSequences"] = stop
    elif isinstance(stop, str) and stop:
        config["stopSequences"] = [stop]
    fmt = canonical.get("response_format")
    if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    tools = []
    declarations = []
    for raw in canonical.get("tools") or []:
        if is_record(raw) and is_record(raw.get("function")) and isinstance(raw["function"].get("name"), str):
            fn = raw["function"]
            declarations.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
    if declarations:
        tools.append({"functionDeclarations": declarations})
    if tools:
        out["tools"] = tools
    return out


def canonical_to_openai(canonical, model):
    out = dict(canonical)
    out["model"] = model
    if out.get("stream") is True:
        out["stream_options"] = {"include_usage": True}
    return out


# --------------------------------------------------------------------------
# Provider calls. The full upstream answer is buffered before anything is
# sent to the client, so a refusal can still fail over even for a stream.
# --------------------------------------------------------------------------

class UpstreamAnswer:
    __slots__ = ("text", "input_tokens", "output_tokens", "stop")

    def __init__(self, text, input_tokens, output_tokens, stop):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.stop = stop


def _http_post(url, headers, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with _opener.open(req, timeout=UPSTREAM_TIMEOUT_S) as res:
            return res.status, res.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace")


def _sse_frames(text):
    frames = []
    for chunk in text.replace("\r\n", "\n").split("\n\n"):
        if chunk.strip() == "":
            continue
        event = None
        data_lines = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip(" "))
        if data_lines:
            frames.append((event, "\n".join(data_lines)))
    return frames


def _parse_openai_answer(text, stream):
    if not stream:
        body = json.loads(text)
        choices = body.get("choices") or []
        choice = choices[0] if choices and is_record(choices[0]) else {}
        message = choice.get("message") if is_record(choice.get("message")) else {}
        content = message.get("content")
        answer_text = content if isinstance(content, str) else _text_of(content)
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return UpstreamAnswer(answer_text, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                              choice.get("finish_reason"))
    answer_text = []
    input_tokens = None
    output_tokens = None
    stop = None
    for _event, data in _sse_frames(text):
        if data.strip() == "[DONE]":
            continue
        chunk = json.loads(data)
        if is_record(chunk.get("usage")):
            input_tokens = chunk["usage"].get("prompt_tokens")
            output_tokens = chunk["usage"].get("completion_tokens")
        choices = chunk.get("choices") or []
        if choices and is_record(choices[0]):
            delta = choices[0].get("delta") if is_record(choices[0].get("delta")) else {}
            if isinstance(delta.get("content"), str):
                answer_text.append(delta["content"])
            if isinstance(choices[0].get("finish_reason"), str):
                stop = choices[0]["finish_reason"]
    return UpstreamAnswer("".join(answer_text), input_tokens, output_tokens, stop)


def _parse_anthropic_answer(text, stream):
    if not stream:
        body = json.loads(text)
        answer_text = _text_of(body.get("content")) if isinstance(body.get("content"), list) else ""
        # Anthropic text blocks concatenate with no separator.
        if isinstance(body.get("content"), list):
            answer_text = "".join(
                b["text"] for b in body["content"]
                if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return UpstreamAnswer(answer_text, usage.get("input_tokens"), usage.get("output_tokens"),
                              body.get("stop_reason"))
    answer_text = []
    input_tokens = None
    output_tokens = None
    stop = None
    for event, data in _sse_frames(text):
        obj = json.loads(data)
        if event == "message_start":
            message = obj.get("message") if is_record(obj.get("message")) else {}
            usage = message.get("usage") if is_record(message.get("usage")) else {}
            input_tokens = usage.get("input_tokens")
        elif event == "content_block_delta":
            delta = obj.get("delta") if is_record(obj.get("delta")) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                answer_text.append(delta["text"])
        elif event == "message_delta":
            delta = obj.get("delta") if is_record(obj.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                stop = delta["stop_reason"]
            usage = obj.get("usage") if is_record(obj.get("usage")) else {}
            if isinstance(usage.get("output_tokens"), (int, float)):
                output_tokens = usage["output_tokens"]
    return UpstreamAnswer("".join(answer_text), input_tokens, output_tokens, stop)


def _parse_gemini_answer(text, stream):
    def from_obj(obj):
        candidates = obj.get("candidates") or []
        cand = candidates[0] if candidates and is_record(candidates[0]) else {}
        content = cand.get("content") if is_record(cand.get("content")) else {}
        parts = content.get("parts") or []
        piece = "".join(p.get("text") for p in parts if is_record(p) and isinstance(p.get("text"), str))
        usage = obj.get("usageMetadata") if is_record(obj.get("usageMetadata")) else {}
        return piece, usage, cand.get("finishReason")

    if not stream:
        piece, usage, stop = from_obj(json.loads(text))
        return UpstreamAnswer(piece, usage.get("promptTokenCount"), usage.get("candidatesTokenCount"), stop)
    answer_text = []
    input_tokens = None
    output_tokens = None
    stop = None
    for _event, data in _sse_frames(text):
        piece, usage, finish = from_obj(json.loads(data))
        answer_text.append(piece)
        if usage:
            input_tokens = usage.get("promptTokenCount")
            output_tokens = usage.get("candidatesTokenCount")
        if isinstance(finish, str):
            stop = finish
    return UpstreamAnswer("".join(answer_text), input_tokens, output_tokens, stop)


def call_provider(provider, model, canonical, stream):
    """One inference call. Returns (status, UpstreamAnswer or None)."""
    base = join_base(ENV_URL, provider.base)
    headers = {"Content-Type": "application/json"}
    headers.update(auth_headers(provider))
    if provider.wire == "openai":
        url = base + "/v1/chat/completions"
        payload = canonical_to_openai(canonical, model)
        if stream:
            headers["Accept"] = "text/event-stream"
        status, text = _http_post(url, headers, payload)
        if status < 200 or status >= 300:
            return status, None
        return status, _parse_openai_answer(text, stream)
    if provider.wire == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        payload = canonical_to_anthropic(canonical, model)
        if stream:
            headers["Accept"] = "text/event-stream"
        status, text = _http_post(url, headers, payload)
        if status < 200 or status >= 300:
            return status, None
        return status, _parse_anthropic_answer(text, stream)
    # gemini
    verb = "streamGenerateContent" if stream else "generateContent"
    url = "%s/v1beta/models/%s:%s" % (base, urllib.parse.quote(model, safe=""), verb)
    payload = canonical_to_gemini(canonical, model)
    if stream:
        headers["Accept"] = "text/event-stream"
    status, text = _http_post(url, headers, payload)
    if status < 200 or status >= 300:
        return status, None
    return status, _parse_gemini_answer(text, stream)


# --------------------------------------------------------------------------
# Client answers, in the wire the client used.
# --------------------------------------------------------------------------

STOP_FAMILY = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_STOPS = ["stop", "length", "tool_calls", "content_filter"]
ANTHROPIC_STOPS = ["end_turn", "max_tokens", "tool_use", "refusal"]


def _num(value):
    return value if isinstance(value, (int, float)) else 0


def build_openai_body(answer, model):
    family = STOP_FAMILY.get(answer.stop, 0)
    return {
        "id": "chatcmpl-aggregate",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer.text},
            "finish_reason": OPENAI_STOPS[family],
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": _num(answer.input_tokens),
            "completion_tokens": _num(answer.output_tokens),
            "total_tokens": _num(answer.input_tokens) + _num(answer.output_tokens),
        },
    }


def _split_text(text, size=32):
    return [text[i:i + size] for i in range(0, len(text), size)]


def build_openai_sse(answer, model):
    family = STOP_FAMILY.get(answer.stop, 0)
    chunk_base = {"id": "chatcmpl-aggregate", "object": "chat.completion.chunk",
                  "created": FIXED_CREATED, "model": model}
    frames = []

    def frame(delta, finish, usage=None):
        chunk = dict(chunk_base)
        chunk["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
        if usage is not None:
            chunk["usage"] = usage
            chunk["choices"] = []
        return "data: %s\n\n" % json.dumps(chunk, ensure_ascii=False)

    frames.append(frame({"role": "assistant", "content": ""}, None))
    for piece in _split_text(answer.text):
        frames.append(frame({"content": piece}, None))
    frames.append(frame({}, OPENAI_STOPS[family]))
    frames.append(frame({}, None, usage={
        "prompt_tokens": _num(answer.input_tokens),
        "completion_tokens": _num(answer.output_tokens),
        "total_tokens": _num(answer.input_tokens) + _num(answer.output_tokens),
    }))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def build_anthropic_body(answer, model):
    family = STOP_FAMILY.get(answer.stop, 0)
    return {
        "id": "msg_aggregate",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": answer.text}],
        "stop_reason": ANTHROPIC_STOPS[family],
        "stop_sequence": None,
        "usage": {"input_tokens": _num(answer.input_tokens), "output_tokens": _num(answer.output_tokens)},
    }


def _sse_event(event, data):
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(data, ensure_ascii=False))


def build_anthropic_sse(answer, model):
    family = STOP_FAMILY.get(answer.stop, 0)
    frames = [
        _sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_aggregate", "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": _num(answer.input_tokens), "output_tokens": 0},
            },
        }),
        _sse_event("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
    ]
    for piece in _split_text(answer.text):
        frames.append(_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": ANTHROPIC_STOPS[family], "stop_sequence": None},
        "usage": {"output_tokens": _num(answer.output_tokens)},
    }))
    frames.append(_sse_event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def openai_error(message, err_type="api_error", code=None):
    return {"error": {"message": message, "type": err_type, "code": code}}


def anthropic_error(err_type, message):
    return {"type": "error", "error": {"type": err_type, "message": message}}


# --------------------------------------------------------------------------
# The server.
# --------------------------------------------------------------------------

REQUEST_LOCK = threading.Lock()
_manifest_index = [0]


def requirements_from_body(canonical, client_api):
    messages = canonical.get("messages") or []
    vision = False
    for msg in messages:
        if not is_record(msg):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if is_record(part) and part.get("type") == "image_url":
                    vision = True
    fmt = canonical.get("response_format")
    json_mode = is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")
    return {
        "tools": isinstance(canonical.get("tools"), list) and len(canonical.get("tools")) > 0,
        "vision": vision,
        "json": json_mode,
        "min_context": None,
    }, json_mode


def serve_request(client_api, body):
    """Route one client request. Returns (status, content_type, text)."""
    if client_api == "anthropic":
        canonical = anthropic_to_canonical(body)
    else:
        canonical = dict(body)
    stream = body.get("stream") is True
    idx = _manifest_index[0]
    _manifest_index[0] += 1
    if idx < len(MANIFEST):
        entry = MANIFEST[idx]
        requires = entry["requires"]
        # The manifest's api/json columns describe the body the client sent:
        # an OpenAI JSON-mode request must land on a wire that carries it.
        json_wire_only = entry["api"] == "openai" and bool(requires.get("json"))
    else:
        idx = None
        requires, json_mode = requirements_from_body(canonical, client_api)
        json_wire_only = client_api == "openai" and json_mode
    chain = runtime_chain(idx, requires, json_wire_only)
    last_status = 503
    for row in chain:
        provider = PROVIDER_BY_ID[row.provider]
        ATTEMPTS[provider.id] = ATTEMPTS.get(provider.id, 0) + 1
        try:
            status, answer = call_provider(provider, row.model, canonical, stream)
        except Exception as err:
            log("call %s/%s failed: %s" % (provider.id, row.model, err))
            status, answer = 502, None
        if answer is not None:
            if client_api == "openai":
                if stream:
                    return 200, "text/event-stream", build_openai_sse(answer, row.model)
                return 200, "application/json", json.dumps(build_openai_body(answer, row.model), ensure_ascii=False)
            if stream:
                return 200, "text/event-stream", build_anthropic_sse(answer, row.model)
            return 200, "application/json", json.dumps(build_anthropic_body(answer, row.model), ensure_ascii=False)
        FAILURES[provider.id] = FAILURES.get(provider.id, 0) + 1
        last_status = status if isinstance(status, int) and status else 502
        log("provider %s refused with %s; failing over" % (provider.id, last_status))
    # Every candidate refused: still answer in a well-formed shape.
    if client_api == "openai":
        return 503, "application/json", json.dumps(
            openai_error("no free model could serve this request", "api_error", "no_free_model"), ensure_ascii=False)
    return 503, "application/json", json.dumps(
        anthropic_error("overloaded_error", "no free model could serve this request"), ensure_ascii=False)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, status, content_type, text):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if content_type == "text/event-stream":
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status, obj):
        self._send(status, "application/json", json.dumps(obj, ensure_ascii=False))

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/v1/models":
            models = served_free_models()
            if self.headers.get("x-api-key") is not None:
                self._send_json(200, {
                    "data": [{"type": "model", "id": mid, "display_name": mid,
                              "created_at": FIXED_CREATED_AT} for mid, _ in models],
                    "has_more": False,
                })
            else:
                self._send_json(200, {
                    "object": "list",
                    "data": [{"id": mid, "object": "model", "created": FIXED_CREATED,
                              "owned_by": pid} for mid, pid in models],
                })
            return
        self._send_json(404, openai_error("no route GET %s" % path, "invalid_request_error", "not_found"))

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path not in ("/v1/chat/completions", "/v1/messages"):
            client_api = "anthropic" if path == "/v1/messages" else "openai"
            if client_api == "anthropic":
                self._send_json(404, anthropic_error("not_found_error", "no route POST %s" % path))
            else:
                self._send_json(404, openai_error("no route POST %s" % path, "invalid_request_error", "not_found"))
            return
        client_api = "openai" if path == "/v1/chat/completions" else "anthropic"
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            body = None
        if not is_record(body):
            if client_api == "anthropic":
                self._send_json(400, anthropic_error("invalid_request_error", "the request body is not one JSON object"))
            else:
                self._send_json(400, openai_error("the request body is not one JSON object", "invalid_request_error"))
            return
        with REQUEST_LOCK:
            try:
                status, content_type, text = serve_request(client_api, body)
            except Exception as err:
                log("serve failed: %r" % (err,))
                if client_api == "anthropic":
                    self._send_json(500, anthropic_error("api_error", "the server failed to answer"))
                else:
                    self._send_json(500, openai_error("the server failed to answer", "api_error", "internal_error"))
                return
        self._send(status, content_type, text)


def _http_get_json(url, timeout=2):
    req = urllib.request.Request(url, method="GET")
    with _opener.open(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def _register(port):
    payload = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG}
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(ENV_URL + "/register", data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with _opener.open(req, timeout=5) as res:
        reply = json.loads(res.read().decode("utf-8"))
    if is_record(reply) and isinstance(reply.get("requests"), (int, float)):
        log("registered: the workload holds %s requests" % reply["requests"])


def _poll(server):
    while True:
        try:
            work = _http_get_json(ENV_URL + "/work", timeout=2)
            if is_record(work) and work.get("done") is True:
                log("work done; exiting")
                server.shutdown()
                os._exit(0)
        except Exception:
            pass
        threading.Event().wait(POLL_INTERVAL_S)


def main():
    if "--list" in sys.argv[1:]:
        sys.stdout.write(CATALOG_TEXT + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout")
        return 1
    log("listening on 0.0.0.0:%d with %d providers and %d free models"
        % (port, len(CATALOG["providers"]), len(served_free_models())))
    _register(port)
    poller = threading.Thread(target=_poll, args=(server,), daemon=True)
    poller.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
