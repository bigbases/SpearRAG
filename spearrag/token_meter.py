"""Process-wide LLM token meter.

WHY
---
LinearRAG's Table 2 reports "Token Consumption (x10^6): Prompt / Completion"
alongside indexing and retrieval time. Their LinearRAG row is 0 / 0, because
they count only the METHOD's own LLM usage (graph construction + retrieval) and
exclude answer generation, which every method pays equally. To produce the same
table we need per-phase token accounting for every system, including inside
vendor code we do not want to edit (HippoRAG2, GFM-RAG, EHRAG, LinearRAG).

HOW
---
`install()` monkeypatches the OpenAI SDK's chat-completions entry point and
accumulates `response.usage` into a per-phase counter. Because every system here
ultimately goes through that SDK, one patch captures our shared generation/judge
harness AND the vendors' internal OpenIE / NER / rerank calls.

`set_phase()` labels whatever runs next. The eval scripts wrap their stages:

    token_meter.install()
    token_meter.set_phase("index");      rag.index(passages)
    token_meter.set_phase("retrieval");  rag.retrieve(questions)
    token_meter.set_phase("generation"); preds = generate_answers(msgs)
    token_meter.set_phase("judge");      gpt_acc = compute_gpt_acc_scores(...)
    out["token_usage"] = token_meter.report()

`report()["method_only"]` is index + retrieval -- the number comparable to
LinearRAG's Table 2. `report()["total"]` includes generation and judging.

A phase that never calls an LLM simply reports zeros, which is the point:
SpearRAG, LinearRAG and EHRAG must show 0 / 0 for method_only, and that is then a
measured fact rather than an assumption.

Thread-safe: the harness fans generation out over a ThreadPoolExecutor.
"""
from __future__ import annotations

import threading
from collections import defaultdict

_lock = threading.Lock()
_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "calls": 0})
_phase = "other"
_installed = False

# Phases counted as "the method's own cost" in LinearRAG's Table 2 sense.
METHOD_PHASES = ("index", "retrieval")


def set_phase(name: str) -> None:
    """Label every LLM call made from now until the next set_phase()."""
    global _phase
    with _lock:
        _phase = name


def record(prompt_tokens: int, completion_tokens: int, phase: str | None = None) -> None:
    with _lock:
        c = _counts[phase or _phase]
        c["prompt"] += int(prompt_tokens or 0)
        c["completion"] += int(completion_tokens or 0)
        c["calls"] += 1


def install() -> bool:
    """Patch the OpenAI SDK so every chat completion is metered. Idempotent.

    Patching `create` alone is not enough. Structured-output calls go through
    `parse`, and GFM-RAG's query NER (langchain_openai.ChatOpenAI with a response
    schema) takes exactly that path -- which is why its index/retrieval phases
    reported zero tokens on all four datasets while its generation and judge
    phases were counted normally. Async clients have their own class, and both
    sync and async are covered here.
    """
    global _installed
    if _installed:
        return True

    targets = []
    try:
        from openai.resources.chat import completions as _c
        targets.append(_c.Completions)
        async_cls = getattr(_c, "AsyncCompletions", None)
        if async_cls is not None:
            targets.append(async_cls)
    except Exception:
        return False
    try:  # older SDKs put structured output under beta
        from openai.resources.beta.chat import completions as _b
        targets.append(_b.Completions)
        async_b = getattr(_b, "AsyncCompletions", None)
        if async_b is not None:
            targets.append(async_b)
    except Exception:
        pass  # not present in newer SDKs, where parse lives on Completions

    try:
        # The Responses API. openai>=3 exposes client.responses, and
        # langchain_openai>=1.x routes ChatOpenAI through it rather than through
        # chat.completions -- which is why GFM-RAG's OpenIE reported 0 tokens on
        # 658 chunks that demonstrably took 17 minutes of API latency
        # (measured 2026-09-22, openai 3.7.0 / langchain_openai 1.6.0).
        from openai.resources import responses as _r
        for _name in ("Responses", "AsyncResponses"):
            _cls = getattr(_r, _name, None)
            if _cls is not None:
                targets.append(_cls)
    except Exception:
        pass  # older SDKs have no Responses API

    def _count(resp) -> None:
        try:
            usage = getattr(resp, "usage", None)
            if usage is None:
                return
            # Two field vocabularies. Chat Completions reports prompt_tokens /
            # completion_tokens; the Responses API reports input_tokens /
            # output_tokens on a ResponseUsage. Reading only the first pair
            # silently records 0 for every Responses call.
            prompt = getattr(usage, "prompt_tokens", None)
            if prompt is None:
                prompt = getattr(usage, "input_tokens", 0)
            completion = getattr(usage, "completion_tokens", None)
            if completion is None:
                completion = getattr(usage, "output_tokens", 0)
            record(prompt, completion)
        except Exception:
            pass  # metering must never break a run

    def _wrap_sync(orig):
        def _metered(self, *args, **kwargs):
            resp = orig(self, *args, **kwargs)
            _count(resp)
            return resp
        return _metered

    def _wrap_async(orig):
        async def _metered(self, *args, **kwargs):
            resp = await orig(self, *args, **kwargs)
            _count(resp)
            return resp
        return _metered

    patched = 0
    for cls in targets:
        is_async = cls.__name__.startswith("Async")
        for name in ("create", "parse"):
            orig = getattr(cls, name, None)
            if orig is None or getattr(orig, "_token_metered", False):
                continue
            wrapped = (_wrap_async if is_async else _wrap_sync)(orig)
            wrapped._token_metered = True  # type: ignore[attr-defined]
            setattr(cls, name, wrapped)
            patched += 1

    patched += _install_langchain()
    _installed = patched > 0
    return _installed


def _install_langchain() -> int:
    """Meter langchain_openai, which never touches the methods patched above.

    langchain_openai calls `client.responses.with_raw_response.create(...)` (and
    the chat.completions equivalent). `with_raw_response` is a SEPARATE wrapper
    class built by to_raw_response_wrapper(), so patching Responses.create or
    Completions.create misses it entirely -- GFM-RAG reported 0 tokens for 658
    OpenIE chunks that took 17 minutes of real API latency, twice, before this
    was found (2026-09-22).

    Rather than chase SDK internals, this hooks BaseChatOpenAI._generate and
    reads `usage_metadata` off the returned message, which langchain populates
    on every call and exposes as public API. Returns the number of methods
    patched so install() can report honestly.
    """
    try:
        from langchain_openai.chat_models.base import BaseChatOpenAI
    except Exception:
        return 0

    def _from_result(result) -> None:
        try:
            for gen in getattr(result, "generations", []) or []:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if usage:
                    record(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        except Exception:
            pass

    count = 0
    for name in ("_generate", "_agenerate"):
        orig = getattr(BaseChatOpenAI, name, None)
        if orig is None or getattr(orig, "_token_metered", False):
            continue
        if name == "_agenerate":
            async def wrapper(self, *args, _orig=orig, **kwargs):
                result = await _orig(self, *args, **kwargs)
                _from_result(result)
                return result
        else:
            def wrapper(self, *args, _orig=orig, **kwargs):  # type: ignore[misc]
                result = _orig(self, *args, **kwargs)
                _from_result(result)
                return result
        wrapper._token_metered = True  # type: ignore[attr-defined]
        setattr(BaseChatOpenAI, name, wrapper)
        count += 1
    return count


def report() -> dict:
    """Per-phase totals plus the two aggregates used in the efficiency table."""
    with _lock:
        per_phase = {k: dict(v) for k, v in _counts.items()}
    method = {"prompt": 0, "completion": 0, "calls": 0}
    total = {"prompt": 0, "completion": 0, "calls": 0}
    for name, c in per_phase.items():
        for k in ("prompt", "completion", "calls"):
            total[k] += c[k]
            if name in METHOD_PHASES:
                method[k] += c[k]
    return {
        "per_phase": per_phase,
        # index + retrieval only -- comparable to LinearRAG Table 2
        "method_only": method,
        # everything, including the shared generation + judge harness
        "total": total,
        "note": "method_only = index+retrieval LLM usage (LinearRAG Table 2 convention); "
                "generation and judging are excluded there because every system pays them equally.",
    }


def reset() -> None:
    with _lock:
        _counts.clear()
