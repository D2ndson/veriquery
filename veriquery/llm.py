"""The team's central LLM client (owner: A).

One object serves every agent:

- `llm(prompt) -> str` for the SQL writer (C) and verifier (E)
- `llm.chat(messages, max_tokens=..., tag=...) -> str` for the clarifier (D)

Every call is cached on disk (ablation reruns repeat the same prompts) and counted per agent:
calls, cache hits, prompt/completion tokens, seconds and estimated cost. Attribute calls made
through the plain callable interface with `with llm.agent("sql_writer"): ...`.

Providers:
- "ollama": a local Ollama server (default; `qwen2.5-coder:7b` fits an 8 GB GPU)
- "openai": any OpenAI-compatible endpoint (vLLM, Ollama's /v1, a hosted API)

`LLM.from_env()` reads VERIQUERY_LLM_PROVIDER, VERIQUERY_LLM_MODEL, VERIQUERY_LLM_BASE_URL,
VERIQUERY_LLM_API_KEY and VERIQUERY_LLM_CACHE.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_BASE_URLS = {"ollama": "http://localhost:11434", "openai": "http://localhost:8000/v1"}


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cost_usd: float = 0.0

    def add(self, other: "LLMStats") -> None:
        for k in self.__dict__:
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def snapshot(self) -> dict:
        return dict(self.__dict__)


def _strip_think(text: str) -> str:
    # Qwen3 / DeepSeek style reasoning block at the start of a reply
    return re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S)


class LLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        provider: str = "ollama",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        num_ctx: int = 16384,
        cache_path: str | Path | None = ".cache/llm_cache.sqlite",
        timeout_s: float = 300.0,
        price_per_mtok: tuple[float, float] = (0.0, 0.0),  # (input, output) USD; 0 for local models
        seed: int = 6130,
        think: bool | None = None,  # Ollama: False turns off Qwen3-style reasoning output
    ):
        if provider not in DEFAULT_BASE_URLS:
            raise ValueError(f"provider must be one of {sorted(DEFAULT_BASE_URLS)}")
        self.model = model
        self.provider = provider
        self.base_url = (base_url or DEFAULT_BASE_URLS[provider]).rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.price_per_mtok = price_per_mtok
        self.seed = seed
        self.think = think
        self.stats = LLMStats()
        self.by_agent: dict[str, LLMStats] = {}
        self._local = threading.local()
        self._lock = threading.Lock()
        self._cache = None
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._cache = sqlite3.connect(str(cache_path), check_same_thread=False)
            self._cache.execute(
                "CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, model TEXT, prompt TEXT, "
                "completion TEXT, prompt_tokens INT, completion_tokens INT, seconds REAL, created REAL)"
            )

    @classmethod
    def from_env(cls, **overrides) -> "LLM":
        env = {
            "provider": os.environ.get("VERIQUERY_LLM_PROVIDER"),
            "model": os.environ.get("VERIQUERY_LLM_MODEL"),
            "base_url": os.environ.get("VERIQUERY_LLM_BASE_URL"),
            "api_key": os.environ.get("VERIQUERY_LLM_API_KEY"),
            "cache_path": os.environ.get("VERIQUERY_LLM_CACHE"),
        }
        return cls(**{**{k: v for k, v in env.items() if v}, **overrides})

    def with_temperature(self, temperature: float, seed: int | None = None) -> "LLM":
        """Same model and cache, different sampling temperature (e.g. for the second route).

        A different `seed` gives a different sample (and cache entry) for the same prompt, which
        candidate voting needs. The copy counts its own usage, so a writer and a checker can be
        measured separately."""
        other = object.__new__(type(self))
        other.__dict__.update(self.__dict__)
        other.temperature = temperature
        other.seed = self.seed if seed is None else seed
        other.stats = LLMStats()
        other.by_agent = {}
        other._local = threading.local()
        return other

    # -------- per-agent accounting --------
    @contextmanager
    def agent(self, name: str):
        """Attribute every call inside the block to `name` (calls via `__call__` have no tag)."""
        previous = getattr(self._local, "agent", None)
        self._local.agent = name
        try:
            yield self
        finally:
            self._local.agent = previous

    def usage(self) -> dict:
        """Totals plus a per-agent breakdown, JSON-friendly."""
        with self._lock:
            return {"total": self.stats.snapshot(), "by_agent": {k: v.snapshot() for k, v in self.by_agent.items()}}

    def snapshot(self) -> dict:
        # Same shape as the clarifier's client: {tag: {calls, cached, prompt_tokens, ...}}
        with self._lock:
            return {k: {**v.snapshot(), "cached": v.cache_hits, "latency_s": v.seconds}
                    for k, v in self.by_agent.items()}

    def _record(self, tag: str, delta: LLMStats) -> None:
        with self._lock:
            self.stats.add(delta)
            self.by_agent.setdefault(tag, LLMStats()).add(delta)

    # -------- calls --------
    def __call__(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    def chat(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None,
             tag: str | None = None, use_cache: bool = True, **_ignored) -> str:
        tag = tag or getattr(self._local, "agent", None) or "untagged"
        max_tokens = max_tokens or self.max_tokens
        temperature = self.temperature if temperature is None else temperature
        key_parts = [self.provider, self.model, self.num_ctx, temperature, max_tokens, messages]
        if self.seed != 6130:  # keep cache keys of the default seed unchanged
            key_parts.append(self.seed)
        if self.think is not None:
            key_parts.append({"think": self.think})
        key = hashlib.sha256(json.dumps(key_parts, sort_keys=True).encode()).hexdigest()
        if use_cache and self._cache is not None:
            with self._lock:
                row = self._cache.execute("SELECT completion FROM calls WHERE key = ?", (key,)).fetchone()
            if row:
                self._record(tag, LLMStats(cache_hits=1))
                return row[0]

        start = time.monotonic()
        if self.provider == "ollama":
            text, pt, ct = self._ollama(messages, max_tokens, temperature)
        else:
            text, pt, ct = self._openai(messages, max_tokens, temperature)
        seconds = time.monotonic() - start
        text = _strip_think(text)
        cost = (pt * self.price_per_mtok[0] + ct * self.price_per_mtok[1]) / 1e6
        self._record(tag, LLMStats(calls=1, prompt_tokens=pt, completion_tokens=ct, seconds=seconds, cost_usd=cost))
        if use_cache and self._cache is not None and text.strip():
            with self._lock:
                self._cache.execute(
                    "INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (key, self.model, json.dumps(messages), text, pt, ct, seconds, time.time()),
                )
                self._cache.commit()
        return text

    def _post(self, url: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.load(resp)

    def _ollama(self, messages, max_tokens, temperature) -> tuple[str, int, int]:
        body = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx, "num_predict": max_tokens,
                        "seed": self.seed},
        }
        if self.think is not None:
            body["think"] = self.think
        data = self._post(f"{self.base_url}/api/chat", body)
        return data.get("message", {}).get("content", ""), data.get("prompt_eval_count", 0), data.get("eval_count", 0)

    def _openai(self, messages, max_tokens, temperature) -> tuple[str, int, int]:
        data = self._post(f"{self.base_url}/chat/completions", {
            "model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "seed": self.seed,
        })
        usage = data.get("usage") or {}
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()


class OllamaLLM(LLM):
    """The verifier branch's original client signature, now backed by the central LLM."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0, num_ctx: int = 8192,
                 base_url: str = "http://localhost:11434", cache_path: str | Path | None = ".cache/llm_cache.sqlite",
                 timeout_s: float = 300.0):
        super().__init__(model, "ollama", base_url, temperature=temperature, num_ctx=num_ctx,
                         cache_path=cache_path, timeout_s=timeout_s, max_tokens=2048)
