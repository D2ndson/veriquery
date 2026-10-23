"""Local LLM client (Ollama) with a disk cache for every call.

Stand-in until A picks the team's LLM: anything with the same `__call__(prompt) -> str`
works with the verifier. The plan requires caching every call, because ablation reruns
otherwise repeat the same prompts.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "qwen2.5-coder:7b"


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def snapshot(self) -> dict:
        return dict(self.__dict__)


class OllamaLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        base_url: str = "http://localhost:11434",
        cache_path: str | Path | None = ".cache/llm_cache.sqlite",
        timeout_s: float = 300.0,
    ):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.stats = LLMStats()
        self._lock = threading.Lock()
        self._cache = None
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._cache = sqlite3.connect(str(cache_path), check_same_thread=False)
            self._cache.execute(
                "CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, model TEXT, prompt TEXT, "
                "completion TEXT, prompt_tokens INT, completion_tokens INT, seconds REAL, created REAL)"
            )

    def with_temperature(self, temperature: float) -> "OllamaLLM":
        """Same model and cache, different sampling temperature (e.g. for the second route)."""
        other = object.__new__(OllamaLLM)
        other.__dict__.update(self.__dict__)
        other.temperature = temperature
        other.stats = LLMStats()  # count this client's tokens separately
        return other

    def _key(self, prompt: str) -> str:
        raw = json.dumps([self.model, self.temperature, self.num_ctx, prompt])
        return hashlib.sha256(raw.encode()).hexdigest()

    def __call__(self, prompt: str) -> str:
        key = self._key(prompt)
        if self._cache is not None:
            with self._lock:
                row = self._cache.execute("SELECT completion FROM calls WHERE key = ?", (key,)).fetchone()
            if row:
                self.stats.cache_hits += 1
                return row[0]

        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }).encode()
        req = urllib.request.Request(f"{self.base_url}/api/generate", body, {"Content-Type": "application/json"})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.load(resp)
        seconds = time.monotonic() - start
        text = data.get("response", "")
        pt, ct = data.get("prompt_eval_count", 0), data.get("eval_count", 0)

        self.stats.calls += 1
        self.stats.prompt_tokens += pt
        self.stats.completion_tokens += ct
        self.stats.seconds += seconds
        if self._cache is not None:
            with self._lock:
                self._cache.execute(
                    "INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (key, self.model, prompt, text, pt, ct, seconds, time.time()),
                )
                self._cache.commit()
        return text
