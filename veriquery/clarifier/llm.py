"""Small OpenAI-compatible LLM client (local vLLM or any API) with a disk cache and token/latency logging."""
import hashlib
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict

from openai import OpenAI, BadRequestError


def _strip_think(text):
    # Qwen3 / DeepSeek style reasoning block at the start of a reply
    return re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S)


class LLM:
    def __init__(self, base_url, api_key, model, cache_path, extra_body=None,
                 default_max_tokens=1024, max_retries=4, name="llm"):
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=900)
        self.model = model
        self.extra_body = extra_body or {}
        self.default_max_tokens = default_max_tokens
        self.max_retries = max_retries
        self.name = name
        self._lock = threading.Lock()
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._db.commit()
        self.stats = defaultdict(lambda: {"calls": 0, "cached": 0, "errors": 0,
                                          "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0})

    def _key(self, messages, max_tokens, temperature, stop):
        blob = json.dumps([self.model, messages, max_tokens, temperature, stop, self.extra_body],
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages, max_tokens=None, temperature=0.0, stop=None, tag="misc", use_cache=True):
        max_tokens = max_tokens or self.default_max_tokens
        k = self._key(messages, max_tokens, temperature, stop)
        if use_cache:
            with self._lock:
                row = self._db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
            if row:
                with self._lock:
                    self.stats[tag]["cached"] += 1
                return row[0]
        last_err = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens,
                    temperature=temperature, stop=stop,
                    extra_body=self.extra_body or None)
                text = _strip_think(resp.choices[0].message.content or "")
                dt = time.time() - t0
                with self._lock:
                    s = self.stats[tag]
                    s["calls"] += 1
                    s["latency_s"] += dt
                    if resp.usage:
                        s["prompt_tokens"] += resp.usage.prompt_tokens or 0
                        s["completion_tokens"] += resp.usage.completion_tokens or 0
                    if use_cache:
                        self._db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (k, text))
                        self._db.commit()
                return text
            except BadRequestError as e:  # e.g. prompt longer than max_model_len -> do not retry
                with self._lock:
                    self.stats[tag]["errors"] += 1
                print(f"[{self.name}] BadRequest ({tag}): {str(e)[:200]}")
                return ""
            except Exception as e:  # network hiccup / server busy -> back off and retry
                last_err = e
                time.sleep(2 * (attempt + 1))
        with self._lock:
            self.stats[tag]["errors"] += 1
        print(f"[{self.name}] failed after retries ({tag}): {str(last_err)[:200]}")
        return ""

    def snapshot(self):
        with self._lock:
            return {k: dict(v) for k, v in self.stats.items()}


def stats_diff(after, before):
    out = {}
    for tag, s in after.items():
        b = before.get(tag, {})
        out[tag] = {k: s[k] - b.get(k, 0) for k in s}
    return out


def extract_json(text):
    """Pull the first JSON object out of an LLM reply (handles ```json fences and chatter)."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    cand = t[start:i + 1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        try:  # common LLM slip: trailing commas
                            return json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
                        except Exception:
                            break
        start = t.find("{", start + 1)
    return None


def extract_sql(text):
    """Get SQL out of ```sqlite / ```sql blocks, falling back to the first SELECT/WITH statement."""
    if not text:
        return ""
    m = re.findall(r"```(?:sqlite|sql)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        cands = [x.strip() for x in m if x.strip()]
        if cands:
            return cands[-1].replace("</t>", "").strip()
    # fenced block opened but never closed
    m = re.search(r"```(?:sqlite|sql)\s*(.*)", text, flags=re.S | re.I)
    if m:
        return m.group(1).replace("</t>", "").strip()
    m = re.search(r"\b(WITH|SELECT)\b.*", text, flags=re.S | re.I)
    return m.group(0).replace("</t>", "").strip() if m else ""
