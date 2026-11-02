"""Optional Ollama transport and per-call disk cache for Role C.

Kept separate from veriquery.llm (owned by the integration/verifier branch).
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import time
import urllib.request
from pathlib import Path


class CachedOllama:
    def __init__(self, model: str, *, base_url="http://localhost:11434",
                 cache_path="runs/sql_calls.sqlite", timeout_seconds=120.0,
                 max_output_tokens=1024, context_tokens=16384, model_digest=None):
        if not model or timeout_seconds <= 0 or max_output_tokens < 1 or context_tokens < 1:
            raise ValueError("Invalid model/client budget")
        self.model, self.base_url = model, base_url.rstrip('/')
        self.model_digest = model_digest
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens, self.context_tokens = max_output_tokens, context_tokens
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.cache = sqlite3.connect(str(cache_path))
        self.cache.execute('CREATE TABLE IF NOT EXISTS sql_calls (key TEXT PRIMARY KEY, response TEXT NOT NULL)')
        self.stats = {"calls":0,"cache_hits":0,"prompt_tokens":0,"completion_tokens":0,"seconds":0.0}

    def __call__(self, prompt):
        body = {"model":self.model,"prompt":prompt,"stream":False,
                "options":{"temperature":0,"seed":6130,"num_predict":self.max_output_tokens,
                           "num_ctx":self.context_tokens}}
        key=hashlib.sha256(json.dumps([self.base_url,self.model_digest,body],sort_keys=True).encode()).hexdigest()
        cached=self.cache.execute('SELECT response FROM sql_calls WHERE key=?',(key,)).fetchone()
        if cached:
            self.stats['cache_hits']+=1
            return cached[0]
        request=urllib.request.Request(self.base_url+'/api/generate',json.dumps(body).encode(),
                                       {'Content-Type':'application/json'})
        start=time.monotonic()
        self.stats['calls']+=1
        try:
            with urllib.request.urlopen(request,timeout=self.timeout_seconds) as response:
                data=json.loads(response.read(8_000_001))
            text=data.get('response')
            if not isinstance(text,str) or not text.strip():
                raise ValueError('Model endpoint returned no text response')
            self.stats['prompt_tokens']+=data.get('prompt_eval_count',0)
            self.stats['completion_tokens']+=data.get('eval_count',0)
            if data.get('done_reason')=='length':
                raise ValueError('Model output exceeded the token limit')
            self.cache.execute('INSERT OR REPLACE INTO sql_calls VALUES (?,?)',(key,text))
            self.cache.commit()
            return text
        finally:
            self.stats['seconds']+=time.monotonic()-start

    def close(self):
        self.cache.close()
