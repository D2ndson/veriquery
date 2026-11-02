"""Central LLM client: caching, per-agent accounting and both providers (no server needed)."""
import pytest

from veriquery.llm import LLM, OllamaLLM


def scripted(monkeypatch, llm, reply="SELECT 1", tokens=(10, 2)):
    calls = []

    def post(url, body):
        calls.append((url, body))
        if llm.provider == "ollama":
            return {"message": {"content": reply}, "prompt_eval_count": tokens[0], "eval_count": tokens[1]}
        return {"choices": [{"message": {"content": reply}}],
                "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]}}

    monkeypatch.setattr(llm, "_post", post)
    return calls


def test_cache_and_per_agent_usage(tmp_path, monkeypatch):
    llm = LLM(cache_path=tmp_path / "c.sqlite", price_per_mtok=(1.0, 2.0))
    calls = scripted(monkeypatch, llm)
    with llm.agent("sql_writer"):
        assert llm("q") == "SELECT 1"
        assert llm("q") == "SELECT 1"  # cached
    assert llm.chat([{"role": "user", "content": "x"}], tag="clarifier") == "SELECT 1"
    assert len(calls) == 2
    u = llm.usage()
    assert u["by_agent"]["sql_writer"]["calls"] == 1 and u["by_agent"]["sql_writer"]["cache_hits"] == 1
    assert u["by_agent"]["clarifier"]["prompt_tokens"] == 10
    assert u["total"]["cost_usd"] == pytest.approx(2 * (10 * 1.0 + 2 * 2.0) / 1e6)


def test_cache_survives_restart_and_keys_on_temperature(tmp_path, monkeypatch):
    path = tmp_path / "c.sqlite"
    first = LLM(cache_path=path)
    scripted(monkeypatch, first)
    first("q")
    second = LLM(cache_path=path)
    calls = scripted(monkeypatch, second)
    second("q")
    assert calls == []
    hot = second.with_temperature(0.3)
    hot("q")  # different temperature, so not a cache hit
    assert len(calls) == 1 and hot.stats.calls == 1 and second.stats.calls == 0


def test_openai_compatible_provider(tmp_path, monkeypatch):
    llm = LLM(model="m", provider="openai", base_url="http://h/v1", api_key="k", cache_path=None)
    calls = scripted(monkeypatch, llm, reply="<think>hmm</think>answer")
    assert llm("q") == "answer"
    assert calls[0][0] == "http://h/v1/chat/completions" and calls[0][1]["model"] == "m"


def test_from_env_and_legacy_signature(monkeypatch):
    monkeypatch.setenv("VERIQUERY_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VERIQUERY_LLM_MODEL", "qwen3")
    llm = LLM.from_env(cache_path=None)
    assert (llm.provider, llm.model) == ("openai", "qwen3")
    legacy = OllamaLLM("m", temperature=0.2, cache_path=None)
    assert legacy.provider == "ollama" and legacy.temperature == 0.2
    with pytest.raises(ValueError):
        LLM(provider="nope")
