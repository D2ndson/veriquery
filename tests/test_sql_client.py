"""No-network tests for the optional local Ollama transport and cache."""
import io
import json
import urllib.error

import pytest
from veriquery.sql_client import CachedOllama


@pytest.fixture
def transport(monkeypatch):
    calls = []
    replies = []
    def open_(request, timeout):
        calls.append((request, timeout))
        value = replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return io.BytesIO(json.dumps(value).encode())
    monkeypatch.setattr('urllib.request.urlopen', open_)
    return calls, replies


def reply(text='SELECT 1', **kwargs):
    return dict(response=text, done=True, prompt_eval_count=13, eval_count=4, **kwargs)


def test_request_settings_and_cache_hits_do_not_count_tokens_twice(tmp_path, transport):
    calls, replies = transport
    replies.append(reply())
    client = CachedOllama('test-model', cache_path=tmp_path / 'cache.sqlite', max_output_tokens=42, context_tokens=1024, timeout_seconds=7)
    try:
        assert client('prompt') == client('prompt') == 'SELECT 1'
        assert len(calls) == 1
        request, timeout = calls[0]
        payload = json.loads(request.data)
        assert payload['model'] == 'test-model' and payload['prompt'] == 'prompt'
        assert payload['options']['num_predict'] == 42 and payload['options']['num_ctx'] == 1024
        assert payload['stream'] is False and timeout == 7
        assert client.stats['calls'] == 1 and client.stats['cache_hits'] == 1
        assert client.stats['prompt_tokens'] == 13 and client.stats['completion_tokens'] == 4
    finally:
        client.close()


@pytest.mark.parametrize('change', [{'model':'other'}, {'base_url':'http://other:11434'}, {'max_output_tokens':20}, {'context_tokens':1000}, {'model_digest':'new-weights'}])
def test_cache_key_includes_provider_model_and_generation_limits(tmp_path, transport, change):
    calls, replies = transport
    replies.extend([reply('SELECT 1'), reply('SELECT 2')])
    path = tmp_path / 'cache.sqlite'
    first = CachedOllama('m', cache_path=path)
    assert first('same prompt') == 'SELECT 1'
    first.close()
    options = {'model':'m', **change}
    second = CachedOllama(**options, cache_path=path)
    try:
        assert second('same prompt') == 'SELECT 2'
        assert len(calls) == 2 and second.stats['cache_hits'] == 0
    finally:
        second.close()


def test_cache_is_persistent_and_prompt_sensitive(tmp_path, transport):
    calls, replies = transport
    replies.extend([reply('SELECT 1'), reply('SELECT 2')])
    path = tmp_path / 'cache.sqlite'
    a = CachedOllama('m', cache_path=path)
    a('p1'); a.close()
    b = CachedOllama('m', cache_path=path)
    try:
        assert b('p1') == 'SELECT 1'
        assert b('p2') == 'SELECT 2'
        assert len(calls) == 2 and b.stats['cache_hits'] == 1
    finally:
        b.close()


@pytest.mark.parametrize('bad', [{}, {'response':''}, {'response':3}, reply(done_reason='length')])
def test_bad_responses_are_not_cached(tmp_path, transport, bad):
    calls, replies = transport
    replies.extend([bad, reply()])
    client = CachedOllama('m', cache_path=tmp_path / 'cache.sqlite')
    try:
        with pytest.raises(ValueError): client('p')
        assert client('p') == 'SELECT 1'
        assert len(calls) == 2 and client.stats['cache_hits'] == 0
    finally:
        client.close()


def test_http_failure_is_not_cached(tmp_path, transport):
    calls, replies = transport
    replies.extend([urllib.error.URLError('offline'), reply()])
    client = CachedOllama('m', cache_path=tmp_path / 'cache.sqlite')
    try:
        with pytest.raises(urllib.error.URLError): client('p')
        assert client('p') == 'SELECT 1'
        assert client.stats['calls'] == 2 and len(calls) == 2
    finally:
        client.close()


def test_length_failure_still_counts_consumed_tokens(tmp_path, transport):
    _, replies = transport
    replies.append(reply(done_reason='length'))
    client = CachedOllama('m', cache_path=tmp_path / 'cache.sqlite')
    try:
        with pytest.raises(ValueError): client('p')
        assert client.stats['prompt_tokens'] == 13
        assert client.stats['completion_tokens'] == 4
    finally:
        client.close()
