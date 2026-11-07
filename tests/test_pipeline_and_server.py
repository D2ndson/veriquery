"""Pipeline helpers, runner planning and the scripted-demo server. The server/runner parts run in a
subprocess with their own VERIQUERY_HOME so they never touch the project's data folders."""
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _sub(code: str, home: Path, *args: str):
    shutil.copytree(ROOT / "configs", home / "configs", dirs_exist_ok=True)
    env = {**os.environ, "VERIQUERY_HOME": str(home), "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code), *args], env=env, capture_output=True, text=True, timeout=300)


def test_lf_score():
    from veriquery.pipeline.mine import lf_score
    assert lf_score(False, 1.0, 0, 0) == -1.0
    assert lf_score(True, 1.0, 0, 1) == 0.0
    assert lf_score(True, 1.0, 2, 0) == pytest.approx(0.7225, abs=1e-3)


def test_plan_from_sql():
    from veriquery.pipeline.plans import plan_from_sql
    p = plan_from_sql("SELECT COUNT(DISTINCT a.id) FROM a JOIN b ON a.id=b.a_id WHERE b.x > 3")
    assert p and "a" in p and "b" in p
    assert plan_from_sql("SELEC ((") is None


def test_runner_dry_run_lists_stages(tmp_path):
    r = _sub("""
        import sys
        from veriquery.pipeline import runner
        raise SystemExit(runner.main(['run', '--dry-run', '--allow-no-gpu']))
    """, tmp_path)
    assert r.returncode == 0, r.stderr[-800:]
    for stage in ("sft_train", "mine", "dpo_train", "report"):
        assert stage in r.stdout


def test_scripted_server_catches_fanout(tmp_path):
    r = _sub("""
        import json, time
        from veriquery.datasets import shoptrap
        from veriquery.config import path
        shoptrap.write_all(path('data/shoptrap'))
        from fastapi.testclient import TestClient
        from veriquery.server.app import app
        c = TestClient(app)
        assert c.get('/api/health').status_code == 200
        rid = c.post('/api/runs', json={'question': 'What is the total shipping fee on orders that contain at least one Electronics item?',
                                         'mode': 'scripted', 'latency': 0.0}).json()['run_id']
        types = []
        with c.stream('GET', f'/api/runs/{rid}/events') as s:
            for line in s.iter_lines():
                if line.startswith('data:'):
                    types.append(json.loads(line[5:])['type'])
        st = c.get(f'/api/runs/{rid}').json()
        assert st['status'] == 'done', st
        assert st['result']['answers'][0]['repairs'] >= 1
        print('OK', len(types))
    """, tmp_path)
    assert r.returncode == 0 and "OK" in r.stdout, (r.stdout + r.stderr)[-1200:]
