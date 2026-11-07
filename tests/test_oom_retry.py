from types import SimpleNamespace
from pathlib import Path

from veriquery.pipeline.runner import _oom_retry


class FakeOOM(Exception):
    pass


FakeOOM.__name__ = "OutOfMemoryError"


def _ctx(tmp_path):
    return SimpleNamespace(paths=lambda rel: tmp_path / rel)


def test_retries_with_shorter_sequences(tmp_path):
    (tmp_path / "artifacts/checkpoints/x").mkdir(parents=True)
    seen = []

    def run(ml):
        seen.append(ml)
        if ml > 2048:
            raise FakeOOM("CUDA out of memory")
        return {"ok": ml}

    assert _oom_retry(_ctx(tmp_path), "x", run, 4096) == {"ok": 2048}
    assert seen == [4096, 3072, 2048]
    assert not (tmp_path / "artifacts/checkpoints/x").exists()   # half-finished checkpoint removed


def test_other_errors_are_not_swallowed(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        _oom_retry(_ctx(tmp_path), "x", lambda ml: (_ for _ in ()).throw(ValueError("bad")), 4096)
