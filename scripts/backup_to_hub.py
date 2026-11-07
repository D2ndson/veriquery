#!/usr/bin/env python
"""Back up everything you cannot cheaply recreate: LoRA adapters, results, calibrator, golden runs, mined data.

    python scripts/backup_to_hub.py --repo your-username/veriquery-artifacts   # private HF model repo (token needs WRITE scope)
    python scripts/backup_to_hub.py --local-only                               # just make artifacts/backup/veriquery_backup.tgz

Restore on a fresh instance:   python scripts/backup_to_hub.py --restore --repo your-username/veriquery-artifacts
Vast instances are deleted when you destroy them: do this BEFORE you destroy the instance (or download the .tgz).
"""
import argparse
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from veriquery.config import hf_token, path  # noqa: E402

KEEP = ["artifacts/adapters", "artifacts/calibrator.json", "artifacts/fewshot", "artifacts/semantic_layer",
        "artifacts/sft/build_stats.json", "artifacts/sft/pref_stats.json", "artifacts/trajectories", "artifacts/state",
        "results", "data/golden", "logs"]


def make_tgz() -> Path:
    out = path("artifacts/backup/veriquery_backup.tgz")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as t:
        for rel in KEEP:
            p = path(rel)
            if p.exists():
                t.add(p, arcname=rel)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--restore", action="store_true")
    a = ap.parse_args()
    try:
        from veriquery.config import load_env
        load_env()
    except Exception:
        pass
    a.repo = a.repo or os.environ.get("HF_BACKUP_REPO")
    if a.local_only or not a.repo:
        make_tgz()
        if not a.repo:
            print("(set HF_BACKUP_REPO in .env to also push to the Hugging Face Hub)")
        return
    from huggingface_hub import HfApi, snapshot_download
    tok = hf_token()
    if a.restore:
        snapshot_download(a.repo, local_dir=str(path(".")), token=tok, repo_type="model")
        print("restored into", path("."))
        return
    api = HfApi(token=tok)
    api.create_repo(a.repo, private=True, exist_ok=True)
    for rel in KEEP:
        p = path(rel)
        if p.is_dir():
            api.upload_folder(folder_path=str(p), path_in_repo=rel, repo_id=a.repo, ignore_patterns=["*.tmp", ".tmp-*", "checkpoint-*"])
        elif p.is_file():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=rel, repo_id=a.repo)
        print("uploaded", rel)
    make_tgz()
    print(f"backed up to https://huggingface.co/{a.repo} (private)")


if __name__ == "__main__":
    main()
