"""Download, unpack and locate the BIRD benchmark (Mini-Dev, Dev, Train).

BIRD is free but too large to ship in a zip. `download_bird(["minidev", "train"])` fetches the official archives
with resume and a progress bar, unpacks them (including the nested *_databases.zip files), and writes
data/bird/manifest.json that every later stage reads.

If the download URLs ever change (or the Vast box is slow to reach them), put the zip files into data/bird/_downloads/
yourself (names: minidev.zip, dev.zip, train.zip) and re-run: files that already exist are never downloaded again.
"""
from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from veriquery.config import path

URLS = {
    "minidev": ["https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"],
    "dev": ["https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"],
    "train": ["https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"],
}
JSON_NAMES = {"minidev": "mini_dev_sqlite.json", "dev": "dev.json", "train": "train.json"}
DB_DIRS = {"minidev": "dev_databases", "dev": "dev_databases", "train": "train_databases"}


@dataclass
class BirdSplit:
    name: str
    json: Path
    db_root: Path


def bird_dir() -> Path:
    return path("data/bird")


def download(url: str, dest: Path, retries: int = 6, chunk: int = 1 << 20) -> Path:
    """Resumable download with a progress bar. Returns dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                status = getattr(r, "status", 200)
                total = int(r.headers.get("Content-Length", 0))
                if status == 200 and have:        # server ignored Range: start over
                    have, part_mode = 0, "wb"
                else:
                    part_mode = "ab" if have else "wb"
                with part.open(part_mode) as f, tqdm(total=total + have if total else None, initial=have, unit="B",
                                                     unit_scale=True, desc=dest.name) as bar:
                    while True:
                        buf = r.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        bar.update(len(buf))
            part.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  download attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
    raise RuntimeError("unreachable")


def _extract(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if not m.startswith("__MACOSX") and "/._" not in m]
        for m in tqdm(members, desc=f"unzip {archive.name}", unit="file"):
            z.extract(m, target)


def unzip(archive: Path, target: Path) -> None:
    """Extract `archive`, then extract any nested *databases.zip (BIRD ships databases as a zip in the zip)."""
    _extract(archive, target)
    for inner in sorted(target.rglob("*databases.zip")):
        marker = inner.parent / f".{inner.stem}.unpacked"
        if marker.exists():
            continue
        _extract(inner, inner.parent)       # databases zips do not nest further
        marker.write_text("ok")


def locate(split_dir: Path, name: str) -> BirdSplit | None:
    """Find the question JSON and the database root inside an unpacked split."""
    jsons = [p for p in split_dir.rglob(JSON_NAMES[name]) if "__MACOSX" not in str(p)]
    dbs = [p for p in split_dir.rglob(DB_DIRS[name]) if p.is_dir() and "__MACOSX" not in str(p)
           and any(p.glob("*/*.sqlite"))]
    if not jsons or not dbs:
        return None
    return BirdSplit(name, jsons[0], dbs[0])


def load_manifest() -> dict[str, BirdSplit]:
    f = bird_dir() / "manifest.json"
    if not f.is_file():
        return {}
    raw = json.loads(f.read_text())
    return {k: BirdSplit(k, Path(v["json"]), Path(v["db_root"])) for k, v in raw.items()}


def get_split(name: str) -> BirdSplit:
    split = load_manifest().get(name)
    if split is None or not split.json.is_file():
        raise FileNotFoundError(f"BIRD split {name!r} not found. Run: python scripts/download_data.py --what {name}")
    return split


def download_bird(what: list[str], urls: dict | None = None, keep_zips: bool = False) -> dict[str, BirdSplit]:
    """Fetch + unpack the requested splits; returns {name: BirdSplit}. Already-unpacked splits are skipped."""
    root = bird_dir()
    (root / "_downloads").mkdir(parents=True, exist_ok=True)
    found = load_manifest()
    for name in what:
        if name in found and found[name].json.is_file():
            print(f"[bird] {name}: already unpacked at {found[name].db_root.parent}")
            continue
        target = root / name
        split = locate(target, name) if target.exists() else None
        if split is None:
            archive = root / "_downloads" / f"{name}.zip"
            if not archive.exists():
                last = None
                for url in (urls or URLS)[name]:
                    try:
                        print(f"[bird] downloading {name} from {url}")
                        download(url, archive)
                        break
                    except Exception as e:  # try the next mirror
                        last = e
                else:
                    raise RuntimeError(
                        f"Could not download BIRD {name}: {last}\nDownload it manually from "
                        f"https://bird-bench.github.io/ and place the file at {archive}, then re-run.")
            unzip(archive, target)
            split = locate(target, name)
            if split is None:
                raise RuntimeError(f"Unpacked {archive} but could not find {JSON_NAMES[name]} and {DB_DIRS[name]}/")
            if not keep_zips:
                archive.unlink(missing_ok=True)
        found[name] = split
        print(f"[bird] {name}: json={split.json}  databases={split.db_root}")
    (root / "manifest.json").write_text(json.dumps(
        {k: {"json": str(v.json), "db_root": str(v.db_root)} for k, v in found.items()}, indent=1))
    return found


def load_questions(name: str) -> list[dict]:
    """BIRD questions with duplicate ids removed (Mini-Dev lists a couple twice)."""
    split = get_split(name)
    seen, out = set(), []
    for i, q in enumerate(json.loads(split.json.read_text(encoding="utf-8"))):
        qid = q.get("question_id", i)
        if qid in seen:
            continue
        seen.add(qid)
        out.append({**q, "question_id": qid})
    return out


def db_file(split: BirdSplit, db_id: str) -> str:
    return str(split.db_root / db_id / f"{db_id}.sqlite")


def disk_free_gb() -> float:
    p = bird_dir()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9
