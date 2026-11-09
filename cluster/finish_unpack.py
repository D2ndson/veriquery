"""Finish unpacking BIRD train's nested train_databases.zip after an interrupted extraction (disk quota).

The pipeline treats a split as unpacked once one database exists, so a partial unpack must be completed before it
runs. Files already extracted at full size are skipped; the inner zip is deleted afterwards to free 8.6 GB.
"""
import zipfile
from pathlib import Path

d = Path.home() / "vq-combined/data/bird/train/train"
inner = d / "train_databases.zip"
if inner.exists():
    with zipfile.ZipFile(inner) as z:
        n = 0
        for m in z.infolist():
            if m.filename.startswith("__MACOSX") or "/._" in m.filename or m.is_dir():
                continue
            out = d / m.filename
            if out.is_file() and out.stat().st_size == m.file_size:
                continue
            z.extract(m, d)
            n += 1
    (d / ".train_databases.unpacked").write_text("ok")
    inner.unlink()
    print(f"[finish_unpack] extracted {n} remaining files; removed {inner.name}")
dbs = list((d / "train_databases").glob("*/*.sqlite"))
print(f"[finish_unpack] {len(dbs)} train databases present")
