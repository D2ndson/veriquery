"""SFT of the SQL writer (LoRA / rsLoRA / DoRA / QLoRA), completion-only loss, resumable.

    python -m veriquery.pipeline.train_lora --name writer-sft --train artifacts/sft/sft_train.jsonl

Re-running resumes from the last checkpoint in artifacts/checkpoints/<name>/; once artifacts/adapters/<name>/ exists the
runner skips the stage (use --force to retrain).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from veriquery.config import load_models_config, path
from veriquery.pipeline import lora_core as lc


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def train_sft(name: str, train_file: str | Path, val_file: str | Path | None = None, base_model: str | None = None,
              method: str = "lora", r: int = 32, alpha: int = 64, dropout: float = 0.05, epochs: float = 2.0,
              lr: float = 2e-4, micro_bs: int = 4, accum: int = 4, max_len: int = 4096, load_4bit: bool = False,
              save_every: int = 100, eval_every: int = 100, max_steps: int | None = None, tokenizer=None,
              model=None, seed: int = 13, limit: int | None = None) -> dict:
    tc = load_models_config().get("training", {})
    base_model = base_model or tc.get("base_model", "Qwen/Qwen2.5-Coder-7B-Instruct")
    tok = tokenizer or lc.load_tokenizer(base_model)
    rows = read_jsonl(Path(train_file))[: limit or None]
    items, dropped = [], 0
    for r_ in tqdm(rows, desc="tokenise", unit="ex"):
        e = lc.encode_pair(tok, r_["prompt"], r_["completion"], max_len)
        if e is None:
            dropped += 1
        else:
            items.append(e)
    ev_items = []
    if val_file and Path(val_file).is_file():
        for r_ in read_jsonl(Path(val_file)):
            e = lc.encode_pair(tok, r_["prompt"], r_["completion"], max_len)
            if e:
                ev_items.append(e)
    print(f"[{name}] {len(items)} train examples ({dropped} dropped for length > {max_len}), {len(ev_items)} val")
    if not items:
        raise RuntimeError("No training examples after tokenisation")
    model = model or lc.load_base(base_model, load_4bit)
    model = lc.attach_lora(model, method, r, alpha, dropout)
    print(f"[{name}] {method} r={r} alpha={alpha}: {lc.trainable_summary(model)}")
    cfg = lc.LoopConfig(
        name=name, out_dir=path(f"artifacts/adapters/{name}"), ckpt_dir=path(f"artifacts/checkpoints/{name}"),
        state_file=path(f"artifacts/state/train_{name}.json"), log_file=path(f"logs/train_{name}.jsonl"),
        epochs=epochs, lr=lr, micro_bs=micro_bs, accum=accum, save_every=save_every, eval_every=eval_every,
        max_steps=max_steps, seed=seed)
    meta = {"kind": "sft", "base_model": base_model, "method": method, "r": r, "alpha": alpha, "lr": lr,
            "epochs": epochs, "micro_bs": micro_bs, "accum": accum, "max_len": max_len, "load_4bit": load_4bit,
            "dropped_for_length": dropped, "train_file": str(train_file)}
    pad = tok.pad_token_id
    loop = lc.Loop(model, items, lambda b: lc.collate(b, pad), lc.sft_loss, cfg, ev_items, meta)
    out = loop.run()
    print(f"[{name}] done: final loss {out['final_loss']:.4f}, adapter at {cfg.out_dir}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="writer-sft")
    ap.add_argument("--train", default="artifacts/sft/sft_train.jsonl")
    ap.add_argument("--val", default="artifacts/sft/sft_val.jsonl")
    ap.add_argument("--base-model")
    ap.add_argument("--method", default="lora", choices=["lora", "rslora", "dora"])
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--4bit", dest="load_4bit", action="store_true")
    ap.add_argument("--max-steps", type=int)
    a = ap.parse_args()
    train_sft(a.name, path(a.train), path(a.val), a.base_model, a.method, a.r, a.alpha, epochs=a.epochs, lr=a.lr,
              micro_bs=a.micro_bs, accum=a.accum, max_len=a.max_len, load_4bit=a.load_4bit, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
