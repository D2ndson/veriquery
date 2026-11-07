"""Preference tuning (DPO) of the SQL writer, starting from the SFT adapter.

The reference policy is the SFT model itself, so we never keep two models in memory: reference log-probs of every
(prompt, chosen, rejected) triple are computed once before training and cached in artifacts/state/dpo_ref_<name>.json.
Loss = DPO sigmoid loss + `sft_alpha` * length-normalised NLL on the chosen answer (keeps the format and stops the
chosen likelihood from collapsing). Resumable like SFT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from veriquery.config import load_models_config, path
from veriquery.pipeline import lora_core as lc
from veriquery.pipeline.train_lora import read_jsonl


def _collate(batch, pad_id):
    import torch
    seqs = [(b[0], b[1]) for b in batch] + [(b[2], b[3]) for b in batch]
    out = lc.collate(seqs, pad_id)
    out["ref_c"] = torch.tensor([b[4] for b in batch])
    out["ref_r"] = torch.tensor([b[5] for b in batch])
    out["n_c"] = torch.tensor([sum(1 for x in b[1] if x != lc.IGNORE) for b in batch], dtype=torch.float)
    return out


def make_dpo_loss(beta: float, sft_alpha: float):
    def loss_fn(model, batch, device):
        import torch
        import torch.nn.functional as F
        lp = lc.sequence_logps(model, batch, device)
        B = batch["ref_c"].shape[0]
        pc, pr = lp[:B], lp[B:]
        rc, rr = batch["ref_c"].to(device), batch["ref_r"].to(device)
        margin = beta * ((pc - rc) - (pr - rr))
        dpo = -F.logsigmoid(margin).mean()
        nll = -(pc / batch["n_c"].to(device)).mean()
        loss = dpo + sft_alpha * nll
        return loss, {"dpo": float(dpo.detach()), "acc": float((margin > 0).float().mean())}
    return loss_fn


def train_dpo(name: str, pairs_file: str | Path, init_adapter: str | Path, base_model: str | None = None,
              beta: float = 0.1, sft_alpha: float = 0.2, epochs: float = 1.0, lr: float = 5e-6, micro_bs: int = 2,
              accum: int = 8, max_len: int = 4096, load_4bit: bool = False, save_every: int = 50,
              max_steps: int | None = None, tokenizer=None, model=None, seed: int = 13) -> dict:
    tc = load_models_config().get("training", {})
    base_model = base_model or tc.get("base_model", "Qwen/Qwen2.5-Coder-7B-Instruct")
    tok = tokenizer or lc.load_tokenizer(base_model)
    items, dropped = [], 0
    for r in tqdm(read_jsonl(Path(pairs_file)), desc="tokenise pairs", unit="pair"):
        c = lc.encode_pair(tok, r["prompt"], r["chosen"], max_len)
        j = lc.encode_pair(tok, r["prompt"], r["rejected"], max_len)
        if c is None or j is None:
            dropped += 1
        else:
            items.append([c[0], c[1], j[0], j[1], 0.0, 0.0])
    if not items:
        raise RuntimeError("No preference pairs to train on")
    print(f"[{name}] {len(items)} pairs ({dropped} dropped for length)")
    model = model or lc.load_base(base_model, load_4bit)
    model = lc.attach_lora(model, init_from=str(init_adapter))
    print(f"[{name}] starting from SFT adapter {init_adapter}: {lc.trainable_summary(model)}")
    pad = tok.pad_token_id
    device = lc.pick_device() if not hasattr(model, "hf_device_map") else "cuda"

    ref_file = path(f"artifacts/state/dpo_ref_{name}.json")
    cached = json.loads(ref_file.read_text()) if ref_file.is_file() else None
    if cached and len(cached) == len(items):
        for it, (a, b) in zip(items, cached):
            it[4], it[5] = a, b
        print(f"[{name}] reference log-probs loaded from cache")
    else:
        import torch
        model.eval()
        refs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(items), micro_bs), desc="reference log-probs", unit="batch"):
                chunk = items[i:i + micro_bs]
                b = lc.collate([(x[0], x[1]) for x in chunk] + [(x[2], x[3]) for x in chunk], pad)
                lp = lc.sequence_logps(model, b, device).tolist()
                n = len(chunk)
                refs += [[lp[k], lp[n + k]] for k in range(n)]
        for it, (a, b) in zip(items, refs):
            it[4], it[5] = a, b
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        ref_file.write_text(json.dumps(refs))
        model.train()

    cfg = lc.LoopConfig(name=name, out_dir=path(f"artifacts/adapters/{name}"), ckpt_dir=path(f"artifacts/checkpoints/{name}"),
                        state_file=path(f"artifacts/state/train_{name}.json"), log_file=path(f"logs/train_{name}.jsonl"),
                        epochs=epochs, lr=lr, micro_bs=micro_bs, accum=accum, save_every=save_every, eval_every=0,
                        max_steps=max_steps, seed=seed)
    meta = {"kind": "dpo", "base_model": base_model, "init_adapter": str(init_adapter), "beta": beta,
            "sft_alpha": sft_alpha, "lr": lr, "epochs": epochs, "pairs": len(items), "pairs_file": str(pairs_file)}
    out = lc.Loop(model, items, lambda b: _collate(b, pad), make_dpo_loss(beta, sft_alpha), cfg, None, meta).run()
    print(f"[{name}] done: adapter at {cfg.out_dir}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="writer-dpo")
    ap.add_argument("--pairs", default="artifacts/sft/pairs_gold.jsonl")
    ap.add_argument("--init", default="artifacts/adapters/writer-sft")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--4bit", dest="load_4bit", action="store_true")
    a = ap.parse_args()
    train_dpo(a.name, path(a.pairs), path(a.init), beta=a.beta, lr=a.lr, epochs=a.epochs, load_4bit=a.load_4bit)


if __name__ == "__main__":
    main()
