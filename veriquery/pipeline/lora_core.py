"""Shared LoRA training machinery (plain torch + peft, no HF Trainer, so library version changes cannot break it).

Features: LoRA / rsLoRA / DoRA, optional 4-bit (QLoRA), completion-only loss, gradient accumulation, cosine schedule,
checkpoint every N steps with automatic resume, SIGTERM-safe (a pre-empted Vast instance saves before exiting),
live progress file + tqdm bar, validation loss.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tqdm import tqdm

IGNORE = -100


# ---------------------------------------------------------------- model / tokenizer
def pick_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(name: str, load_4bit: bool = False, grad_ckpt: bool = True):
    import torch
    from transformers import AutoModelForCausalLM
    kw = {"trust_remote_code": True}
    device = pick_device()
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": 0}
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kw)
    except TypeError:                                  # newest transformers renamed it to dtype
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw)
    if load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)
    else:
        model.to(device)
        if grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def attach_lora(model, method: str = "lora", r: int = 32, alpha: int = 64, dropout: float = 0.05,
                target_modules="all-linear", init_from: str | None = None):
    from peft import LoraConfig, PeftModel, get_peft_model
    if init_from:
        return PeftModel.from_pretrained(model, init_from, is_trainable=True)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules,
                     use_rslora=(method == "rslora"), use_dora=(method == "dora"), task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def trainable_summary(model) -> str:
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return f"trainable {tr/1e6:.1f}M / {tot/1e6:.0f}M params ({100*tr/max(tot,1):.2f}%)"


# ---------------------------------------------------------------- encoding
def chat_prompt_ids(tok, prompt: str) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def encode_pair(tok, prompt: str, completion: str, max_len: int):
    """(input_ids, labels) with loss only on the completion (+EOS); None if longer than max_len."""
    p = chat_prompt_ids(tok, prompt)
    c = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p) + len(c) > max_len:
        return None
    return p + c, [IGNORE] * len(p) + c


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int):
    import torch
    n = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, :len(a)] = torch.tensor(a)
        lab[i, :len(b)] = torch.tensor(b)
        att[i, :len(a)] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def sequence_logps(model, batch, device) -> "torch.Tensor":
    """Sum of token log-probs over the labelled (completion) positions, one value per sequence."""
    import torch
    ids, lab, att = (batch[k].to(device) for k in ("input_ids", "labels", "attention_mask"))
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    tgt = lab[:, 1:]
    mask = tgt != IGNORE
    # Only the completion positions need a softmax over the 150k-word vocabulary; doing it for the whole
    # prompt as well costs several GB on a 32 GB card. Same numbers, far less memory.
    sel = logits[mask]
    lp = torch.log_softmax(sel.float(), dim=-1).gather(-1, tgt[mask].unsqueeze(-1)).squeeze(-1)
    out = torch.zeros(ids.size(0), device=lp.device, dtype=lp.dtype)
    return out.index_add(0, mask.nonzero()[:, 0], lp)


def sft_loss(model, batch, device):
    """Mean token cross-entropy over completion tokens."""
    import torch
    import torch.nn.functional as F
    ids, lab, att = (batch[k].to(device) for k in ("input_ids", "labels", "attention_mask"))
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    tgt = lab[:, 1:]
    mask = tgt != IGNORE
    loss = F.cross_entropy(logits[mask].float(), tgt[mask])   # mean over completion tokens; no full-vocab float copy of the prompt
    return loss, {"loss": float(loss.detach())}


# ---------------------------------------------------------------- the loop
@dataclass
class LoopConfig:
    name: str
    out_dir: Path                 # final adapter
    ckpt_dir: Path                # rolling checkpoints
    state_file: Path              # live progress json
    log_file: Path | None = None  # jsonl metrics
    epochs: float = 2.0
    lr: float = 2e-4
    micro_bs: int = 4
    accum: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    save_every: int = 100
    eval_every: int = 100
    max_steps: int | None = None
    seed: int = 13
    keep: int = 2
    clip: float = 1.0


class Loop:
    def __init__(self, model, items: list, collate_fn: Callable, loss_fn: Callable, cfg: LoopConfig,
                 eval_items: list | None = None, meta: dict | None = None):
        self.model, self.items, self.collate_fn, self.loss_fn, self.cfg = model, items, collate_fn, loss_fn, cfg
        self.eval_items = eval_items or []
        self.meta = meta or {}
        self.device = pick_device() if not hasattr(model, "hf_device_map") else "cuda"
        self.stop = False
        per_step = cfg.micro_bs * cfg.accum
        self.steps_per_epoch = max(1, len(items) // per_step)
        self.total = cfg.max_steps or max(1, int(round(cfg.epochs * self.steps_per_epoch)))

    # --- checkpoints
    def _ckpts(self) -> list[tuple[int, Path]]:
        out = []
        for p in self.cfg.ckpt_dir.glob("checkpoint-*"):
            if (p / "COMPLETE").exists():
                out.append((int(p.name.split("-")[1]), p))
        return sorted(out)

    def save_ckpt(self, step: int, opt, sched) -> None:
        import torch
        d = self.cfg.ckpt_dir / f"checkpoint-{step}"
        tmp = self.cfg.ckpt_dir / f".tmp-{step}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        self.model.save_pretrained(tmp)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, tmp / "train_state.pt")
        (tmp / "COMPLETE").write_text("ok")
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)
        for s, p in self._ckpts()[:-self.cfg.keep]:
            shutil.rmtree(p, ignore_errors=True)

    def load_ckpt(self, opt, sched) -> int:
        import torch
        cks = self._ckpts()
        if not cks:
            return 0
        step, d = cks[-1]
        from peft import set_peft_model_state_dict
        f = d / "adapter_model.safetensors"
        if f.exists():
            from safetensors.torch import load_file
            sd = load_file(str(f))
        else:
            sd = torch.load(d / "adapter_model.bin", map_location="cpu")
        set_peft_model_state_dict(self.model, sd)
        st = torch.load(d / "train_state.pt", map_location="cpu")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        print(f"[{self.cfg.name}] resumed from {d.name}")
        return step

    # --- progress
    def _progress(self, step: int, loss: float, t0: float, extra: dict) -> None:
        el = time.time() - t0
        done = step - self._start
        eta = el / max(done, 1) * (self.total - step) if done else None
        st = {"name": self.cfg.name, "step": step, "total": self.total, "loss": loss, "elapsed_s": el,
              "eta_s": eta, "updated": time.time(), **extra}
        self.cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.state_file.write_text(json.dumps(st))
        if self.cfg.log_file:
            self.cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cfg.log_file.open("a") as f:
                f.write(json.dumps(st) + "\n")

    def evaluate(self, n: int = 64) -> float | None:
        import torch
        if not self.eval_items:
            return None
        self.model.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, min(len(self.eval_items), n), self.cfg.micro_bs):
                b = self.collate_fn(self.eval_items[i:i + self.cfg.micro_bs])
                loss, _ = self.loss_fn(self.model, b, self.device)
                losses.append(float(loss))
        self.model.train()
        return sum(losses) / len(losses) if losses else None

    def run(self) -> dict:
        import torch
        cfg = self.cfg
        cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
        warm = max(1, int(cfg.warmup_ratio * self.total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else
            0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, self.total - warm))))
        start = self.load_ckpt(opt, sched)
        self._start = start
        use_amp = self.device == "cuda" and torch.cuda.is_bf16_supported()

        def on_term(*_):
            self.stop = True
        old = signal.signal(signal.SIGTERM, on_term)
        self.model.train()
        t0 = time.time()
        n = len(self.items)
        per_step = cfg.micro_bs * cfg.accum
        bar = tqdm(total=self.total, initial=start, desc=cfg.name, unit="step", file=sys.stdout)
        last_loss, ev = float("nan"), None
        step = start
        try:
            while step < self.total:
                epoch, within = divmod(step, self.steps_per_epoch)
                perm = random.Random(cfg.seed + epoch).sample(range(n), n)
                idx = perm[within * per_step:(within + 1) * per_step]
                tot = 0.0
                for a in range(cfg.accum):
                    chunk = idx[a * cfg.micro_bs:(a + 1) * cfg.micro_bs]
                    if not chunk:
                        continue
                    batch = self.collate_fn([self.items[i] for i in chunk])
                    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cpu", enabled=False)
                    with ctx:
                        loss, _m = self.loss_fn(self.model, batch, self.device)
                    (loss / cfg.accum).backward()
                    tot += float(loss.detach()) / cfg.accum
                gn = float(torch.nn.utils.clip_grad_norm_(params, cfg.clip))
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                last_loss = tot
                if cfg.eval_every and step % cfg.eval_every == 0:
                    ev = self.evaluate()
                extra = {"lr": sched.get_last_lr()[0], "grad_norm": gn, "epoch": step / self.steps_per_epoch,
                         "val_loss": ev}
                self._progress(step, last_loss, t0, extra)
                bar.update(1)
                bar.set_postfix(loss=f"{last_loss:.4f}", val=(f"{ev:.4f}" if ev else "-"))
                if step % cfg.save_every == 0 or self.stop:
                    self.save_ckpt(step, opt, sched)
                if self.stop:
                    print(f"[{cfg.name}] SIGTERM: checkpoint saved at step {step}, exiting (re-run to resume)")
                    sys.exit(143)
        finally:
            bar.close()
            signal.signal(signal.SIGTERM, old)
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(cfg.out_dir)
        meta = {**self.meta, "steps": step, "final_loss": last_loss, "val_loss": ev,
                "train_seconds": time.time() - t0, "n_examples": n, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        (cfg.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str))
        return meta
