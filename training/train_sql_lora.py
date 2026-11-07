"""Single-GPU QLoRA experiment. Requires prepared train/validation JSONL."""
import argparse
import hashlib
import json
import re
import importlib.metadata
import platform
import resource
import signal
import sys
import time
from pathlib import Path


def validate_revision(revision):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be an immutable 40-character model commit SHA")
    return revision.lower()


def runtime_metadata(torch):
    packages = {}
    for name in ['torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'datasets']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'packages': packages, 'python': platform.python_version(),
            'cuda': torch.version.cuda,
            'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in Path(__file__).parent.glob('*.py')}}


def validate_resume(output, checkpoint, expected):
    if checkpoint is None:
        if output.exists():
            raise ValueError('Use a new output directory or explicit --resume-from-checkpoint')
        return
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.parent != output.resolve() or not (checkpoint/'trainer_state.json').is_file():
        raise ValueError('Resume checkpoint must be a Trainer checkpoint inside the output directory')
    previous = json.loads((output/'experiment.json').read_text())
    for key, value in expected.items():
        if previous.get(key) != value:
            raise ValueError(f'Resume experiment mismatch: {key}')


def validate_adapter(adapter, model, revision):
    experiment = json.loads((adapter.parent/'experiment.json').read_text())
    if experiment.get('model') != model or experiment.get('revision') != revision:
        raise ValueError('Adapter base model/revision does not match evaluation model')
    config = json.loads((adapter/'adapter_config.json').read_text())
    if config.get('base_model_name_or_path') != model:
        raise ValueError('Adapter configuration base model mismatch')
    return experiment


def encode_example(tokenizer, row, max_length):
    messages = [{'role': 'user', 'content': row['prompt']}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': row['completion']}], tokenize=True)
    if full[:len(prefix)] != prefix:
        raise ValueError('Chat template does not preserve the assistant-prefix boundary')
    if len(full) > max_length:
        return None  # Never truncate SQL targets or their schema context.
    if len(full) <= len(prefix):
        raise ValueError('Example has no answer tokens')
    return {'input_ids': full, 'attention_mask': [1]*len(full),
            'labels': [-100]*len(prefix) + full[len(prefix):]}


def select_training_examples(tokenizer, rows, max_length, smoke=False):
    encoded = [encode_example(tokenizer, row, max_length) for row in rows]
    fitting = [item for item in encoded if item is not None]
    # Scan the complete split: the first fitting rows need not stress peak memory.
    selected = sorted(fitting, key=lambda item: len(item['input_ids']), reverse=True)[:2] if smoke else fitting
    return selected, {'source': len(rows), 'examined': len(encoded), 'retained': len(selected),
                      'eligible': len(fitting), 'overlength': len(encoded)-len(fitting),
                      'selected_token_lengths': [len(item['input_ids']) for item in selected] if smoke else None}


class RuntimeBudget:
    """Request a full Trainer checkpoint only at an optimizer-step boundary."""
    def __init__(self, max_runtime_seconds=None, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.max_runtime_seconds = max_runtime_seconds
        self.signal_number = None
        self.stop_reason = None

    def request_stop(self, signum, frame=None):
        self.signal_number = signum

    def on_step_end(self, args, state, control, **kwargs):
        reason = ('signal_' + str(self.signal_number) if self.signal_number is not None else
                  'runtime_budget' if self.max_runtime_seconds is not None and
                  self.clock()-self.started >= self.max_runtime_seconds else None)
        if reason and state.global_step < state.max_steps:
            self.stop_reason = reason
            control.should_save = True
            control.should_training_stop = True
        return control


def memory_metrics(torch):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
            'host_peak_rss_bytes': int(rss if sys.platform == 'darwin' else rss*1024)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-Coder-7B-Instruct')
    p.add_argument('--revision', required=True, help='Pinned Hugging Face model commit')
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--resume-from-checkpoint', type=Path)
    p.add_argument('--save-steps', type=int, default=100, help='Optimizer steps between resumable checkpoints')
    p.add_argument('--max-runtime-seconds', type=float, help='Stop and checkpoint at a step boundary; reserve time for checkpoint I/O before Slurm expiry')
    p.add_argument('--smoke', action='store_true', help='Two optimizer steps on the two longest fitting examples from each split')
    p.add_argument('--max-steps', type=int, default=-1, help='Use 2 for a GPU smoke test, a new output directory for the full run')
    a = p.parse_args()
    a.revision = validate_revision(a.revision)
    if a.save_steps < 1 or (a.max_runtime_seconds is not None and a.max_runtime_seconds <= 0):
        p.error('Save steps and runtime budget must be positive')
    runtime_budget = RuntimeBudget(a.max_runtime_seconds)
    if a.smoke:
        a.max_steps = 2
    if not a.resume_from_checkpoint and a.output.exists():
        p.error('Use a new output directory or --resume-from-checkpoint')
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise RuntimeError('This recipe requires an allocated CUDA GPU')
    set_seed(6130)
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    splits, counts, sources = {}, {}, {}
    for split in ['train', 'validation']:
        path = a.data/f'{split}.jsonl'
        sources[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        retained, counts[split] = select_training_examples(tokenizer, rows, a.max_length, a.smoke)
        if not retained:
            raise ValueError(f'No {split} examples fit the sequence budget')
        splits[split] = Dataset.from_list(retained)
    resume_identity = {'model': a.model, 'revision': a.revision, 'source_sha256': sources,
                       'max_length': a.max_length, 'smoke': a.smoke, 'max_steps': a.max_steps, 'epochs': a.epochs}
    if a.smoke:
        resume_identity['smoke_selection'] = 'longest_fitting'
    validate_resume(a.output, a.resume_from_checkpoint, resume_identity)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    torch.cuda.reset_peak_memory_stats(0)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision,
        quantization_config=quant, torch_dtype=dtype, device_map={'': 0}, attn_implementation='sdpa', low_cpu_mem_usage=True)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                           target_modules='all-linear', task_type='CAUSAL_LM', bias='none'))
    model.config.use_cache = False
    args = TrainingArguments(output_dir=str(a.output), num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=1 if a.smoke else 16,
        learning_rate=1e-4, warmup_ratio=.03, lr_scheduler_type='cosine', gradient_checkpointing=True,
        bf16=bf16, fp16=not bf16, logging_steps=10, save_steps=a.save_steps, save_total_limit=2,
        eval_strategy='no', label_names=['labels'], report_to='none', seed=6130, data_seed=6130)
    a.output.mkdir(parents=True, exist_ok=a.resume_from_checkpoint is not None)
    class BudgetCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            return runtime_budget.on_step_end(args, state, control, **kwargs)

    trainer = Trainer(model=model, args=args, train_dataset=splits['train'], eval_dataset=splits['validation'],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[BudgetCallback()])
    manifest = {**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                'smoke_selection': 'longest_fitting' if a.smoke else None,
                'runtime':runtime_metadata(torch), 'source_sha256':sources, 'counts':counts, 'gpu':torch.cuda.get_device_name(0),
                'trainable_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad),
                'notice':'Training loss is not SQL accuracy. Compare base and adapter on identical held-out questions.'}
    manifest_path = a.output/('resume_experiment.json' if a.resume_from_checkpoint else 'experiment.json')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    previous_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        previous_handlers[sig] = signal.signal(sig, runtime_budget.request_stop)
    status_path = a.output/'run_status.json'
    try:
        result = trainer.train(resume_from_checkpoint=str(a.resume_from_checkpoint) if a.resume_from_checkpoint else None)
        if runtime_budget.stop_reason:
            checkpoint = a.output/f'checkpoint-{trainer.state.global_step}'
            required = ['trainer_state.json', 'optimizer.pt', 'scheduler.pt']
            if not all((checkpoint/name).is_file() for name in required):
                raise RuntimeError(f'Graceful stop did not save a complete checkpoint: {checkpoint}')
            trainer.save_metrics('partial_train', result.metrics)
            status = {'status': 'paused', 'reason': runtime_budget.stop_reason,
                      'checkpoint': str(checkpoint), 'global_step': trainer.state.global_step,
                      'planned_max_steps': trainer.state.max_steps}
        else:
            trainer.save_model(str(a.output/'adapter'))
            tokenizer.save_pretrained(a.output/'adapter')
            trainer.save_metrics('train', result.metrics)
            trainer.save_metrics('validation', trainer.evaluate())
            trainer.save_state()
            status = {'status': 'completed', 'global_step': trainer.state.global_step,
                      'planned_max_steps': trainer.state.max_steps}
        status_path.write_text(json.dumps(status, indent=2))
        print(json.dumps(status), flush=True)
    except BaseException as exc:
        status_path.write_text(json.dumps({'status': 'failed', 'error_type': type(exc).__name__,
                                          'error': str(exc)}, indent=2))
        raise
    finally:
        (a.output/'memory_metrics.json').write_text(json.dumps(memory_metrics(torch), indent=2))
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)



if __name__ == '__main__':
    main()
