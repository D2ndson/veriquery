"""Single-GPU QLoRA experiment. Requires prepared train/validation JSONL."""
import argparse
import hashlib
import json
import re
import importlib.metadata
import platform
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-Coder-7B-Instruct')
    p.add_argument('--revision', required=True, help='Pinned Hugging Face model commit')
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--resume-from-checkpoint', type=Path)
    p.add_argument('--smoke', action='store_true', help='Two optimizer steps, at most two train and two validation examples')
    p.add_argument('--max-steps', type=int, default=-1, help='Use 2 for a GPU smoke test, a new output directory for the full run')
    a = p.parse_args()
    a.revision = validate_revision(a.revision)
    if a.smoke:
        a.max_steps = 2
    if not a.resume_from_checkpoint and a.output.exists():
        p.error('Use a new output directory or --resume-from-checkpoint')
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForSeq2Seq, set_seed
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
        encoded = []
        retained_count = 0
        for row in rows:
            item = encode_example(tokenizer, row, a.max_length)
            encoded.append(item)
            retained_count += item is not None
            if a.smoke and retained_count == 2:
                break
        retained = [x for x in encoded if x is not None]
        if not retained:
            raise ValueError(f'No {split} examples fit the sequence budget')
        counts[split] = {'source': len(rows), 'examined': len(encoded), 'retained': len(retained),
                         'overlength': len(encoded)-len(retained)}
        splits[split] = Dataset.from_list(retained)
    resume_identity = {'model': a.model, 'revision': a.revision, 'source_sha256': sources,
                       'max_length': a.max_length, 'smoke': a.smoke, 'max_steps': a.max_steps, 'epochs': a.epochs}
    validate_resume(a.output, a.resume_from_checkpoint, resume_identity)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision,
        quantization_config=quant, torch_dtype=dtype, device_map={'': 0}, attn_implementation='sdpa')
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                           target_modules='all-linear', task_type='CAUSAL_LM', bias='none'))
    model.config.use_cache = False
    args = TrainingArguments(output_dir=str(a.output), num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=1 if a.smoke else 16,
        learning_rate=1e-4, warmup_ratio=.03, lr_scheduler_type='cosine', gradient_checkpointing=True,
        bf16=bf16, fp16=not bf16, logging_steps=10, save_steps=100, save_total_limit=2,
        eval_strategy='no', label_names=['labels'], report_to='none', seed=6130, data_seed=6130)
    a.output.mkdir(parents=True, exist_ok=a.resume_from_checkpoint is not None)
    trainer = Trainer(model=model, args=args, train_dataset=splits['train'], eval_dataset=splits['validation'],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    manifest = {**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                'runtime':runtime_metadata(torch), 'source_sha256':sources, 'counts':counts, 'gpu':torch.cuda.get_device_name(0),
                'trainable_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad),
                'notice':'Training loss is not SQL accuracy. Compare base and adapter on identical held-out questions.'}
    manifest_path = a.output/('resume_experiment.json' if a.resume_from_checkpoint else 'experiment.json')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    result = trainer.train(resume_from_checkpoint=str(a.resume_from_checkpoint) if a.resume_from_checkpoint else None)
    trainer.save_model(str(a.output/'adapter'))
    tokenizer.save_pretrained(a.output/'adapter')
    trainer.save_metrics('train', result.metrics)
    trainer.save_metrics('validation', trainer.evaluate())
    trainer.save_state()


if __name__ == '__main__':
    main()
