"""Score an untuned model or its LoRA adapter with identical SQL prompts."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.train_sql_lora import validate_revision, validate_adapter, runtime_metadata
from benchmarks.bird import load_bird, fixed_subset, database_path
from benchmarks.run_sql import evaluate, summarize, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--db-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-Coder-7B-Instruct')
    p.add_argument('--revision', required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--size', type=int, default=100)
    p.add_argument('--mode', choices=['once', 'repair', 'react'], default='repair')
    p.add_argument('--context-tokens', type=int, default=4096)
    p.add_argument('--max-output-tokens', type=int, default=1024)
    a = p.parse_args()
    a.revision = validate_revision(a.revision)
    if a.adapter:
        validate_adapter(a.adapter, a.model, a.revision)
    if a.output.exists():
        p.error('Use a new output directory')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    examples = fixed_subset(load_bird(a.data), size=a.size, seed=6130)
    for x in examples:
        database_path(a.db_root, x.db_id)
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        device_map={'': 0}, torch_dtype=dtype, attn_implementation='sdpa')
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    class Client:
        def __init__(self):
            self.stats = {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'seconds': 0.0}
        def __call__(self, prompt):
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
            if ids.shape[1] + a.max_output_tokens > a.context_tokens:
                raise ValueError('Prompt plus output budget exceeds context; no silent truncation')
            start = time.monotonic()
            self.stats['calls'] += 1
            with torch.inference_mode():
                result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    do_sample=False, max_new_tokens=a.max_output_tokens, pad_token_id=tokenizer.eos_token_id)
            new = result[0, ids.shape[1]:]
            self.stats['prompt_tokens'] += ids.shape[1]
            self.stats['completion_tokens'] += len(new)
            self.stats['seconds'] += time.monotonic()-start
            if len(new) == a.max_output_tokens and new[-1].item() != tokenizer.eos_token_id:
                raise ValueError('Output token budget exhausted')
            return tokenizer.decode(new, skip_special_tokens=True)

    a.output.mkdir(parents=True, exist_ok=False)
    manifest = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}
    manifest.update(runtime=runtime_metadata(torch), dataset_sha256=digest(a.data),
        selected_ids=[{'db_id':x.db_id, 'question_id':x.question_id} for x in examples],
        database_sha256={x.db_id:digest(database_path(a.db_root,x.db_id)) for x in examples},
        adapter_sha256={p.name:digest(p) for p in a.adapter.glob('*') if p.is_file()} if a.adapter else None)
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    with (a.output/f'{a.mode}.jsonl').open('w') as out:
        def save(record):
            out.write(json.dumps(record, default=str)+'\n');out.flush()
        records = evaluate(examples, a.db_root, Client(), mode=a.mode, on_record=save)
    (a.output/'summary.json').write_text(json.dumps(summarize(records), indent=2))


if __name__ == '__main__':
    main()
