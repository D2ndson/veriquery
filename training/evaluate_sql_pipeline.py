"""Controlled local-model SQL pipeline ablations; gold SQL enters scoring only."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from veriquery.llm import LLM, LLMStats
from training.freeze_sql_splits import resolve_selection
from training.train_sql_lora import validate_revision, validate_adapter, runtime_metadata
from benchmarks.bird import load_bird, database_path, score_prediction
from benchmarks.run_sql import digest


class GenerationFailure(ValueError):
    def __init__(self, message, prompt_tokens=0, completion_tokens=0):
        super().__init__(message)
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens


class LocalPipelineClient(LLM):
    """LLM facade. Samplers share backend/budget, while usage follows LLM semantics."""
    def __init__(self, backend, model, context_tokens=16384, max_tokens=1024, max_calls=24):
        super().__init__(model=model, provider='openai', cache_path=None,
                         num_ctx=context_tokens, max_tokens=max_tokens, seed=6130)
        self.backend = backend
        self.budget = {'calls': 0, 'limit': max_calls}

    def reset_question_budget(self):
        self.budget['calls'] = 0

    def chat(self, messages, max_tokens=None, temperature=None, tag=None, use_cache=True, **kwargs):
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if not isinstance(max_tokens, int) or not 0 < max_tokens <= self.max_tokens:
            raise ValueError('Requested output budget exceeds frozen per-call limit')
        if self.budget['calls'] >= self.budget['limit']:
            raise ValueError('Frozen per-question model-call budget exhausted')
        self.budget['calls'] += 1
        temperature = self.temperature if temperature is None else temperature
        tag = tag or getattr(self._local, 'agent', None) or 'untagged'
        # Per-prompt RNG makes sampling independent of interrupted/prior questions.
        payload = json.dumps([self.seed, messages], sort_keys=True, ensure_ascii=False)
        seed = int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)
        started = time.monotonic(); delta = LLMStats(calls=1)
        try:
            text, delta.prompt_tokens, delta.completion_tokens = self.backend(messages, max_tokens, temperature, seed)
            if not text.strip():
                raise ValueError('Model returned no final answer text')
            return text
        except GenerationFailure as exc:
            delta.prompt_tokens, delta.completion_tokens = exc.prompt_tokens, exc.completion_tokens
            raise
        finally:
            delta.seconds = time.monotonic()-started
            self._record(tag, delta)


def hf_backend(model, tokenizer, torch, context_tokens):
    def generate(messages, max_tokens, temperature, seed):
        ids=tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_tensors='pt').to(model.device)
        if ids.shape[1]+max_tokens > context_tokens:
            raise ValueError('Prompt plus output exceeds context; no silent truncation')
        options={'do_sample': temperature > 0, 'max_new_tokens':max_tokens,
                 'pad_token_id':tokenizer.eos_token_id}
        if temperature > 0:
            options.update(temperature=temperature, top_p=1.0, top_k=0)
        devices=[model.device.index or 0] if model.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            result=model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **options)
        generated=result[0,ids.shape[1]:]
        if len(generated)==max_tokens and generated[-1].item()!=tokenizer.eos_token_id:
            raise GenerationFailure('Output token budget exhausted; truncated answer rejected', ids.shape[1], len(generated))
        return tokenizer.decode(generated,skip_special_tokens=True),ids.shape[1],len(generated)
    return generate


def summarize(records):
    scorable=[r for r in records if r['score']['correct'] is not None]
    high=[r for r in scorable if r.get('report_confidence')=='high']
    return {'attempted':len(records),'scorable':len(scorable),
            'correct':sum(r['score']['correct'] for r in scorable),
            'accuracy':sum(r['score']['correct'] for r in scorable)/len(scorable) if scorable else None,
            'high_confidence_answers':len(high),
            'high_confidence_accuracy':sum(r['score']['correct'] for r in high)/len(high) if high else None,
            'high_confidence_coverage':len(high)/len(scorable) if scorable else None,
            'confidently_wrong_rate':sum(not r['score']['correct'] for r in high)/len(scorable) if scorable else None,
            'model_calls':sum(r.get('usage',{}).get('total',{}).get('calls',0) for r in records),
            'seconds':sum(r.get('seconds',0) for r in records),
            'diagnostic_sql_calls':sum(d.get('sql_calls',0) for r in records for d in r.get('diagnostics',[])),
            'diagnostic_schema_reads':sum(d.get('schema_reads',0) for r in records for d in r.get('diagnostics',[])),
            'diagnostic_seconds':sum(d.get('elapsed_seconds',0) for r in records for d in r.get('diagnostics',[])),
            'confidence_notice':'Existing pipeline labels/scores are heuristic, not calibrated probabilities.'}


def load_progress(output, identity, resume=False):
    """Resume only identical manifests and a valid sequential question prefix."""
    output=Path(output)
    if not output.exists():
        if resume:
            raise ValueError('Resume output does not exist')
        output.mkdir(parents=True)
        (output/'manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
        return []
    if not resume:
        raise ValueError('Output exists; use --resume explicitly')
    if json.loads((output/'manifest.json').read_text()) != identity:
        raise ValueError('Resume manifest identity mismatch')
    path=output/'traces.jsonl'
    if not path.exists():
        return []
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raw=raw[:raw.rfind(b'\n')+1]  # Only discard an interrupted final write.
        path.write_bytes(raw)
    records=[json.loads(line) for line in raw.splitlines()]
    ids=[{'db_id':r['db_id'],'question_id':r['question_id']} for r in records]
    if ids != identity['selected_ids'][:len(ids)] or len(ids)>len(identity['selected_ids']):
        raise ValueError('Resume records are not the selected question prefix')
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','db-root','selection','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--partition',choices=['historical','development','calibration','final'],required=True)
    p.add_argument('--model',default='Qwen/Qwen2.5-Coder-32B-Instruct')
    p.add_argument('--revision',required=True)
    p.add_argument('--adapter',type=Path)
    p.add_argument('--mode',choices=['generic','structure','evidence'],default='generic')
    p.add_argument('--context-tokens',type=int,default=16384)
    p.add_argument('--max-output-tokens',type=int,default=1024)
    p.add_argument('--max-calls',type=int,default=24)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-runtime-seconds',type=float,default=0)
    a=p.parse_args();a.revision=validate_revision(a.revision)
    if a.max_calls < 1 or a.max_output_tokens < 1 or a.context_tokens <= a.max_output_tokens:
        p.error('Invalid inference budgets')
    if a.adapter:
        validate_adapter(a.adapter,a.model,a.revision)
    from veriquery.agents.orchestrator import Config,VeriQuery
    from veriquery.sql_execution import SQLiteExecutor
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('An allocated CUDA GPU is required')
    set_seed(6130)
    examples=resolve_selection(load_bird(a.data),json.loads(a.selection.read_text()),a.partition,digest(a.data))
    databases={e.db_id:database_path(a.db_root,e.db_id) for e in examples}
    config=Config(use_clarifier=False,use_verifier=True,second_route='sql',n_candidates=3,
                  max_sql_repairs=2,max_repair_rounds=2,max_rows=10000,query_timeout_s=5.0,
                  summarise=False,notebook_dir=str(a.output/'schema_notebook'),disagreement_mode=a.mode)
    code_root=Path(__file__).resolve().parents[1]
    code_paths=list((code_root/'veriquery').rglob('*.py'))+[Path(__file__).resolve(),code_root/'benchmarks/bird.py',code_root/'training/freeze_sql_splits.py']
    identity={'scope':'SQL pipeline; no clarifier or LLM answer summarization',
              'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in {'resume','max_runtime_seconds'}},
              'config':asdict(config),'runtime':runtime_metadata(torch),'dataset_sha256':digest(a.data),
              'selection_sha256':digest(a.selection),'selected_ids':[{'db_id':e.db_id,'question_id':e.question_id} for e in examples],
              'database_sha256':{k:digest(v) for k,v in databases.items()},
              'adapter_sha256':{p.name:digest(p) for p in a.adapter.glob('*') if p.is_file()} if a.adapter else None,
              'code_sha256':{str(p.relative_to(code_root)):digest(p) for p in sorted(code_paths)},
              'decoding':{'draft_temperatures':[0,0.7,0.7],'draft_seeds':[6130,1,2],'checker_temperature':0.3,'seed_derivation':'sha256(client seed, messages) first32bits','top_p':1.0,'top_k':0},
              'scoring':{'mode':'set','timeout_seconds':5,'max_rows':10000}}
    records=load_progress(a.output,identity,a.resume)
    if len(records)==len(examples):
        (a.output/'summary.json').write_text(json.dumps(summarize(records),indent=2)+'\n');return
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer=AutoTokenizer.from_pretrained(a.model,revision=a.revision)
    model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype),
        device_map={'':0},torch_dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True)
    if a.adapter:
        model=PeftModel.from_pretrained(model,a.adapter)
    model.eval()
    client=LocalPipelineClient(hf_backend(model,tokenizer,torch,a.context_tokens),a.model,
                               a.context_tokens,a.max_output_tokens,a.max_calls)
    pipeline=VeriQuery(client,config)
    started=time.monotonic()
    with (a.output/'traces.jsonl').open('a') as out:
        for example in examples[len(records):]:
            if a.max_runtime_seconds and time.monotonic()-started >= a.max_runtime_seconds:
                break
            before=pipeline._usage();t0=time.monotonic();client.reset_question_budget()
            state=None;error=None
            try:
                state=pipeline.ask(example.question,databases[example.db_id],evidence=example.evidence)
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
            answers=(state or {}).get('answers',[])
            answer=answers[0] if answers else {}
            sql=answer.get('sql') or ''
            # Reference SQL enters only this separate scorer, after generation finishes.
            score=score_prediction(SQLiteExecutor(databases[example.db_id],timeout_seconds=5,max_rows=10000),sql,example.gold_sql)
            record={'db_id':example.db_id,'question_id':example.question_id,'question':example.question,
                    'sql':sql,'score':asdict(score),'confidence':answer.get('confidence',0.0),
                    'report_confidence':answer.get('report',{}).get('confidence','unverified'),
                    'agreement':answer.get('agreement',0.0),'candidates':answer.get('candidates',[]),
                    'verification_rounds':answer.get('verification_rounds',[]),
                    'diagnostics':[check.get('evidence',{}) for report in answer.get('verification_rounds',[])
                                   for check in report.get('checks',[]) if check.get('name')=='disagreement_diagnostics'],
                    'usage':pipeline._usage_delta(before),'seconds':time.monotonic()-t0,'error':error,
                    'model_calls_attempted':client.budget['calls']}
            out.write(json.dumps(record,default=str)+'\n');out.flush();os.fsync(out.fileno());records.append(record)
            print(json.dumps({'question_id':example.question_id,'status':score.status,'completed':len(records)}),flush=True)
    (a.output/'summary.json').write_text(json.dumps(summarize(records),indent=2)+'\n')


if __name__ == '__main__':
    main()
