"""Persistent fixed-batch scheduling, independent of the inference backend."""
import hashlib
import json
from pathlib import Path

from common import append_jsonl, read_jsonl, write_json


def batched_generation(examples, path, generate, *, batch_size, retry_batch_size,
                       cap, retry_cap, seed_offset=0, extras=None, lengths=None,
                       on_progress=lambda rows: None, recoverable_errors=(),
                       release_memory=lambda: None,on_memory_pressure=lambda details: None):
    """Save first attempts before queuing retries across the entire workload.

    Full immutable groups are regenerated after a partial write. This preserves
    sampled RNG assignment on resume; completed groups are never run again.
    Lengths must depend only on prompts, never answers or held-out rollouts.
    """
    if min(batch_size, retry_batch_size, cap, retry_cap) < 1:
        raise ValueError('Batch sizes and token caps must be positive')
    if extras is not None and len(extras) != len(examples):
        raise ValueError('One extra instruction is required per example')
    if lengths is not None and len(lengths) != len(examples):
        raise ValueError('One prompt length is required per example')
    ids = [x['id'] for x in examples]
    if len(set(ids)) != len(ids):
        raise ValueError('Example IDs must be unique')
    path = Path(path)
    initial_path = path.with_name(path.stem + '_initial.jsonl')
    plan_path = path.with_name(path.stem + '_generation.json')
    order = sorted(range(len(examples)), key=lambda i: (lengths[i], i)) if lengths is not None else list(range(len(examples)))
    signature = dict(version=1, examples=examples, extras=extras, lengths=lengths,
                     batch_size=batch_size, retry_batch_size=retry_batch_size,
                     cap=cap, retry_cap=retry_cap, seed_offset=seed_offset)
    plan = dict(version=1, fingerprint=hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
                initial_batches=[order[i:i+batch_size] for i in range(0, len(order), batch_size)])
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise RuntimeError('Generation plan changed; use a new run directory')
    else:
        if read_jsonl(path) or read_jsonl(initial_path):
            raise RuntimeError('Existing outputs have no generation plan; use a new run directory')
        write_json(plan_path, plan)
    existing = {r['id']: r for r in read_jsonl(path)}
    initial = {r['id']: r for r in read_jsonl(initial_path)}
    execution_path=path.with_name(path.stem+'_execution.json')
    splits=json.loads(execution_path.read_text()) if execution_path.exists() else {}

    def save(row):
        if row['id'] not in existing:
            append_jsonl(path, row)
            existing[row['id']] = row

    def invoke(indices, offset, limit, execution_key):
        signature=dict(indices=indices,offset=offset,cap=limit)
        split=splits.get(execution_key)
        if split is not None and split['signature']!=signature:
            raise RuntimeError('Memory recovery plan changed; use a new run directory')
        if split is None:
            failed=False
            try:
                rows = generate([examples[i] for i in indices],
                                [extras[i] for i in indices] if extras is not None else None,
                                seed_offset=offset, cap=limit)
            except recoverable_errors:
                saved=initial if execution_key.startswith('initial:') else existing
                # A partly written group must keep its original RNG assignment.
                if len(indices)==1 or any(ids[i] in saved for i in indices):raise
                failed=True
            # Leave the except block first, releasing traceback references to KV
            # cache tensors, before reclaiming memory or attempting smaller groups.
            if not failed:
                if [r['id'] for r in rows] != [ids[i] for i in indices]:
                    raise RuntimeError('Generation returned mismatched example IDs')
                return rows
            split=dict(signature=signature,at=len(indices)//2)
            splits[execution_key]=split
            write_json(execution_path,splits)
            on_memory_pressure(dict(key=execution_key,batch_size=len(indices),split_at=split['at']))
            release_memory()
        at=split['at']
        def child_seed(suffix):
            digest=hashlib.sha256(f'{offset}:{execution_key}:{suffix}'.encode()).hexdigest()
            return 2_000_000+int(digest[:7],16)
        rows=invoke(indices[:at],child_seed('L'),limit,execution_key+'/L')
        rows+=invoke(indices[at:],child_seed('R'),limit,execution_key+'/R')
        if [r['id'] for r in rows] != [ids[i] for i in indices]:
            raise RuntimeError('Generation returned mismatched example IDs')
        return rows

    for group_number, indices in enumerate(plan['initial_batches']):
        if not all(ids[i] in initial for i in indices):
            rows = invoke(indices, seed_offset + group_number * batch_size, cap,f'initial:{group_number}')
            for row in rows:
                if row['id'] not in initial:
                    append_jsonl(initial_path, row)
                    initial[row['id']] = row
        for i in indices:
            row = initial[ids[i]]
            if not row['truncated']:
                save({**row, 'total_compute_seconds': row['seconds']})
        on_progress(existing)

    # Membership uses ALL first attempts, including already completed retries.
    # Filtering by unfinished work here would change seeds/batchmates on resume.
    retry_indices = [i for i in order if initial[ids[i]]['truncated']]
    for start in range(0, len(retry_indices), retry_batch_size):
        indices = retry_indices[start:start+retry_batch_size]
        if all(ids[i] in existing for i in indices):
            continue
        rows = invoke(indices, seed_offset + 1_000_000 + start, retry_cap,f'retry:{start}')
        for row in rows:
            first = initial[row['id']]
            save({**row, 'initial_attempt': first,
                  'total_compute_seconds': first['seconds'] + row['seconds']})
        on_progress(existing)
    return [existing[x] for x in ids]
