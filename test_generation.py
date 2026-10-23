"""Scheduling regressions: batch retries, crash recovery, and RNG pairing."""
import pytest
from common import append_jsonl, read_jsonl
from generation import batched_generation


def workload(tmp_path, generate, **overrides):
    options=dict(batch_size=4,retry_batch_size=4,cap=10,retry_cap=20,
                 seed_offset=100,extras=[f'extra{i}' for i in range(10)])
    options.update(overrides)
    examples=[dict(id=str(i),question=str(i)) for i in range(10)]
    return batched_generation(examples,tmp_path/'outputs.jsonl',generate,**options)


def generator(calls):
    def generate(examples,extras,seed_offset,cap):
        ids=[x['id'] for x in examples]
        calls.append((ids,extras,seed_offset,cap))
        return [dict(id=x['id'],truncated=cap==10 and int(x['id'])%2==0,
                     seconds=2,seed=seed_offset,batch_ids=ids,correct=True) for x in examples]
    return generate


def test_retries_span_original_batches_and_preserve_order(tmp_path):
    calls=[]
    rows=workload(tmp_path,generator(calls))
    assert [c[0] for c in calls]==[['0','1','2','3'],['4','5','6','7'],['8','9'],['0','2','4','6'],['8']]
    assert calls[3][1]==['extra0','extra2','extra4','extra6']
    assert [r['id'] for r in rows]==list(map(str,range(10)))
    assert rows[0]['initial_attempt']['seed']==100
    assert rows[0]['total_compute_seconds']==4
    assert rows[1]['total_compute_seconds']==2
    assert calls[3][2]==1_000_100
    before=len(calls)
    assert workload(tmp_path,generator(calls))==rows
    assert len(calls)==before


def test_crash_during_retry_keeps_queue_and_seed(tmp_path):
    calls=[]
    normal=generator(calls)
    def crash(*args,**kwargs):
        rows=normal(*args,**kwargs)
        if kwargs['cap']==20:
            append_jsonl(tmp_path/'outputs.jsonl',{**rows[0],'initial_attempt':{'saved':True}})
            raise InterruptedError
        return rows
    with pytest.raises(InterruptedError):workload(tmp_path,crash)
    assert len(read_jsonl(tmp_path/'outputs_initial.jsonl'))==10
    calls.clear()
    rows=workload(tmp_path,normal)
    assert calls[0][0]==['0','2','4','6']
    assert calls[0][2]==1_000_100
    assert rows[0]['initial_attempt']=={'saved':True}
    assert len(read_jsonl(tmp_path/'outputs.jsonl'))==10


def test_partial_initial_group_recovers_without_losing_attempts(tmp_path):
    calls=[]
    normal=generator(calls)
    def crash(*args,**kwargs):
        rows=normal(*args,**kwargs)
        append_jsonl(tmp_path/'outputs_initial.jsonl',rows[0])
        raise InterruptedError
    with pytest.raises(InterruptedError):workload(tmp_path,crash)
    calls.clear()
    workload(tmp_path,normal)
    assert calls[0][0]==['0','1','2','3']
    assert len(read_jsonl(tmp_path/'outputs_initial.jsonl'))==10


def test_length_grouping_keeps_extras_aligned_and_refuses_plan_change(tmp_path):
    calls=[]
    rows=workload(tmp_path,generator(calls),lengths=list(reversed(range(10))))
    assert calls[0][0]==['9','8','7','6']
    assert calls[0][1]==['extra9','extra8','extra7','extra6']
    assert rows[0]['id']=='0'
    with pytest.raises(RuntimeError,match='plan changed'):
        workload(tmp_path,generator(calls),retry_batch_size=8)


def test_refuses_legacy_outputs_without_plan(tmp_path):
    append_jsonl(tmp_path/'outputs.jsonl',dict(id='0'))
    with pytest.raises(RuntimeError,match='no generation plan'):
        workload(tmp_path,generator([]))


def test_retry_attempt_stays_final_when_still_truncated(tmp_path):
    def generate(examples,extras,seed_offset,cap):
        return [dict(id=x['id'],truncated=True,seconds=1,seed=seed_offset) for x in examples]
    rows=workload(tmp_path,generate)
    assert all(r['truncated'] and 'initial_attempt' in r for r in rows)
    assert len(read_jsonl(tmp_path/'outputs.jsonl'))==10


def test_empty_workload_and_invalid_batch_size(tmp_path):
    rows=batched_generation([],tmp_path/'empty.jsonl',generator([]),batch_size=4,
                            retry_batch_size=8,cap=10,retry_cap=20)
    assert rows==[]
    with pytest.raises(ValueError,match='positive'):
        workload(tmp_path,generator([]),retry_batch_size=0)


class BudgetError(RuntimeError):pass


def test_allocator_rejection_splits_without_shortening_outputs(tmp_path):
    calls=[];released=[]
    normal=generator(calls)
    def limited(examples,*args,**kwargs):
        if len(examples)>2:raise BudgetError('Application cap')
        return normal(examples,*args,**kwargs)
    rows=workload(tmp_path,limited,recoverable_errors=(BudgetError,),release_memory=lambda:released.append(True))
    assert len(rows)==10 and released
    assert all(len(call[0])<=2 for call in calls)
    assert {call[3] for call in calls}=={10,20}
    assert rows[0]['initial_attempt']['batch_ids']==['0','1']
    assert (tmp_path/'outputs_execution.json').exists()


def test_split_plan_keeps_grouping_and_seeds_after_crash(tmp_path):
    calls=[]
    normal=generator(calls)
    def crash(examples,*args,**kwargs):
        if len(examples)>2:raise BudgetError
        if [x['id'] for x in examples]==['2','3']:raise InterruptedError
        return normal(examples,*args,**kwargs)
    with pytest.raises(InterruptedError):workload(tmp_path,crash,recoverable_errors=(BudgetError,))
    first=calls[0]
    calls.clear()
    # More VRAM is now available, but the stored subgroup is still replayed.
    workload(tmp_path,normal,recoverable_errors=(BudgetError,))
    assert calls[0]==first
    assert calls[1][0]==['2','3']


def test_does_not_retry_driver_fault_or_single_request_budget_failure(tmp_path):
    def fault(*args,**kwargs):raise RuntimeError('Device reset')
    with pytest.raises(RuntimeError,match='Device reset'):
        workload(tmp_path,fault,recoverable_errors=(BudgetError,))
    def limited(*args,**kwargs):raise BudgetError('Even one request is too large')
    with pytest.raises(BudgetError):
        workload(tmp_path,limited,recoverable_errors=(BudgetError,))


def test_partial_saved_group_cannot_silently_change_seed(tmp_path):
    normal=generator([])
    def partial(*args,**kwargs):
        append_jsonl(tmp_path/'outputs_initial.jsonl',normal(*args,**kwargs)[0])
        raise InterruptedError
    with pytest.raises(InterruptedError):workload(tmp_path,partial)
    def limited(*args,**kwargs):raise BudgetError
    with pytest.raises(BudgetError):workload(tmp_path,limited,recoverable_errors=(BudgetError,))
    assert not (tmp_path/'outputs_execution.json').exists()
