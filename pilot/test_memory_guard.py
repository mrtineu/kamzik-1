from types import SimpleNamespace
import pytest
from memory_guard import GIB,VRAMBudgetExceeded,configure_allocator,apply_budget,check_headroom


def test_allocator_cap_and_rejection_flag_cover_hip_aliases(monkeypatch):
    for key in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_HIP_ALLOC_CONF'):
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv('PYTORCH_HIP_ALLOC_CONF','max_split_size_mb:128,roundup_power2_divisions:[256:1,512:2]')
    settings=configure_allocator(.8)
    import os
    assert all(os.environ[key]==settings for key in (
        'PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_HIP_ALLOC_CONF'))
    assert 'throw_on_cudamalloc_oom:True' in settings
    assert 'per_process_memory_fraction:0.8' in settings
    assert 'backend:native' in settings
    assert 'roundup_power2_divisions:[256:1,512:2]' in settings


def test_budget_preserves_headroom_for_other_processes():
    caps=[]
    cuda=SimpleNamespace(mem_get_info=lambda device:(5*GIB,16*GIB),
        memory_reserved=lambda device:1*GIB,memory_allocated=lambda device:GIB//2,
        set_per_process_memory_fraction=lambda fraction,device:caps.append(fraction))
    budget=apply_budget(SimpleNamespace(cuda=cuda),reserve_gib=2)
    assert budget['limit_bytes']==4*GIB
    assert caps==[.25]


def test_portable_cuda_uses_runtime_budget_without_rocm_flags(monkeypatch):
    monkeypatch.setenv('QRL_PORTABLE_CUDA', '1')
    for key in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_HIP_ALLOC_CONF'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('PYTORCH_ALLOC_CONF',
        'max_split_size_mb:128,per_process_memory_fraction:0.8,throw_on_cudamalloc_oom:True')
    settings = configure_allocator(.8)
    assert settings == 'max_split_size_mb:128,backend:native'


def test_guard_rejects_before_any_more_allocations():
    cuda=SimpleNamespace(mem_get_info=lambda device:(GIB,16*GIB),
        memory_reserved=lambda device:0,memory_allocated=lambda device:0,
        set_per_process_memory_fraction=lambda *args:pytest.fail('Budget must fail before updating allocator'))
    with pytest.raises(VRAMBudgetExceeded):apply_budget(SimpleNamespace(cuda=cuda))
    with pytest.raises(VRAMBudgetExceeded):check_headroom(SimpleNamespace(cuda=cuda))


def test_invalid_fraction_and_reserve():
    with pytest.raises(ValueError):configure_allocator(1)
    with pytest.raises(ValueError):apply_budget(None,reserve_gib=0)
