"""Reject oversized PyTorch allocations before they reach the ROCm driver."""
import os
import re

GIB = 2**30


class VRAMBudgetExceeded(RuntimeError):
    """The application must reduce its workload, without retrying a driver fault."""


def configure_allocator(fraction=0.8):
    """Call before importing torch. Set all aliases: HIP precedence differs by build."""
    if not 0 < fraction < 1:
        raise ValueError('VRAM fraction must be between zero and one')
    previous = next((os.environ[k] for k in (
        'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_HIP_ALLOC_CONF', 'PYTORCH_ALLOC_CONF')
        if os.environ.get(k)), '')
    options = {}
    for item in re.split(r',(?=[A-Za-z_]+\s*:)', previous):
        if item.strip():
            key, separator, value = item.partition(':')
            if not separator:raise ValueError('Malformed allocator configuration')
            options[key.strip()] = value.strip()
    options.update(backend='native')
    if os.environ.get('QRL_PORTABLE_CUDA') == '1':
        # CUDA wheels need the public runtime API in apply_budget(); these
        # allocator flags are specific to the local ROCm installation.
        options.pop('per_process_memory_fraction', None)
        options.pop('throw_on_cudamalloc_oom', None)
    else:
        options.update(per_process_memory_fraction=str(float(fraction)),
                       throw_on_cudamalloc_oom='True')
    settings = ','.join(f'{key}:{value}' for key,value in options.items())
    for key in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_HIP_ALLOC_CONF'):
        os.environ[key] = settings
    return settings


def apply_budget(torch, device=None, fraction=0.8, reserve_gib=2.0):
    """Keep both a fraction of total VRAM and headroom relative to current users."""
    if not 0 < fraction < 1 or reserve_gib <= 0:
        raise ValueError('Use a fraction below one and a positive VRAM reserve')
    free,total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    limit = min(int(total*fraction), int(free+reserved-reserve_gib*GIB))
    if limit <= torch.cuda.memory_allocated(device):
        raise VRAMBudgetExceeded('Insufficient free VRAM to preserve the configured reserve')
    torch.cuda.set_per_process_memory_fraction(float(limit/total),device)
    return dict(limit_bytes=limit,reserve_bytes=int(reserve_gib*GIB),total_bytes=total)


def check_headroom(torch, device=None, reserve_gib=2.0):
    free,_ = torch.cuda.mem_get_info(device)
    if free < reserve_gib*GIB:
        raise VRAMBudgetExceeded('Free VRAM reached the reserve; reduce this batch')
