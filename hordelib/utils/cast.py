from functools import wraps
from typing import TypeVar

from torch import amp, dtype, float16, no_grad

T = TypeVar("T")


def autocast_cuda(func: T, dtype: dtype = float16) -> T:
    @wraps(func)
    def wrap(*args, **kwargs):
        return amp.autocast(device_type="cuda", dtype=dtype)(no_grad()(func))(
            *args,
            **kwargs,
        )

    return wrap


def autocast_cpu(func: T, dtype: dtype = float16) -> T:
    @wraps(func)
    def wrap(*args, **kwargs):
        return amp.autocast(device_type="cpu", dtype=dtype)(no_grad()(func))(
            *args,
            **kwargs,
        )

    return wrap
