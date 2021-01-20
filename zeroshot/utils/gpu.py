from functools import wraps

import torch


def move_args_to_device(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        args = [
            arg.to(self.device) if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]
        kwargs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        return func(self, *args, **kwargs)

    return inner
