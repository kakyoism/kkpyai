import functools
import operator
import typing
# 3rd party
import kkpyutil as util
import torch as tc


# region tensor ops

class TensorFactory:
    def __init__(self, device='cpu', dtype=tc.float32, requires_grad=False):
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad

    def init(self, device='cpu', dtype=tc.float32, requires_grad=False):
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad

    def ramp(self, size: typing.Union[list, tuple], start=1):
        end = start + functools.reduce(operator.mul, size)
        return tc.arange(start, end).reshape(*size).to(self.device, self.dtype, self.requires_grad)

    def rand_repro(self, size: typing.Union[list, tuple], seed=42):
        """
        - to reproduce a random tensor n times, simply call this method with the same seed
        - to start a new reproducible sequence, call this method with a new seed
        """
        tc.manual_seed(seed)
        return tc.rand(size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)


# endregion
