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

    def init(self, device=None, dtype=tc.float32, requires_grad=False):
        gpu = 'mps' if util.PLATFORM == 'Darwin' else 'cuda'
        self.device = tc.device(device or (gpu if self.can_use_gpu() else 'cpu'))
        self.dtype = dtype
        self.requires_grad = requires_grad

    @staticmethod
    def can_use_gpu():
        """
        - Apple Silicon uses Apple's own Metal Performance Shaders (MPS) instead of CUDA
        """
        return tc.backends.mps.is_available() if util.PLATFORM == 'Darwin' else tc.cuda.is_available()

    def ramp(self, size: typing.Union[list, tuple], start=1):
        """
        - ramp is easier to understand than random numbers
        - so they can come in handy for debugging and test-drive
        """
        end = start + functools.reduce(operator.mul, size)
        return tc.arange(start, end).reshape(*size).to(self.device, self.dtype, self.requires_grad)

    def rand_repro(self, size: typing.Union[list, tuple], seed=42):
        """
        - to reproduce a random tensor n times, simply call this method with the same seed (flavor of randomness)
        - to start a new reproducible sequence, call this method with a new seed
        """
        tc.manual_seed(seed)
        return tc.rand(size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)


# endregion
