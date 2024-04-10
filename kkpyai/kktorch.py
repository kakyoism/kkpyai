import functools
import operator
import os
import os.path as osp
import typing
# 3rd party
import kkpyutil as util
import torch as tc
import matplotlib.pyplot as plt


# region tensor ops

class TensorFactory:
    def __init__(self, device=None, dtype=tc.float32, requires_grad=False):
        self.device = tc.device(device) if device else self.find_fastest_device()
        self.dtype = dtype
        self.requires_grad = requires_grad

    def init(self, device: str = '', dtype=tc.float32, requires_grad=False):
        self.device = tc.device(device) if device else self.find_fastest_device()
        self.dtype = dtype
        self.requires_grad = requires_grad

    @staticmethod
    def find_fastest_device():
        """
        - Apple Silicon uses Apple's own Metal Performance Shaders (MPS) instead of CUDA
        """
        if util.PLATFORM == 'Darwin':
            return 'mps' if tc.backends.mps.is_available() else 'cpu'
        if tc.cuda.is_available():
            return 'cuda'
        return 'cpu'

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
        if self.device == 'cuda':
            tc.cuda.manual_seed(seed)
        else:
            tc.manual_seed(seed)
        return tc.rand(size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)


# endregion


# region dataset

def split_dataset(data, labels, train_ratio=0.8, validation_ratio=0):
    """
    - split dataset into training and testing sets
    """
    split_train = int(train_ratio * len(data))
    split_val = int(validation_ratio * len(data))
    X_train, y_train = data[:split_train], labels[:split_train]
    X_test, y_test = data[split_train:], labels[split_train:]
    if validation_ratio:
        X_val, y_val = X_test[:split_val], y_test[:split_val]
        X_test, y_test = X_test[split_val:], y_test[split_val:]
        train_set = {'data': X_train, 'labels': y_train}
        test_set = {'data': X_test, 'labels': y_test}
        validation_set = {'data': X_val, 'labels': y_val}
        return train_set, test_set, validation_set
    train_set = {'data': X_train, 'labels': y_train}
    test_set = {'data': X_test, 'labels': y_test}
    return train_set, test_set, validation_ratio

# endregion


# region visualization

class Plot:
    def __init__(self, *args, **kwargs):
        self.legendConfig = {'prop': {'size': 14}}
        self.useBlocking = True

    def plot_predictions(self, train_set, test_set, predictions=None):
        """
        - sets contain data and labels
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(train_set['data'], train_set['labels'], s=4, color='blue', label='Training Data')
        ax.scatter(test_set['data'], test_set['labels'], s=4, color='green', label='Testing Data')
        if predictions is not None:
            ax.scatter(test_set['data'], predictions, s=4, color='red', label='Predictions')
        ax.legend(prop=self.legendConfig['prop'])
        plt.show(block=self.useBlocking)

    def block(self):
        self.useBlocking = True

    def unblock(self):
        self.useBlocking = False

    @staticmethod
    def export_png(path=osp.join(util.get_platform_home_dir(), 'Desktop', 'plot.png')):
        os.makedirs(osp.dirname(path), exist_ok=True)
        plt.savefig(path, format='png')

    @staticmethod
    def export_svg(path):
        os.makedirs(osp.dirname(path), exist_ok=True)
        plt.savefig(path, format='svg')

    @staticmethod
    def close():
        plt.close()

# endregion


def test():
    pass


if __name__ == '__main__':
    test()
