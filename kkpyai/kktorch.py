import functools
import operator
import os
import os.path as osp
import typing
# 3rd party
import kkpyutil as util
import torch as tc
import matplotlib.pyplot as plt


# region globals

def find_fastest_device():
    """
    - Apple Silicon uses Apple's own Metal Performance Shaders (MPS) instead of CUDA
    """
    if util.PLATFORM == 'Darwin':
        return 'mps' if tc.backends.mps.is_available() else 'cpu'
    if tc.cuda.is_available():
        return 'cuda'
    return 'cpu'


class Loggable:
    def __init__(self, logger=None):
        self.logger = logger or util.glogger

# endregion


# region tensor ops

class TensorFactory(Loggable):
    def __init__(self, device=None, dtype=tc.float32, requires_grad=False, logger=None):
        super().__init__(logger)
        self.device = tc.device(device) if device else find_fastest_device()
        self.dtype = dtype
        self.requires_grad = requires_grad

    def init(self, device: str = '', dtype=tc.float32, requires_grad=False):
        self.device = tc.device(device) if device else find_fastest_device()
        self.dtype = dtype
        self.requires_grad = requires_grad

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

# region model


class Model(Loggable):
    def __init__(self, model, lossfn_name='L1Loss', optm_name='SGD', learning_rate=0.01, device_name=None, logger=None):
        super().__init__(logger)
        self.device = device_name or find_fastest_device()
        self.model = model.to(self.device)
        self.lossFunction = eval(f'tc.nn.{lossfn_name}()')
        self.optimizer = eval(f'tc.optim.{optm_name}(self.model.parameters(), lr={learning_rate})')

    def set_lossfunction(self, lossfn_name='L1Loss'):
        """
        - ref: https://pytorch.org/docs/stable/nn.html#loss-functions
        """
        self.lossFunction = eval(f'nn.{lossfn_name}()')

    def set_optimizer(self, opt_name='SGD', learning_rate=0.01):
        """
        - ref: https://pytorch.org/docs/stable/optim.html#algorithms
        """
        self.optimizer = eval(f'tc.optim.{opt_name}(self.model.parameters(), lr={learning_rate})')

    def train(self, train_set, test_set=None, n_epochs=1000, seed=42, verbose=False, log_every_n_epochs=100):
        tc.manual_seed(seed)
        X_train = train_set['data'].to(self.device)
        y_train = train_set['labels'].to(self.device)
        pred = {'preds': None, 'loss': None}
        if test_set:
            X_test = test_set['data'].to(self.device)
            y_test = test_set['labels'].to(self.device)
        for epoch in range(n_epochs):
            # Training
            # - train mode is on by default after construction
            self.model.train()
            # - forward pass
            y_pred = self.model(X_train)
            # - compute loss
            loss = self.lossFunction(y_pred, y_train)
            # - reset grad before backpropagation
            self.optimizer.zero_grad()
            # - backpropagation
            loss.backward()
            # - update weights and biases
            self.optimizer.step()
            if test_set:
                pred = self.evaluate(test_set)
            if verbose and epoch % log_every_n_epochs == 0:
                msg = f"Epoch: {epoch} | Train Loss: {loss} | Test Loss: {pred['loss']}" if test_set else f"Epoch: {epoch} | Train Loss: {loss}"
                self.logger.info(msg)
        # final test predictions
        return pred

    def evaluate(self, test_set):
        X_test = test_set['data'].to(self.device)
        y_test = test_set['labels'].to(self.device)
        # Testing
        # - eval mode is on by default after construction
        self.model.eval()
        # - forward pass
        with tc.inference_mode():
            test_pred = self.model(X_test)
            # - compute loss
            test_loss = self.lossFunction(test_pred, y_test)
        return {'pred': test_pred, 'loss': test_loss}

    def predict(self, test_set):
        """
        - we predict when test_set has no labels
        - otherwise we evaluate the model
        """
        X_test = test_set['data'].to(self.device)
        test_set['labels'] = test_set['labels'].to(self.device)
        # Testing
        # - eval mode is on by default after construction
        self.model.eval()
        # - forward pass
        with tc.inference_mode():
            predictions = self.model(X_test)
        return predictions.to(self.device)

    def save(self, model_basename=None, optimized=True):
        ext = '.pth' if optimized else '.pt'
        path = self._compose_model_name(model_basename, ext)
        os.makedirs(osp.dirname(path), exist_ok=True)
        tc.save(self.model.state_dict(), path)

    def load(self, model_basename=None, optimized=True):
        ext = '.pth' if optimized else '.pt'
        path = self._compose_model_name(model_basename, ext)
        self.model.load_state_dict(tc.load(path))

    @staticmethod
    def _compose_model_name(model_basename, ext):
        return osp.join(util.get_platform_tmp_dir(), 'torch', f'{model_basename}.{ext}')

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
