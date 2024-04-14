import functools
import operator
import os
import os.path as osp
import typing
# 3rd party
import kkpyutil as util
import torch as tc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# region globals

def find_fast_device():
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
        self.device = tc.device(device) if device else find_fast_device()
        self.dtype = dtype
        self.requires_grad = requires_grad

    def init(self, device: str = '', dtype=tc.float32, requires_grad=False):
        self.device = tc.device(device) if device else find_fast_device()
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

def split_dataset(data, labels, train_ratio=0.8, random_seed=42, ):
    """
    - split dataset into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_ratio, random_state=random_seed)
    train_set = {'data': X_train, 'labels': y_train}
    test_set = {'data': X_test, 'labels': y_test}
    return train_set, test_set


# endregion

# region model


class Regressor(Loggable):
    LossFuncType = typing.Callable[[tc.Tensor, tc.Tensor], tc.Tensor]

    def __init__(self, model, loss_fn: typing.Union[str, LossFuncType] = 'L1', optm='SGD', learning_rate=0.01, device_name=None, logger=None):
        super().__init__(logger)
        self.device = device_name or find_fast_device()
        self.model = model.to(self.device)
        self.lossFunction = eval(f'tc.nn.{loss_fn}Loss()') if isinstance(loss_fn, str) else loss_fn
        self.optimizer = eval(f'tc.optim.{optm}(self.model.parameters(), lr={learning_rate})')
        self.plot = Plot()

    def set_lossfunction(self, loss_fn: typing.Union[str, LossFuncType] = 'L1Loss'):
        """
        - ref: https://pytorch.org/docs/stable/nn.html#loss-functions
        """
        self.lossFunction = eval(f'nn.{loss_fn}()') if isinstance(loss_fn, str) else loss_fn

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
        losses = {'train': [], 'test': []}
        for epoch in range(n_epochs):
            # Training
            # - train mode is on by default after construction
            self.model.train()
            # - forward pass
            y_pred = self._forward_pass(X_train)
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
                if verbose:
                    losses['test'].append(pred['loss'].cpu().detach().numpy())
            if verbose:
                losses['train'].append(loss.cpu().detach().numpy())
            if verbose and epoch % log_every_n_epochs == 0:
                msg = f"Epoch: {epoch} | Train Loss: {loss} | Test Loss: {pred['loss']}" if test_set else f"Epoch: {epoch} | Train Loss: {loss}"
                self.logger.info(msg)
        if verbose:
            # plot predictions
            self.plot.unblock()
            self.plot.plot_predictions(train_set, test_set, pred['pred'])
            self.plot.plot_learning(losses['train'], losses['test'])
        # final test predictions
        return pred

    def _forward_pass(self, X):
        return self.model(X)

    def evaluate(self, test_set, verbose=False):
        """
        - test_set must contain ground-truth labels
        """
        X_test = test_set['data'].to(self.device)
        y_test = test_set['labels'].to(self.device)
        # Testing
        # - eval mode is on by default after construction
        self.model.eval()
        # - forward pass
        with tc.inference_mode():
            test_pred = self.model(X_test)
            test_loss = self.lossFunction(test_pred, y_test)
        if verbose:
            self.logger.info(f'Test Loss: {test_loss}')
            self.plot.unblock()
            self.plot.plot_predictions(None, test_set, test_pred)
        return {'pred': test_pred, 'loss': test_loss}

    def predict(self, test_set, for_plot_only=False):
        """
        - test_set can have no labels
        """
        dev = 'cpu' if for_plot_only else self.device
        X_test = test_set['data'].to(dev)
        # Testing
        # - eval mode is on by default after construction
        self.model.eval()
        # - forward pass
        with tc.inference_mode():
            y_pred = self.model(X_test)
        test_set['labels'] = y_pred.to(dev)
        return test_set['labels']

    def close_plot(self):
        self.plot.close()

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
        return osp.join(util.get_platform_tmp_dir(), 'torch', f'{model_basename}{ext}')


class BiClassifier(Regressor):
    def __init__(self, model, loss_fn: typing.Union[str, Regressor.LossFuncType] = 'BCE', optm='SGD', learning_rate=0.01, device_name=None, logger=None):
        super().__init__(model, loss_fn, optm, learning_rate, device_name, logger)

    def train(self, train_set, test_set=None, n_epochs=1000, seed=42, verbose=False, log_every_n_epochs=100):
        tc.manual_seed(seed)
        X_train = train_set['data'].to(self.device)
        y_train = train_set['labels'].to(self.device)
        test_pred = {'preds': None, 'loss': None}
        losses = {'train': [], 'test': [], 'accuracy': []}
        for epoch in range(n_epochs):
            # Training
            # - train mode is on by default after construction
            self.model.train()
            y_pred = self._forward_pass(X_train)
            # - compute loss
            #   - we don't support BCEWithLogitsLoss for consistency
            #   - because activation is a hyperparameter
            #   - and all but BCEWithLogitsLoss require explicit activation
            loss = self.lossFunction(y_pred, y_train)
            acc = self.accuracy(y_true=y_train, y_pred=y_pred)
            # - reset grad before backpropagation
            self.optimizer.zero_grad()
            # - backpropagation
            loss.backward()
            # - update weights and biases
            self.optimizer.step()
            if test_set:
                test_pred = self.evaluate(test_set)
                if verbose:
                    losses['test'].append(test_pred['loss'].cpu().detach().numpy())
            if verbose:
                losses['train'].append(loss.cpu().detach().numpy())
                acc = self.accuracy(y_true=y_train, y_pred=y_pred)
                losses['accuracy'].append(acc)
            if verbose and epoch % log_every_n_epochs == 0:
                msg = f"Epoch: {epoch} | Train Loss: {loss} | Train Accuracy: {losses['accuracy']} | Test Loss: {test_pred['loss']} | Test Accuracy: {test_pred['accuracy']}%" if test_set else f"Epoch: {epoch} | Train Loss: {loss} | Train Accuracy: {losses['accuracy']}%"
                self.logger.info(msg)
        if verbose:
            # plot predictions
            self.plot.unblock()
            self.plot_decision_boundary(train_set)
            self.plot.plot_learning(losses['train'], losses['test'])
        # final test predictions
        return test_pred

    def _forward_pass(self, X):
        # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device
        y_logits = self.model(X).squeeze()
        # turn logits -> pred probs -> pred labels
        return self._logits_to_labels(y_logits)

    @staticmethod
    def _logits_to_labels(y_logits):
        """
        - logits -> pred probs -> pred labels
        - raw model output must be activated to get probabilities then labels
        - special activators, e.g., softmax, must override this method
        """
        return tc.round(tc.sigmoid(y_logits))

    def evaluate(self, test_set, verbose=False):
        """
        - test_set must contain ground-truth labels
        """
        X_test = test_set['data'].to(self.device)
        y_test = test_set['labels'].to(self.device)
        # Testing
        # - eval mode is on by default after construction
        self.model.eval()
        with tc.inference_mode():
            # 1. Forward pass
            test_logits = self.model(X_test).squeeze()
            test_pred = tc.round(tc.sigmoid(test_logits))
            # 2. Calculate loss/accuracy
            test_loss = self.lossFunction(test_logits, y_test)
            test_acc = self.accuracy(y_true=y_test, y_pred=test_pred)
        if verbose:
            self.logger.info(f'Test Loss: {test_loss} | Test Accuracy: {test_acc}%')
            self.plot.unblock()
            self.plot.plot_predictions(None, test_set, test_pred)
        return {'pred': test_pred, 'loss': test_loss, 'accuracy': test_acc}

    @staticmethod
    def accuracy(y_pred, y_true):
        """
        - in percentage
        """
        return tc.sum(tc.eq(y_pred, y_true)).item() / len(y_true) * 100

    def plot_decision_boundary(self, dataset):
        """
        - assume 2D dataset
        - create special dataset and run model on it for visualization (2D)
        - ref: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py
        """
        # Put everything to CPU (works better with NumPy + Matplotlib)
        self.model.to("cpu")
        X, y = dataset['data'].to("cpu"), dataset['labels'].to("cpu")
        # Setup prediction boundaries and grid
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        n_data = 100
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_data+1), np.linspace(y_min, y_max, n_data+1))
        # Make features
        X_plottable = tc.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()
        # Make predictions
        plot_set = {'data': X_plottable, 'labels': tc.zeros(X_plottable.shape[0]).to('cpu')}
        y_logits = self.predict(plot_set, for_plot_only=True)
        # # Test for multi-class or binary and adjust logits to prediction labels
        # if len(tc.unique(y)) > 2:
        #     y_pred = tc.softmax(y_logits, dim=1).argmax(dim=1)  # multi-class
        # else:
        #     y_pred = tc.round(tc.sigmoid(y_logits))  # binary
        y_pred = self._logits_to_labels(y_logits)
        # Reshape preds and plot
        y_pred = y_pred.reshape(xx.shape).detach().numpy()
        self.plot.plot_decision_boundary(dataset, y_pred)


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
        if train_set:
            ax.scatter(train_set['data'].cpu(), train_set['labels'].cpu(), s=4, color='blue', label='Training Data')
        if test_set:
            ax.scatter(test_set['data'].cpu(), test_set['labels'].cpu(), s=4, color='green', label='Testing Data')
        if predictions is not None:
            ax.scatter(test_set['data'].cpu(), predictions.cpu(), s=4, color='red', label='Predictions')
        ax.legend(prop=self.legendConfig['prop'])
        plt.show(block=self.useBlocking)

    def plot_learning(self, train_losses, test_losses=None):
        fig, ax = plt.subplots(figsize=(10, 7))
        if train_losses is not None:
            ax.plot(train_losses, label='Training Loss', color='blue')
        if test_losses is not None:
            ax.plot(test_losses, label='Testing Loss', color='orange')
        ax.set_title('Learning Curves')
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend(prop=self.legendConfig['prop'])
        plt.show(block=self.useBlocking)

    def plot_decision_boundary(self, dataset2d, predictions):
        # Setup prediction boundaries and grid
        epsilon = 0.1
        x_min, x_max = dataset2d['data'][:, 0].min() - epsilon, dataset2d['data'][:, 0].max() + epsilon
        y_min, y_max = dataset2d['data'][:, 1].min() - epsilon, dataset2d['data'][:, 1].max() + epsilon
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
        fig, ax = plt.subplots(figsize=(10, 7))
        # draw colour-coded predictions on meshgrid
        ax.contourf(xx, yy, predictions, cmap=plt.cm.RdYlBu, alpha=0.7)
        ax.scatter(dataset2d['data'][:, 0], dataset2d['data'][:, 1], c=dataset2d['labels'], s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

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
