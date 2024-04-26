import functools
import operator
import os
import os.path as osp
import pprint as pp
import typing
# 3rd party
import kkpyutil as util
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from timeit import default_timer as perf_timer
import torch as tc
import torch.utils.data as tud
import torchmetrics as tm
import torchvision as tcv
from tqdm.auto import tqdm


# region globals

def probe_fast_device():
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
        self.device = tc.device(device) if device else probe_fast_device()
        self.dtype = dtype
        self.requires_grad = requires_grad

    def init(self, device: str = '', dtype=tc.float32, requires_grad=False):
        self.device = tc.device(device) if device else probe_fast_device()
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

    def invalids(self, size: typing.Union[list, tuple], value=-1):
        return tc.full(size, value, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)


# endregion


# region dataset


def retrieve_vision_trainset(data_cls=tcv.datasets.FashionMNIST, local_dir=osp.join(util.get_platform_appdata_dir(), 'torch'), transform=tcv.transforms.ToTensor(), target_transform=None):
    """
    - FashionMNIST is a drop-in replacement for MNIST
    - images come as PIL format, we want to turn into Torch tensors
    - ref: https://pytorch.org/vision/stable/datasets.html#fashion-mnist
    """
    return data_cls(local_dir, train=True, download=True, transform=transform, target_transform=target_transform)


def retrieve_vision_testset(data_cls=tcv.datasets.FashionMNIST, local_dir=osp.join(util.get_platform_appdata_dir(), 'torch'), transform=tcv.transforms.ToTensor(), target_transform=None):
    return data_cls(local_dir, train=False, download=True, transform=transform, target_transform=target_transform)


def inspect_dataset(dataset, block=True, cmap='gray'):
    """
    - list key dataset properties for debug, e.g.,
    - shapes and sizes are crucial to later matrix ops for model training and visualization
    """
    assert len(dataset) > 0, 'Dataset is empty'
    data, label = dataset[0]
    pp.pprint(f"""\
- dataset: {type(dataset).__name__}
- data shape: {data.shape}
- label shape: {label.shape if isinstance(label, tc.Tensor) else None}
- data type: {data.dtype}
- label type: {label.dtype if isinstance(label, tc.Tensor) else type(label)}
- data size: {len(dataset.data)}
- label size: {len(dataset.targets)}
- label classes: {dataset.classes}
- first data point: data: {data}, label: {label}
""")
    # if data is a PIL image or numpy array, show as image
    data_disp = data.numpy() if isinstance(data, tc.Tensor) else data
    fig, ax = plt.subplots(1, 1)
    ax.imshow(data_disp.squeeze(), cmap=cmap)
    ax.set_title(dataset.classes[label])
    plt.axis("Off")
    plt.show(block=block)


class DataProxyBase(tud.Dataset):
    def __init__(self, data, targets=None, data_dtype=tc.float32, target_dtype=tc.float32, device=None):
        """
        - initializes the dataset.
        - must instantiate train and test sets separately with this class
        - tc.Dataset offers train/test sets separately, so no split is needed
        - split_train_test() only works for loose data, e.g., tensors or numpy arrays
        """
        self.data = data
        self.targets = targets
        # Check if the data is already a PyTorch dataset
        self.isTorchDataset = isinstance(data, tud.Dataset)
        if not self.isTorchDataset:
            # Ensure data (numpy?) is a tensor for consistency
            if not isinstance(data, tc.Tensor):
                self.data = tc.tensor(data, dtype=data_dtype)
            if targets is not None and not isinstance(targets, tc.Tensor):
                self.targets = tc.tensor(targets, dtype=target_dtype)
        if device:
            self.data = self.data.to(device)
            if self.targets is not None:
                self.targets = self.targets.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        target = self.targets[idx]
        return item, target

    @staticmethod
    def create(data, targets=None, data_dtype=tc.float32, target_dtype=tc.float32, device=None):
        import torchvision.datasets.vision as tdv
        if isinstance(data, tdv.VisionDataset):
            return ImageDataProxy(data, device=device)
        return DataProxy(data, targets, data_dtype, target_dtype, device)

    def split_train_test(self, train_ratio=0.8, random_seed=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.targets, train_size=train_ratio, random_state=random_seed)
        return DataProxy(X_train, y_train), DataProxy(X_test, y_test)

    @staticmethod
    def use_device(X, y, device):
        return X.to(device), y.to(device)


class DataProxy(DataProxyBase):
    def __init__(self, data, targets, data_dtype=tc.float32, target_dtype=tc.float32, device=None):
        super().__init__(data, targets)
        self.data = data
        self.targets = targets
        # Ensure data (numpy?) is a tensor for consistency
        if not isinstance(data, tc.Tensor):
            self.data = tc.tensor(data, dtype=data_dtype)
        if targets is not None and not isinstance(targets, tc.Tensor):
            self.targets = tc.tensor(targets, dtype=target_dtype)
        if device:
            self.data = self.data.to(device)
            if self.targets is not None:
                self.targets = self.targets.to(device)


class ImageDataProxy(DataProxyBase):
    def __init__(self, dataset, device=None):
        super().__init__(dataset, None, device)
        self.data = tc.stack([img for img, label in dataset])
        self.targets = tc.tensor([label for img, label in dataset], dtype=tc.long)


# endregion


# region model

class Regressor(Loggable):
    LossFuncType = typing.Callable[[tc.Tensor, tc.Tensor], tc.Tensor]

    def __init__(self, model, loss_fn: typing.Union[str, LossFuncType] = 'L1', optm='SGD', learning_rate=0.01, batch_size=32, shuffle=True, device_name=None, logger=None, log_every_n_epochs=0):
        super().__init__(logger)
        self.device = device_name or probe_fast_device()
        self.model = model.to(self.device)
        self.lossFunction = eval(f'tc.nn.{loss_fn}Loss()') if isinstance(loss_fn, str) else loss_fn
        self.optimizer = eval(f'tc.optim.{optm}(self.model.parameters(), lr={learning_rate})')
        self.batchSize = batch_size
        self.shuffleBatchEveryEpoch = shuffle
        self.logPeriodEpoch = log_every_n_epochs
        # imp
        self.epochLosses = self.init_epoch_metric()
        self.epochMetrics = self.init_epoch_metric()
        self.plot = Plot()

    @staticmethod
    def init_epoch_metric():
        return {'train': {'_batch': [], 'epoch': []}, 'test': {'_batch': [], 'epoch': []}}

    def reset_batch_metrics(self, dataset_name='train'):
        self.epochLosses[dataset_name]['_batch'] = []
        self.epochMetrics[dataset_name]['_batch'] = []

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

    def train(self, train_set: DataProxy, test_set: DataProxy = None, n_epochs=1000, seed=42):
        """
        - must call DataProxy(data, labels) or DataProxy(dataset: tc.Dataset) to create datasets first
        - have split train/test sets for easy tracking learning performance side-by-side
        - both datasets must contain data and labels
        """
        start_time = perf_timer()
        tc.manual_seed(seed)
        # Turn datasets into iterables (batches)
        train_dl = tud.DataLoader(train_set, batch_size=self.batchSize, shuffle=self.shuffleBatchEveryEpoch)
        test_dl = None
        if test_set:
            # no need to shuffle test data
            test_dl = tud.DataLoader(test_set, batch_size=self.batchSize, shuffle=False)
        # reset
        self.epochLosses = self.init_epoch_metric()
        self.epochMetrics = self.init_epoch_metric()
        verbose = self.logPeriodEpoch > 0
        for epoch in tqdm(range(n_epochs), desc='Training'):
            # Training
            # - train mode is on by default after construction
            self.reset_batch_metrics('train')
            for batch, (X_train, y_train) in enumerate(train_dl):
                X_train, y_train = DataProxy.use_device(X_train, y_train, self.device)
                self.model.train()
                train_pred, train_loss = self.forward_pass(X_train, y_train, 'train')
                # - reset grad before backpropagation
                self.optimizer.zero_grad()
                # - backpropagation
                train_loss.backward()
                # - update weights and biases
                self.optimizer.step()
            self.compute_epoch_loss(train_dl, 'train')
            self.evaluate_epoch(train_dl, 'train')
            # testing using a validation set
            if test_set:
                self.model.eval()
                with tc.inference_mode():
                    self.reset_batch_metrics('test')
                    for X_test, y_test in test_dl:
                        X_test, y_test = DataProxy.use_device(X_test, y_test, self.device)
                        test_pred, test_loss = self.forward_pass(X_test, y_test, 'test')
                    self.compute_epoch_loss(test_dl, 'test')
                    self.evaluate_epoch(test_dl, 'test')
            if verbose:
                self.log_epoch(epoch)
        stop_time = perf_timer()
        # final test predictions
        self.evaluate_training(start_time, stop_time)
        if verbose:
            self.plot_learning()
        return test_pred

    def predict(self, data_set):
        """
        - data_set must have no labels
        """
        self.model.to(self.device)
        dl = tud.DataLoader(data_set, batch_size=self.batchSize, shuffle=False)
        self.model.eval()
        with tc.inference_mode():
            for X, y_true in tqdm(dl, desc='Predicting'):
                X, y_true = DataProxy.use_device(X, y_true, self.device)
                y_pred = self.model(X)
        data_set.targets = y_pred.to(self.device)
        return data_set.targets

    def evaluate_model(self, test_set):
        """
        - test_set must have labels
        """
        assert len(test_set.targets) > 0, 'Test-set must contain ground truth'
        dl = tud.DataLoader(test_set, batch_size=self.batchSize, shuffle=False)
        # Testing
        # - eval mode is on by default after construction
        mean_loss = 0
        self.model.eval()
        # - forward pass
        with tc.inference_mode():
            for b, (X, y_true) in enumerate(dl):
                X, y_true = DataProxy.use_device(X, y_true, self.device)
                y_pred = self.model(X)
                mean_loss += self.lossFunction(y_pred, y_true).item()
            mean_loss /= len(dl)
        return {
            'model': type(self.model).__name__,
            'loss': mean_loss,
        }

    def plot_learning(self):
        """
        - prediction quality
        - learning curves
        """
        self.plot.unblock()
        self.plot.plot_learning(self.epochLosses['train']['epoch'], self.epochLosses['test']['epoch'])

    def forward_pass(self, X, y_true, dataset_name='train'):
        y_pred = self.model(X)
        loss = self.lossFunction(y_pred, y_true)
        # instrumentation
        self.collect_batch_loss(loss, dataset_name)
        self.evaluate_batch(y_pred, y_true, dataset_name)
        return y_pred, loss

    def collect_batch_loss(self, loss, dataset_name='train'):
        self.epochLosses[dataset_name]['_batch'].append(loss.cpu().detach().numpy())

    def compute_epoch_loss(self, dataloader, dataset_name='train'):
        self.epochLosses[dataset_name]['epoch'].append(loss_epoch := (sum(self.epochLosses[dataset_name]['_batch']) / len(dataloader)).item())

    def evaluate_epoch(self, dataloader, dataset_name='train'):
        self.epochMetrics[dataset_name]['epoch'].append(measure_epoch := sum(self.epochMetrics[dataset_name]['_batch']) / len(dataloader))

    def evaluate_batch(self, y_pred, y_true, dataset_name='train'):
        """
        - for classification only, this method should return accuracy, precision, recall
        """
        pass

    def evaluate_training(self, start_time, stop_time):
        """
        - training time
        - loss and metric
        """
        self.logger.info(f'Training time on device {self.device}: {stop_time - start_time:.3f}s')

    def get_performance(self):
        return {'train': self.epochLosses['train']['epoch'][-1], 'test': self.epochLosses['test']['epoch'][-1]}

    def log_epoch(self, epoch):
        if epoch % self.logPeriodEpoch != 0:
            return
        train_loss_percent = 100 * self.epochLosses['train']['epoch'][epoch]
        msg = f"Epoch: {epoch} | Train Loss: {train_loss_percent:.4f}%"
        if self.epochLosses['test']['epoch']:
            test_loss_percent = 100 * self.epochLosses['test']['epoch'][epoch]
            msg += f" | Test Loss: {test_loss_percent:.4f}%"
        self.logger.info(msg)

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
        return osp.join(util.get_platform_appdata_dir(), 'torch', f'{model_basename}{ext}')


class BinaryClassifier(Regressor):
    def __init__(self, model, loss_fn: typing.Union[str, Regressor.LossFuncType] = 'BCE', optm='SGD', learning_rate=0.01, batch_size=32, shuffle=True, device_name=None, logger=None, log_every_n_epochs=0):
        super().__init__(model, loss_fn, optm, learning_rate, batch_size, shuffle, device_name, logger, log_every_n_epochs)
        # TODO: parameterize metric type
        self.metrics = {'train': tm.classification.Accuracy(task='binary').to(self.device), 'test': tm.classification.Accuracy(task='binary').to(self.device)}
        self.performance = {'train': None, 'test': None}

    def predict(self, data_set):
        """
        - data_set must have no labels and must be filled by this method
        - we don't evaluate model here
        """
        # assert tc.all(data_set.targets==-1), f'Expect dataset to contain no ground truth (all NaN), but got: {data_set.targets}'
        self.model.to(self.device)
        dl = tud.DataLoader(data_set, batch_size=self.batchSize, shuffle=False)
        y_pred_set = []
        self.model.eval()
        with tc.inference_mode():
            for X, y_true in tqdm(dl, desc='Predicting'):
                X, y_true = DataProxy.use_device(X, y_true, self.device)
                y_logits = self.model(X).squeeze()
                y_pred = self._logits_to_labels(y_logits)
                y_pred_set.append(y_pred)
        data_set.targets = tc.cat(y_pred_set, dim=0).to(self.device)
        return data_set.targets

    def evaluate_model(self, test_set):
        """
        - test_set must have labels
        """
        assert len(test_set.targets) > 0, 'Test-set must contain ground truth'
        dl = tud.DataLoader(test_set, batch_size=self.batchSize, shuffle=False)
        # Testing
        # - eval mode is on by default after construction
        n_classes = tc.unique(test_set.targets).shape[0]
        mean_loss, mean_acc = 0, 0
        task = 'binary' if n_classes == 2 else 'multiclass'
        metric = tm.classification.Accuracy(task=task).to(self.device) if n_classes < 3 else tm.classification.Accuracy(task=task, num_classes=n_classes).to(self.device)
        self.model.eval()
        # - forward pass
        with tc.inference_mode():
            for b, (X, y_true) in enumerate(dl):
                X, y_true = DataProxy.use_device(X, y_true, self.device)
                y_logits = self.model(X).squeeze()
                y_pred = self._logits_to_labels(y_logits)
                mean_loss += self.lossFunction(self._logits_to_probabilities(y_logits), y_true)
                mean_acc += metric(y_pred, y_true).item()
            mean_loss /= len(dl)
            mean_acc /= len(dl)
        return {
            'model': type(self.model).__name__,
            'loss': mean_loss,
            'accuracy': mean_acc,
        }

    def forward_pass(self, X, y_true, dataset_name='train'):
        """
        - BCEWithLogitsLoss is not supported
          - we don't support BCEWithLogitsLoss for consistency
          - so that all loss functions can adopt an explicit activation function
          - and BCEWithLogitsLoss requires no explicit activation because it builds in sigmoid
        """
        # squeeze to remove extra `1` dimensions, this won't work unless model and data are on the same device
        y_logits = self.model(X).squeeze()
        # turn logits -> pred probs -> pred labels
        y_pred = self._logits_to_labels(y_logits)
        loss = self.lossFunction(self._logits_to_probabilities(y_logits), y_true)
        # instrumentation
        self.collect_batch_loss(loss, dataset_name)
        self.evaluate_batch(y_pred, y_true, dataset_name)
        return y_pred, loss

    @staticmethod
    def _logits_to_labels(y_logits):
        """
        - logits -> pred probs -> pred labels
        - raw model output must be activated to get probabilities then labels
        - special activators, e.g., softmax, must override this method
        """
        return tc.round(BinaryClassifier._logits_to_probabilities(y_logits))

    @staticmethod
    def _logits_to_probabilities(y_logits):
        return tc.sigmoid(y_logits)

    def evaluate_batch(self, y_pred, y_true, dataset_name='train'):
        """
        - for classification only, this method should return accuracy, precision, recall
        """
        meas = self.metrics[dataset_name](y_pred, y_true)
        self.epochMetrics[dataset_name]['_batch'].append(meas)

    def log_epoch(self, epoch):
        if epoch % self.logPeriodEpoch != 0:
            return
        train_loss_percent = 100 * self.epochLosses['train']['epoch'][epoch]
        train_acc_percent = 100 * self.epochMetrics['train']['epoch'][epoch]
        msg = f"""Epoch: {epoch}
Train Loss: {train_loss_percent:.4f}% | Train Accuracy: {train_acc_percent:.4f}%
"""
        if self.epochLosses['test']['epoch']:
            test_loss_percent = 100 * self.epochLosses['test']['epoch'][epoch]
            test_acc_percent = 100 * self.epochMetrics['test']['epoch'][epoch]
            msg += f" Test Loss: {test_loss_percent:.4f}% |  Test Accuracy: {test_acc_percent:.4f}%"
        self.logger.info(msg)

    def evaluate_epoch(self, dataloader, dataset_name='train'):
        self.epochMetrics[dataset_name]['epoch'].append(measure_epoch := (sum(self.epochMetrics[dataset_name]['_batch']) / len(dataloader)).item())

    def evaluate_training(self, start_time, stop_time):
        super().evaluate_training(start_time, stop_time)
        for dataset_name in ['train', 'test']:
            self.performance[dataset_name] = sum(self.epochMetrics[dataset_name]['epoch']) / len(self.epochMetrics[dataset_name]['epoch'])
            self.logger.info(f'{dataset_name.capitalize()} Performance ({type(self.metrics[dataset_name]).__name__}): {100 * self.performance[dataset_name]:.4f}%')
            self.metrics[dataset_name].reset()

    def get_performance(self):
        return self.performance

    def plot_2d_predictions(self, train_set, test_set, predictions=None):
        """
        - assume 2D dataset (ds.data is [dim1, dim2]), plot decision boundaries
        - create special dataset and run model on it for visualization (2D)
        - ref: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py
        """

        def _predict_dataset(dataset):
            # Put everything to CPU (works better with NumPy + Matplotlib)
            self.model.to("cpu")
            X, y = dataset.data.to("cpu"), dataset.targets.to("cpu")
            # Setup prediction boundaries and grid
            x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
            y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
            n_data = 100
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_data + 1), np.linspace(y_min, y_max, n_data + 1))
            # Interpolate to create new data for plotting
            X_plottable = tc.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()
            # Make predictions
            # - loss function requires that label be of the same size as data
            # - instead of using squeeze/unsqueeze, we initialize with dummy labels
            plot_set = DataProxy(X_plottable, tc.full((len(X_plottable),), float('nan')), target_dtype=tc.long, device='cpu')
            y_pred = self.predict(plot_set)
            y_pred = y_pred.to("cpu")
            # reset model device, dataset device has not changed
            self.model.to(self.device)
            return y_pred.reshape(xx.shape).detach().numpy()

        if train_set:
            train_pred = _predict_dataset(train_set)
            self.plot.plot_decision_boundary(train_set, train_pred)
        if test_set:
            test_pred = _predict_dataset(test_set)
            self.plot.plot_decision_boundary(test_set, test_pred)


class MultiClassifier(BinaryClassifier):
    def __init__(self, model, loss_fn: typing.Union[str, Regressor.LossFuncType] = 'CrossEntropy', optm='SGD', learning_rate=0.01, batch_size=32, shuffle=True, device_name=None, logger=None, log_every_n_epochs=0):
        super().__init__(model, loss_fn, optm, learning_rate, batch_size, shuffle, device_name, logger, log_every_n_epochs)
        self.labelCountIsKnown = False
        # we don't know label count until we see the first batch
        self.metrics = {'train': None, 'test': None}

    # def predict(self, data_set, for_plot_only=False):
    #     """
    #     - data_set must have no labels and must be filled by this method
    #     - we don't evaluate model here
    #     """
    #     # assert tc.all(data_set.targets==-1), f'Expect dataset to contain no ground truth (all NaN), but got: {data_set.targets}'
    #     dev = 'cpu' if for_plot_only else self.device
    #     dl = tud.DataLoader(data_set, batch_size=self.batchSize, shuffle=False)
    #     # Testing
    #     # - eval mode is on by default after construction
    #     y_pred_set = []
    #     self.model.eval()
    #     # - forward pass
    #     with tc.inference_mode():
    #         for X, y_true in tqdm(dl, desc='Predicting'):
    #             X, y_true = DataProxy.use_device(X, y_true, dev)
    #             y_logits = self.model(X).squeeze()
    #             y_pred = self._logits_to_labels(y_logits)
    #             y_pred_set.append(y_pred)
    #     data_set.targets = tc.cat(y_pred_set, dim=0).to(dev)
    #     return data_set.targets

    def forward_pass(self, X, y_true, dataset_name='train'):
        y_logits = self.model(X)
        if not self.labelCountIsKnown:
            self.metrics = {'train': tm.classification.Accuracy(task='multiclass', num_classes=y_logits.shape[1]).to(self.device), 'test': tm.classification.Accuracy(task='multiclass', num_classes=y_logits.shape[1]).to(self.device)}
            self.labelCountIsKnown = True
        y_pred = self._logits_to_labels(y_logits)
        loss = self.lossFunction(self._logits_to_probabilities(y_logits), y_true)
        # instrumentation
        self.collect_batch_loss(loss, dataset_name)
        self.evaluate_batch(y_pred, y_true, dataset_name)
        return y_pred, loss

    @staticmethod
    def _logits_to_probabilities(y_logits):
        """
        - softmax is not necessarily needed
        - observation: using probability for loss will often need smaller batches and more epochs than using logits directly, e.g., 100 vs. 1000
        - but using probability is theoretically more accurate
        - ref: https://github.com/mrdbourke/pytorch-deep-learning/discussions/314
        """
        dim_cls = 1
        return tc.softmax(y_logits, dim=dim_cls)

    @staticmethod
    def _logits_to_labels(y_logits):
        """
        - dim 0: along samples
        - dim 1: along classes
        - for each logits sample below, we must first softmax all logits across the classes dimension, then pick the class with the highest probability
        - so the processing is always along the classes dimension (dim 1)
          tensor([[-0.5566, -0.6590,  1.0053, -0.1095],
                [-0.7012, -0.8162,  1.3412, -0.1254],
                [-0.4109,  1.4493,  0.6572,  1.5839],
                ...,
                [-0.3833,  1.5534,  0.5927,  1.6507],
                [ 0.0991,  1.5113, -0.5255,  1.2166],
                [ 0.2739,  1.7573, -0.9321,  1.2839]], device='mps:0',
        """
        dim_cls = 1
        return tc.softmax(y_logits, dim=dim_cls).argmax(dim=dim_cls)


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
            ax.scatter(train_set.data.cpu(), train_set.targets.cpu(), s=4, color='blue', label='Training Data')
        if test_set:
            ax.scatter(test_set.data.cpu(), test_set.targets.cpu(), s=4, color='green', label='Testing Data')
        if predictions is not None:
            ax.scatter(test_set.data.cpu(), predictions.cpu(), s=4, color='red', label='Predictions')
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
        x_min, x_max = dataset2d.data[:, 0].min() - epsilon, dataset2d.data[:, 0].max() + epsilon
        y_min, y_max = dataset2d.data[:, 1].min() - epsilon, dataset2d.data[:, 1].max() + epsilon
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
        fig, ax = plt.subplots(figsize=(10, 7))
        # draw colour-coded predictions on meshgrid
        ax.contourf(xx, yy, predictions, cmap=plt.cm.RdYlBu, alpha=0.7)
        ax.scatter(dataset2d.data[:, 0], dataset2d.data[:, 1], c=dataset2d.targets, s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show(block=self.useBlocking)

    def plot_image_grid(self, img_set, n_rows=4, n_cols=4, fig_size=(9, 9), pick_random=True, color_map='gray', seed=None):
        if seed:
            tc.manual_seed(seed)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        for i, ax in enumerate(axes.flat):
            p = np.random.randint(0, len(img_set)) if pick_random else i
            img, label = img_set[p]
            ax.imshow(img.squeeze(), cmap=color_map)
            ax.set_title(img_set.classes[label])
            ax.axis('off')
        plt.show(block=self.useBlocking)

    def plot_image_predictions(self, img_set, predictions, n_rows=4, n_cols=4, fig_size=(9, 9), color_map='gray', pick_random=True, seed=None):
        """
        - img_set must be a torchvision dataset, not dataproxy
        """
        if seed:
            tc.manual_seed(seed)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        for i, ax in enumerate(axes.flat):
            p = np.random.randint(0, len(img_set)) if pick_random else i
            img, label = img_set[p]
            pred = predictions[p]
            ax.imshow(img.squeeze(), cmap=color_map)
            ax.set_title(f'Pred: {img_set.classes[pred]} -> Truth: {img_set.classes[label]}', c='g' if pred == label else 'r')
            ax.axis('off')
        plt.show(block=self.useBlocking)

    def plot_confusion_matrix(self, y_pred, y_true, class_names, title='Confusion matrix'):
        """
        - ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix
        """
        from torchmetrics import ConfusionMatrix
        from mlxtend.plotting import plot_confusion_matrix
        y_pred, y_true = y_pred.cpu(), y_true.cpu()
        # 2. Setup confusion matrix instance and compare predictions to targets
        confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
        confmat_tensor = confmat(preds=y_pred,
                                 target=y_true)

        # 3. Plot the confusion matrix
        fig, ax = plot_confusion_matrix(
            conf_mat=confmat_tensor.numpy(),  # matplotlib likes working with NumPy
            class_names=class_names,  # turn the row and column labels into class names
            figsize=(10, 7)
        )
        ax.set_title(title)
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
    def export_svg(path=osp.join(util.get_platform_home_dir(), 'Desktop', 'plot.png')):
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
