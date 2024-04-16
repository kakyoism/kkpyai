import os.path as osp
import sys
# 3rd party
import kkpyutil as util
import torch as tc
from torch import nn

# project
_script_dir = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, repo_root := osp.abspath(f'{_script_dir}/../../kkpyai'))
import kktorch as ktc

_case_dir = _script_dir
_src_dir = osp.abspath(osp.dirname(_case_dir))
_org_dir = osp.join(_case_dir, '_org')
_gen_dir = osp.join(_case_dir, '_gen')
_ref_dir = osp.join(_case_dir, '_ref')
_skip_slow_tests = osp.isfile(osp.join(_case_dir, 'skip_slow_tests.cfg.txt'))
_skip_reason = 'tests requires long network or file i/o are temporarily skipped during tdd'


def test_factory_ramp():
    fact = ktc.TensorFactory()
    got = fact.ramp((1, 3, 3), 1)
    assert tc.allclose(got, tc.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device=got.device))


def test_factory_rand_repro():
    fact = ktc.TensorFactory()
    got1 = fact.rand_repro((2, 2), 42)
    got2 = fact.rand_repro((2, 2), 42)
    assert tc.allclose(got1, got2)


def test_plot_predictions():
    # Create a Linear Regression model class
    class LinearRegressionModel(tc.nn.Module):  # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
        def __init__(self):
            super().__init__()
            self.linear_layer = nn.Linear(in_features=1, out_features=1)

        # Forward defines the computation in the model
        def forward(self, x: tc.Tensor) -> tc.Tensor:  # <- "x" is the input data (e.g. training/testing features)
            return self.linear_layer(x)

    model = LinearRegressionModel()
    weight, bias = 0.7, 0.3
    start, end, step = 0, 1, 0.02
    X = tc.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias
    train_set, test_set = ktc.split_dataset(X, y, train_ratio=0.8)
    with tc.inference_mode():
        y_preds = model(test_set['data'])
    plot = ktc.Plot()
    plot.unblock()
    plot.plot_predictions(train_set, test_set, y_preds)
    assert True
    plot.export_png(pic := osp.join(_gen_dir, 'plot_predictions.png'))
    assert osp.isfile(pic)
    util.safe_remove(pic)
    plot.export_svg(pic := osp.join(_gen_dir, 'plot_predictions.svg'))
    assert osp.isfile(pic)
    plot.close()
    util.safe_remove(pic)


def test_regressor_model():
    # Create a Linear Regression model class
    class LinearRegressionModel(tc.nn.Module):  # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
        def __init__(self):
            super().__init__()
            self.linear_layer = nn.Linear(in_features=1, out_features=1)

        # Forward defines the computation in the model
        def forward(self, x: tc.Tensor) -> tc.Tensor:  # <- "x" is the input data (e.g. training/testing features)
            return self.linear_layer(x)

    model = LinearRegressionModel()
    weight, bias = 0.7, 0.3
    start, end, step = 0, 1, 0.02
    X = tc.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias
    train_set, test_set = ktc.split_dataset(X, y, train_ratio=0.8)
    regressor = ktc.Regressor(model, loss_fn='MSE', optm='SGD', learning_rate=0.01, log_every_n_epochs=100)
    regressor.train(train_set, test_set, n_epochs=2000)
    y_preds = regressor.predict(test_set)
    assert tc.allclose(y_preds, test_set['labels'], atol=0.1)
    regressor.save('test_model')
    assert osp.isfile(mdl := osp.join(util.get_platform_tmp_dir(), 'torch', 'test_model.pth'))
    regressor.load('test_model')
    util.safe_remove(mdl)
    regressor.close_plot()
    assert regressor.get_performance()['test'] < 0.2


def test_binaryclassifier_model():
    from sklearn.datasets import make_circles
    classifier = ktc.BinaryClassifier(tc.nn.Sequential(
        tc.nn.Linear(in_features=2, out_features=100),
        tc.nn.ReLU(),
        tc.nn.Linear(in_features=100, out_features=100),
        tc.nn.ReLU(),
        tc.nn.Linear(in_features=100, out_features=1),
    ), learning_rate=0.1, log_every_n_epochs=100)
    # Make 1000 samples
    n_samples = 2000
    # Create circles
    X, y = make_circles(n_samples,
                        noise=0.03,  # a little bit of noise to the dots
                        random_state=42)  # keep random state so we get the same values
    X = tc.from_numpy(X).type(tc.float)
    y = tc.from_numpy(y).type(tc.float)
    train_set, test_set = ktc.split_dataset(X, y, train_ratio=0.8)
    classifier.train(train_set, test_set, n_epochs=1000)
    classifier.plot_predictions(train_set, test_set)
    classifier.close_plot()
    assert classifier.performance['test'].item() > 0.9


def test_multiclassifier_model():
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split

    # Set the hyperparameters for data creation
    NUM_CLASSES = 4
    NUM_FEATURES = 2
    RANDOM_SEED = 42

    class BlobModel(nn.Module):
        def __init__(self, input_features, output_features, hidden_units=8):
            """Initializes all required hyperparameters for a multi-class classification model.

            Args:
                input_features (int): Number of input features to the model.
                out_features (int): Number of output features of the model
                  (how many classes there are).
                hidden_units (int): Number of hidden units between layers, default 8.
            """
            super().__init__()
            self.linear_layer_stack = nn.Sequential(
                nn.Linear(in_features=input_features, out_features=hidden_units),
                # nn.ReLU(), # <- does our dataset require non-linear layers? (try uncommenting and see if the results change)
                nn.Linear(in_features=hidden_units, out_features=hidden_units),
                # nn.ReLU(), # <- does our dataset require non-linear layers? (try uncommenting and see if the results change)
                nn.Linear(in_features=hidden_units, out_features=output_features),  # how many classes are there?
            )

        def forward(self, x):
            return self.linear_layer_stack(x)

    # 1. Create multi-class data
    X, y = make_blobs(n_samples=1000,
                      n_features=NUM_FEATURES,  # X features
                      centers=NUM_CLASSES,  # y labels
                      cluster_std=1.5,  # give the clusters a little shake up (try changing this to 1.0, the default)
                      random_state=RANDOM_SEED
                      )
    # 2. Turn data into tensors
    X = tc.from_numpy(X).type(tc.float)
    y = tc.from_numpy(y).type(tc.LongTensor)
    model = BlobModel(input_features=NUM_FEATURES,
                      output_features=NUM_CLASSES,
                      hidden_units=8)
    classifier = ktc.MultiClassifier(model, learning_rate=0.1, log_every_n_epochs=100)
    train_set, test_set = ktc.split_dataset(X, y, train_ratio=0.8)
    classifier.train(train_set, test_set, n_epochs=1000)
    classifier.plot_predictions(train_set, test_set)
    classifier.close_plot()
    assert classifier.performance['test'].item() > 0.9
