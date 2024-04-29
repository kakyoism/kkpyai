import copy
import os.path as osp
import sys
import time

# 3rd party
import kkpyutil as util
import torch as tc
from torch import nn
import torchinfo as ti
import torchvision as tcv

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
    train_set, test_set = ktc.NumericDataset(X, y).split_train_test(train_ratio=0.8)
    with tc.inference_mode():
        y_preds = model(test_set.data)
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


def test_regressor():
    # Create a Linear Regression model class
    class LinearRegressionModel(tc.nn.Module):  # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
        def __init__(self):
            super().__init__()
            self.linear_layer = nn.Linear(in_features=1, out_features=1)

        # Forward defines the computation in the model
        def forward(self, x: tc.Tensor) -> tc.Tensor:  # <- "x" is the input data (e.g., training/testing features)
            return self.linear_layer(x)

    model = LinearRegressionModel()
    weight, bias = 0.7, 0.3
    start, end, step = 0, 1, 0.02
    X = tc.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias
    train_set, test_set = ktc.NumericDataset(X, y).split_train_test(train_ratio=0.8)
    regressor = ktc.Regressor(model, loss_fn='MSE', optm='SGD', learning_rate=0.01, log_every_n_epochs=100)
    regressor.train(train_set, test_set, n_epochs=2000)
    pred_set = copy.deepcopy(test_set)
    y_preds = regressor.predict(pred_set)
    assert tc.allclose(y_preds, pred_set.targets, atol=0.1)
    regressor.save('test_model')
    assert osp.isfile(mdl := osp.join(util.get_platform_appdata_dir(), 'torch', 'test_model.pth'))
    regressor.load('test_model')
    util.safe_remove(mdl)
    regressor.close_plot()
    assert regressor.get_performance()['test'] < 0.2
    perf = regressor.evaluate_model(test_set)
    assert perf['loss'] < 0.2


def test_binary_classifier():
    from sklearn.datasets import make_circles
    # Make 1000 samples
    n_samples = 2000
    # to save time, we use the biggest batch size possible
    classifier = ktc.BinaryClassifier(tc.nn.Sequential(
        tc.nn.Linear(in_features=2, out_features=100),
        tc.nn.ReLU(),
        tc.nn.Linear(in_features=100, out_features=100),
        tc.nn.ReLU(),
        tc.nn.Linear(in_features=100, out_features=1),
    ), learning_rate=0.1, batch_size=n_samples, log_every_n_epochs=100)
    # Create circles
    X, y = make_circles(n_samples,
                        noise=0.03,  # a little bit of noise to the dots
                        random_state=42)  # keep a random state so we get the same values
    X = tc.from_numpy(X).type(tc.float)
    # convert bin-class labels to multi-class label shapes for consistency
    y = tc.from_numpy(y.reshape(len(y), 1)).type(tc.float)
    train_set, test_set = ktc.NumericDataset(X, y).split_train_test(train_ratio=0.8)
    classifier.train(train_set, test_set, n_epochs=400)
    classifier.plot_2d_predictions(train_set, test_set)
    classifier.close_plot()
    pred_set = copy.deepcopy(test_set)
    preds = classifier.predict(pred_set)
    assert classifier.performance['test'] > 0.8
    assert classifier.evaluate_model(test_set)['accuracy'] > 0.8


def test_multiclass_classifier():
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

    n_samples = 1000
    # 1. Create multi-class data
    X, y = make_blobs(n_samples=n_samples,
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
    classifier = ktc.MultiClassifier(model,
                                     learning_rate=0.1,
                                     batch_size=32,
                                     log_every_n_epochs=100)
    train_set, test_set = ktc.NumericDataset(X, y, target_dtype=tc.long).split_train_test(train_ratio=0.8)
    classifier.train(train_set, test_set, n_epochs=800)
    classifier.plot_2d_predictions(train_set, test_set)
    classifier.close_plot()
    preds = classifier.predict(test_set)
    assert classifier.performance['test'] > 0.9


def test_image_classifier():
    # Create a convolutional neural network
    BATCH_SIZE = 32

    class FashionMNISTModelV2(nn.Module):
        """
        Model architecture copying TinyVGG from:
        https://poloclub.github.io/cnn-explainer/
        """

        def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
            super().__init__()
            self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape,
                          out_channels=hidden_units,
                          kernel_size=3,  # how big is the square that's going over the image?
                          stride=1,  # default
                          padding=1),  # options = "valid" (no padding) or "same" (output has the same shape as input) or int for specific number
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                             stride=2)  # default stride value is same as kernel_size
            )
            self.block_2 = nn.Sequential(
                nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                # Where did this in_features shape come from?
                # It's because each layer of our network compresses and changes the shape of our inputs data.
                nn.Linear(in_features=hidden_units * 7 * 7,
                          out_features=output_shape)
            )

        def forward(self, x: tc.Tensor):
            return self.classifier(self.block_2(self.block_1(x)))

    train_data = ktc.retrieve_vision_trainset(data_cls=tcv.datasets.FashionMNIST)
    test_data = ktc.retrieve_vision_testset(data_cls=tcv.datasets.FashionMNIST)
    ktc.inspect_dataset(train_data, block=False)
    model = FashionMNISTModelV2(input_shape=1,
                                hidden_units=10,
                                output_shape=len(train_data.classes))
    ti.summary(model, input_size=(BATCH_SIZE, *train_data[0][0].shape))
    classifier = ktc.MultiClassifier(model,
                                     learning_rate=0.1,
                                     batch_size=BATCH_SIZE,
                                     log_every_n_epochs=100)

    train_set = ktc.StdImageDataset(train_data)
    test_set = ktc.StdImageDataset(test_data)
    classifier.train(train_set, test_set, n_epochs=3)
    perf = classifier.evaluate_model(test_set)
    plot = ktc.Plot()
    plot.unblock()
    preds = classifier.predict(test_set)
    plot.plot_image_predictions(test_data, preds)
    plot.plot_confusion_matrix(preds, test_set.targets, class_names=test_data.classes)
    assert perf['accuracy'] > 0.6


def test_transfer_learning():
    """
    - create dataset using local data
    - download a pre-trained model
    - train the model on the local data
    - predict a new image
    - experiment to refine the model using tensorboard
    """
    def _create_model_output_layer(n_out_features):
        return tc.nn.Sequential(
            # avoid overfitting
            # - Dropout layers randomly remove connections between two neural network layers with a probability of p
            # - For example, if p=0.2, 20% of connections between neural network layers will be removed at random each pass
            # - This practice is meant to help regularize (prevent overfitting) a model by making sure the connections that remain learn features to compensate for the removal of the other connections (hopefully these remaining features are more general).
            tc.nn.Dropout(p=0.2, inplace=True),
            tc.nn.Linear(in_features=1280, out_features=n_out_features)
        )
    # lazy retrieve data
    pizza_steak_sushi_data = util.lazy_download(osp.join(_gen_dir, 'pizza_steak_sushi.zip'), 'https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')
    time.sleep(1)
    util.unzip_dir(pizza_steak_sushi_data, root := osp.abspath(f'{_gen_dir}/data/pizza_steak_sushi'))
    # create datasets:
    # - make data conform to pre-trained model requirements
    # - it usually comes with the model itself
    # - manual example:
    # data_transform = tcv.transforms.Compose([
    #     tcv.transforms.Resize((224, 224)),
    #     tcv.transforms.ToTensor(),
    #     tcv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])
    # ])
    # initialize model
    weights = tcv.models.EfficientNet_B0_Weights.DEFAULT
    model = tcv.models.efficientnet_b0(weights=weights)
    ti.summary(model, input_size=(32, 3, 224, 224),
               col_names=("input_size", "output_size", "num_params", "trainable"),
               col_width=20,
               row_settings=['var_names'])
    data_transform = weights.transforms()
    train_data, test_data = ktc.ImageFolderDatasetFactory(root, data_transform).create()
    plt = ktc.Plot()
    plt.unblock()
    plt.plot_image_grid(train_data)
    plt.close()
    # create classifier
    # after researching into the model with torchinfo, we update its output layer
    model.classifier = tc.nn.Sequential(
        # avoid overfitting
        # - Dropout layers randomly remove connections between two neural network layers with a probability of p
        # - For example, if p=0.2, 20% of connections between neural network layers will be removed at random each pass
        # - This practice is meant to help regularize (prevent overfitting) a model by making sure the connections that remain learn features to compensate for the removal of the other connections (hopefully these remaining features are more general).
        tc.nn.Dropout(p=0.2, inplace=True),
        tc.nn.Linear(in_features=1280, out_features=len(train_data.classes))
    )
    classifier = ktc.MultiClassifier(model, optimizer='Adam', learning_rate=0.01, batch_size=32, log_every_n_epochs=100, transfer=True)
    classifier.transfer_learn(_create_model_output_layer, len(train_data.classes))
    classifier.train(train_data, test_data, n_epochs=5)
    test_img = util.lazy_download(osp.join(_gen_dir, '04-pizza-dad.jpeg'), url='https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg')
    pred_set = ktc.ImagePredictionDataset(test_img, data_transform)
    y_pred = classifier.predict(pred_set)
    assert train_data.classes[y_pred.argmax()] == 'pizza'
