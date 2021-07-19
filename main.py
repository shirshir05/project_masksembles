import json
import os
from time import time
import numpy as np
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude

from my_masksembles import MyMasksembles2D, MyMasksembles1D

# TODO pip install git+http://github.com/nikitadurasov/masksembles
from masksembles.keras import Masksembles1D, Masksembles2D
from sklearn.metrics import confusion_matrix, precision_recall_curve
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
# TODO: pip install scikit-optimize
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_evaluations
from skopt.plots import plot_objective
from skopt.utils import use_named_args
from skopt.plots import plot_objective_2D

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# TODO: pip install tensorflow_datasets
import tensorflow_datasets as tfds
from keras.utils.np_utils import to_categorical
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# TODO: code documentation
# params
random_state = 42
best_accuracy = 0.0
MAX_SAMPLES_NUM = 320

# region dataset
# all dataset were taken from: https://www.tensorflow.org/datasets/catalog/overview
datasets_info = dict(cassava=[9430, 5], mnist=[70000, 10], imagewang=[14, 669, 20],
                     binary_alpha_digits=[1404, 36], cifar10=[60000, 10],
                     citrus_leaves=[759, 4], rock_paper_scissors=[2520, 3],
                     horses_or_humans=[1280, 2], dmlab=[65550, 6], food101=[75750, 101], cmaterdb=[5000, 10],
                     stl10=[5000, 10], tf_flowers=[2670, 5], cats_vs_dogs=[23262, 2], uc_merced=[2100, 21],
                     kmnist=[60000, 10], deep_weeds=[17509, 9], eurosat=[27000, 10],
                     beans=[1295, 3], svhn_cropped=[73, 257, 10]
                     )  # "dataset_name": [n_samples, NUM_CLASSES]

# TODO: dataset check -malaria =[27,558,2]  Dtd=[1,880,47] caltech101=[3,060, 102], caltech_birds2010=[3,000, 200], caltech_birds2011=[	5,994, 200]
# "oxford_flowers102": [8189, 102], # TODO: n_splits=10 cannot be greater than the number of members in each class. ,
#  "stanford_online_products": [59551, 12],  # TODO: not support  as_supervised=True
#   snli= colorectal_histology,  patch_camelyon,oxford_iiit_pet=
#  ,fashion_mnist = i_naturalist2017 = quickdraw_bitmap,bigearthnet,malaria TODO: X


# endregion

# region hyper-parameters to tune
learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform', name='learning_rate')
n_convolutions = Integer(low=1, high=5, name='n_convolutions')
n = Integer(low=4, high=8, name='n')  # TODO: Change and Check Tuning values

# hold all examnined hyper-parameters dimention i a list
dimensions = [learning_rate, n_convolutions, n]
default_parameters = [1e-5, 1, 4]


# endregion

# region Config model
def add_dropout(model, model_to_run, dim, n):
    """
    the method adds a dropout layer to the input model, according to the wanted model.
    @param model: model to add dropout layer to
    @param model_to_run:  string: "basic", "masksembles" or "pruned_masksembles"
    @param dim: integer, 1 for 1D (after flattening the network) or 2 for 2D.
    @param n: number of masks, parameter of masksembles
    """
    if model_to_run == "basic":
        model.add(layers.Dropout(0.2))
    elif model_to_run == "masksembles":
        if dim == 1:
            model.add(Masksembles1D(n, 2.0))  # 4
        else:  # dim==2
            model.add(Masksembles2D(n, 2.0))
    else:  # model_to_run is "pruned_masksembles"
        if dim == 1:
            model.add(prune_low_magnitude(MyMasksembles1D(n, 2.0)))
        else:  # dim==2
            model.add(prune_low_magnitude(MyMasksembles2D(n, 2.0)))


def add_convolutions(model, n_convolutions, n_filters):
    """
    the method adds convolution layers to the input model
    @param model: model to add dropout layer to
    @param n_convolutions: number of convolution layers to add
    @param n_filters: number of filters of the layers
    """
    # Add n_convolutions convolution layers
    for i in range(n_convolutions):
        model.add(layers.Conv2D(filters=n_filters, kernel_size=(1, 1), activation="elu"))
    pass


def create_model(learning_rate, n_convolutions, n, model_to_run):
    """
    The method builds the model according to the examined hyper-parameters.
    The basic model was taken from https://github.com/nikitadurasov/masksembles/blob/main/notebooks/MNIST_Masksembles.ipynb
    @param learning_rate: learning rate to optimize
    @param n_convolutions: number of convolution layers
    @param n: number of masks, parameter of masksembles
    @param model_to_run: string: "basic", "masksembles" or "pruned_masksembles"
    @return: built model after compilation
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))

    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="elu"))
    add_convolutions(model, n_convolutions, 32)
    add_dropout(model, model_to_run, 2, n)
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="elu"))
    add_convolutions(model, n_convolutions, 64)
    add_dropout(model, model_to_run, 2, n)
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    add_dropout(model, model_to_run, 1, n)
    model.add(layers.Dense(n_classes, activation="softmax"))
    model.summary()

    optimizer = Adam(learning_rate=learning_rate)
    # TODO: change metric to maximize
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# endregion

def divied_4(x_check, y_check):
    while x_check.shape[0] % 4 != 0:
        x_check = x_check[1:]
        y_check = y_check[1:]
    return x_check, y_check


# region fitness - optimization
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, n_convolutions, n):
    """
    This function aims to create and train a network with given hyper-parameters
    and then evaluate model performance with the validation dataset.
    It returns fitness value, negative classification accuracy on the dataset.
    It is negative because skpot performs minimization rather than maximization.
    Parameter optimization is based on:
    https://github.com/mardani72/Hyper-Parameter_optimization/blob/master/Hyper_Param_Facies_tf_final.ipynb
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    n_convolutions:  Number of convolutions layers.
    n:   n parameter of masksembles.
    """
    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('n_convolutions:', n_convolutions)
    print('n:', n)
    print()

    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    average_accuracy = 0
    y_label = np.argmax(Y_train_val, axis=1)
    for train_index, val_index in kf.split(X_train_val, y_label):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = Y_train_val[train_index], Y_train_val[val_index]

        # Create the neural network with these hyper-parameters.
        model = create_model(learning_rate=learning_rate,
                             n_convolutions=n_convolutions,
                             n=n,
                             model_to_run="pruned_masksembles")
        X_train, y_train = divied_4(X_train, y_train)
        X_val, y_val = divied_4(X_val, y_val)

        # TODO: change number epoch
        history = model.fit(x=X_train,
                            y=y_train,
                            epochs=100,
                            batch_size=32,
                            validation_data=(X_val, y_val),
                            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
                            )

        # TODO: change metric
        accuracy = history.history['val_accuracy'][-1]
        average_accuracy += accuracy
        # Delete the Keras model with these hyper-parameters from memory.
        del model
        K.clear_session()

    average_accuracy = average_accuracy / 3
    accuracy = average_accuracy
    print()
    print("Average Accuracy: {0:.2%}".format(accuracy))
    print()
    global best_accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
    return -accuracy


# endregion

def best_result(search_result, ds_name, index_cv=0):
    if not os.path.exists(os.path.join("BayesianOptimization", ds_name)):
        os.mkdir(os.path.join("BayesianOptimization", ds_name))
    if not os.path.exists(os.path.join("BayesianOptimization", ds_name, str(index_cv))):
        os.mkdir(os.path.join("BayesianOptimization", ds_name, str(index_cv)))
    plot_convergence(search_result)
    plt.savefig(os.path.join("BayesianOptimization", ds_name, str(index_cv), "Converge.png"))
    plt.clf()
    print(f" search result {search_result.x}")
    print(f"The best fitness value associated with these hyper-parameters {search_result.fun}")

    fig = plot_objective_2D(result=search_result,
                            dimension_identifier1='learning_rate',
                            dimension_identifier2='n',
                            levels=50)
    plt.savefig(os.path.join("BayesianOptimization", ds_name, str(index_cv), "Lr_numnods.png"))
    plt.clf()
    # create a list for plotting
    dim_names = ['learning_rate', 'n_convolutions', 'n']
    plot_objective(result=search_result, dimensions=dim_names)
    plt.savefig(os.path.join("BayesianOptimization", ds_name, str(index_cv), "all_dimen.png"))
    plt.clf()
    plot_evaluations(result=search_result, dimensions=dim_names)


def evaluate_on_test(y_true, y_pred, ds_name, index_cv, scores):
    def pr_auc_score(y_true, y_score):
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
        return metrics.auc(recall, precision)

    n_classes = [i for i in range(y_true.shape[1])]
    index_y_pred = np.argmax(y_pred, axis=1)
    max_y_pred = np.zeros((index_y_pred.size, len(n_classes)))
    max_y_pred[np.arange(index_y_pred.size), index_y_pred] = 1
    index_y_true = np.argmax(y_true, axis=1)

    scores['accuracy_score'] = metrics.accuracy_score(index_y_true, index_y_pred)
    cnf_matrix = confusion_matrix(index_y_true, index_y_pred)
    FP = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)).astype(float)
    FN = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)).astype(float)
    TP = (np.diag(cnf_matrix)).astype(float)
    TN = (cnf_matrix.sum() - (FP + FN + TP)).astype(float)
    scores['fpr'], scores['tpr'] = sum(FP / (FP + TN)) / len(n_classes), sum(TP / (TP + FN)) / len(n_classes)
    scores['precision_score'] = metrics.precision_score(y_true, max_y_pred, average='macro')
    scores['recall_score'] = metrics.recall_score(y_true, max_y_pred, average='macro')
    scores['auc_score'] = metrics.roc_auc_score(y_true, y_pred)

    precision = {}
    recall = {}
    pr = 0
    plt.clf()
    for i in n_classes:
        pr += pr_auc_score(y_true[:, i], y_pred[:, i])
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    scores['pr_auc_score'] = pr / len(n_classes)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.savefig(os.path.join("BayesianOptimization", ds_name, str(index_cv), "precision_recall.png"))
    plt.clf()
    return scores


def open_dirs():
    if not os.path.exists("BayesianOptimization"):
        os.mkdir("BayesianOptimization")
    # create in cluster - to prevent error
    for ds_name in datasets_info:
        print(f"uploading dataset: {ds_name}")
        for i in range(10):
            if not os.path.exists(os.path.join("BayesianOptimization", ds_name)):
                os.mkdir(os.path.join("BayesianOptimization", ds_name))
            if not os.path.exists(os.path.join("BayesianOptimization", ds_name, str(i))):
                os.mkdir(os.path.join("BayesianOptimization", ds_name, str(i)))


# ================================== start of the script ==================================

# create BayesianOptimization directories
# open_dirs()

all_score = {}
for ds_name in datasets_info:
    print(f"uploading dataset: {ds_name}")

    # constrain the size of train & test sets
    n_samples = MAX_SAMPLES_NUM if datasets_info[ds_name][0] > MAX_SAMPLES_NUM else datasets_info[ds_name][0]
    n_classes = datasets_info[ds_name][1]

    # load a sample of the dataset of size n_samples
    X, labels = tfds.as_numpy(tfds.load(ds_name,
                                        # split=f'train+test[:{n_samples}]',
                                        split=f'train[:{n_samples}]',
                                        batch_size=-1,
                                        as_supervised=True,
                                        shuffle_files=True, ))
    # transform to categorical one-hot vectors
    Y = to_categorical(labels, num_classes=n_classes)
    print(f"data shape before: {X.shape}")

    # preprocess
    X = X / 255
    X = np.resize(X, (X.shape[0], 32, 32, 1))
    print(f"data shape after: {X.shape}")
    input_shape = (X.shape[1], X.shape[2], X.shape[3])

    # perform nested cross validation for hyper-parameter optimization and generalization
    kf_external = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    index_cv = 0
    results = {}

    # # TODO: only for test (remove)
    # X = X[:200]
    # Y = Y[:200]
    # labels = labels[:200]

    for train_val_index, test_index in kf_external.split(X, labels):
        try:
            X_train_val, X_test = X[train_val_index], X[test_index]
            Y_train_val, Y_test = Y[train_val_index], Y[test_index]
            X_test, Y_test = divied_4(X_test, Y_test)
            best_accuracy = 0.0
            search_result = gp_minimize(func=fitness,
                                        dimensions=dimensions,
                                        acq_func='EI',  # Expected Improvement.
                                        n_calls=50,
                                        x0=default_parameters)

            results[tuple(search_result.x)] = best_accuracy

            opt_par = search_result.x
            learning_rate = opt_par[0]
            num_layers = opt_par[1]
            num_nodes = opt_par[2]
            best_model = create_model(learning_rate, num_layers, num_nodes)
            X_train_val, Y_train_val = divied_4(X_train_val, Y_train_val)
            start_train = time()
            history = best_model.fit(X_train_val, Y_train_val, epochs=100)
            end_train = time() - start_train
            y_pred = best_model.predict(X_test)
            score = {'accuracy_score': -1, "fpr": -1, 'tpr': -1, 'precision_score': -1, 'recall_score': -1,
                     'auc_score': -1, 'pr_auc_score': -1, 'Training_time': -1, 'inference_time': -1}
            try:
                score = evaluate_on_test(Y_test, y_pred, ds_name, index_cv, score)
            except:
                pass
            score['Training_time'] = end_train

            if len(test_index) > 1000:
                X_test = X_test[:1000]
            start_test = time()
            best_model.predict(X_test)
            end_test = time() - start_test
            score['inference_time'] = end_test
            all_score[f"{ds_name}:{index_cv}"] = [float(i) if not isinstance(i, str) else i for i in search_result.x] + \
                                                 [float(i) for i in list(score.values())]
            index_cv += 1

            best_result(search_result, ds_name, index_cv=index_cv)
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            print(f"Error {e}")
            pass
    with open(os.path.join("BayesianOptimization", "scores.json"), 'w') as f:  # for tracking
        json.dump(all_score, f)
    print(results)
with open(os.path.join("BayesianOptimization", "scores.json"), 'w') as f:
    json.dump(all_score, f)

# 1. Do 10 cross:
#     1.1 X_train_val, x_test, y_train_val, y_test
#     1.2 for i in setting hyper parameters:
#        1.2.1 Do 3 cross validation with X_train_val -> X_val (test), X_train(train)
#        1.2.2 The average of X_val is the result of the setting i
#     1.3 (search_result.x) create model with the best setting from step 1.2 and train with X_train_val and test
#     1.4 evaluate on x_test
# 2.select the best

# # TODO: check dataset
# opt_par = default_parameters
# learning_rate = opt_par[0]
# num_layers = opt_par[1]
# num_nodes = opt_par[2]
# activation = opt_par[3]
# best_model = create_model(learning_rate, num_layers, num_nodes, activation)
# X_train_val, Y_train_val = divied_4(X, Y)
# history = best_model.fit(X_train_val, Y_train_val, epochs=1)
# continue
