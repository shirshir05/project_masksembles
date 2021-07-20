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
from skopt.space import Real, Integer
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
MAX_SAMPLES_NUM = 1000

# region dataset
# all dataset were taken from: https://www.tensorflow.org/datasets/catalog/overview
datasets_info = dict(
    mnist=[70000, 10],
    beans=[1295, 3],
    mnist_corrupted=[60000, 10],
    plant_village=[54303, 38],
    binary_alpha_digits=[1404, 36],
    cifar10=[60000, 10],
    citrus_leaves=[594, 4],
    rock_paper_scissors=[2520, 3],
    horses_or_humans=[1280, 2],
    dmlab=[65550, 6],
    cmaterdb=[5000, 10],
    stl10=[5000, 10],
    tf_flowers=[2670, 5],
    cats_vs_dogs=[23262, 2],
    uc_merced=[2100, 21],
    kmnist=[60000, 10],
    food101=[75750, 101],
    deep_weeds=[17509, 9],
    eurosat=[27000, 10],
    svhn_cropped=[73257, 10],
)  # "dataset_name": [n_samples, NUM_CLASSES]

# TODO: dataset check -  caltech_birds2011=[5994, 200]
#  "stanford_online_products": [59551, 12],  # TODO: not support  as_supervised=True
#   snli= colorectal_histology,  patch_camelyon,oxford_iiit_pet, caltech101, caltech_birds2010, lfwת curated_breast_imaging_ddsm, pet_finder, plant_leaves
#  ,fashion_mnist = i_naturalist2017 = quickdraw_bitmap,bigearthnet,malaria, cassava,  imagewangת "oxford_flowers102" TODO: X


# endregion

# region hyper-parameters to tune
learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform', name='learning_rate')
n_convolutions = Integer(low=1, high=5, name='n_convolutions')
n = Integer(low=2, high=8, name='n')

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
        flag = True
        s = 1
        while flag:
            if dim == 1:
                model.add(Masksembles1D(n, s))  # 4
                flag = False
            else:  # dim==2
                model.add(Masksembles2D(n, s))
                flag = False
            s += 1
    else:  # model_to_run is "pruned_masksembles"
        flag = True
        s = 1
        while flag:
            if dim == 1:
                model.add(prune_low_magnitude(MyMasksembles1D(n, s)))
                flag = False
            else:  # dim==2
                model.add(prune_low_magnitude(MyMasksembles2D(n, s)))
                flag = False
            s += 1


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
    # model.summary()

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# endregion

def divided(x_check, y_check, n):
    """
    This function verifies that the data is divided by N as required in the configuration of maskensemble
    :param x_check: x data
    :param y_check: y data
    :param n:
    :return: return x and y that divided in n
    """
    while x_check.shape[0] % n != 0:
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
                             model_to_run=model_to_run)
        X_train, y_train = divided(X_train, y_train, n)
        X_val, y_val = divided(X_val, y_val, n)

        history = model.fit(x=X_train,
                            y=y_train,
                            # TODO: change number epoch
                            epochs=1,
                            batch_size=16 * n,
                            validation_data=(X_val, y_val),
                            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()], verbose=0
                            )
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


def evaluate_on_test(y_true, y_pred, scores):
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
    pr = 0
    plt.clf()
    for i in n_classes:
        pr += pr_auc_score(y_true[:, i], y_pred[:, i])
    scores['pr_auc_score'] = pr / len(n_classes)
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
# TODO: run before all running
# open_dirs()

all_score = {}
model_to_run = "masksembles"
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

    for train_val_index, test_index in kf_external.split(X, labels):
        try:
            X_train_val, X_test = X[train_val_index], X[test_index]
            Y_train_val, Y_test = Y[train_val_index], Y[test_index]

            best_accuracy = 0.0
            # search_result = gp_minimize(func=fitness,
            #                             dimensions=dimensions,
            #                             acq_func='EI',  # Expected Improvement.
            #                             # TODO : change to 50
            #                             n_calls=11,
            #                             x0=default_parameters)
            #
            # results[tuple(search_result.x)] = best_accuracy
            opt_par = default_parameters

            # opt_par = search_result.x
            learning_rate = opt_par[0]
            num_layers = opt_par[1]
            num_nodes = opt_par[2]
            X_test, Y_test = divided(X_test, Y_test, num_nodes)
            best_model = create_model(learning_rate, num_layers, num_nodes, model_to_run)
            X_train_val, Y_train_val = divided(X_train_val, Y_train_val, num_nodes)
            start_train = time()
            # TODO: change epoch to 100
            history = best_model.fit(X_train_val, Y_train_val, epochs=1, batch_size=16 * num_nodes, verbose=0)
            end_train = time() - start_train
            y_pred = best_model.predict(X_test, batch_size=16 * num_nodes)
            score = {'accuracy_score': -1, "fpr": -1, 'tpr': -1, 'precision_score': -1, 'recall_score': -1,
                     'auc_score': -1, 'pr_auc_score': -1, 'Training_time': -1, 'inference_time': -1}
            try:
                score = evaluate_on_test(Y_test, y_pred, score)
            except:
                pass
            score['Training_time'] = end_train

            if len(test_index) > 1000:
                X_test = X_test[:1000]
            start_test = time()
            best_model.predict(X_test, batch_size=16 * num_nodes)
            end_test = time() - start_test
            score['inference_time'] = end_test
            # all_score[f"{ds_name}:{index_cv}"] = [float(i) if not isinstance(i, str) else i for i in search_result.x] + \
            #                                      [float(i) for i in list(score.values())]
            # if index_cv == 0:
            #     best_result(search_result, ds_name, index_cv=index_cv)
            index_cv += 1
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
