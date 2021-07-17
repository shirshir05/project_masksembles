import json
import os
from time import time

import numpy as np

# TODO pip install git+http://github.com/nikitadurasov/masksembles
from masksembles.keras import Masksembles1D, Masksembles2D
from sklearn.metrics import confusion_matrix, precision_recall_curve
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
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

# TODO: code documentation
random_state = 42
best_accuracy = 0.0

# region dataset
# all dataset were taken from: https://www.tensorflow.org/datasets/catalog/overview
datasets_info = {

    "binary_alpha_digits": [1404, 36],
    "cifar10": [60000, 10],
    "citrus_leaves": [425, 4],
    "cassava": [9430, 5],
    "rock_paper_scissors": [2520, 3],
    "horses_or_humans": [1280, 2],
    "dmlab": [65550, 6],
    "food101": [75750, 101],
    "cmaterdb": [5000, 10],

    "stl10": [5000, 10],
    "tf_flowers": [2670, 5],
    "cats_vs_dogs": [23262, 2],
    "uc_merced": [2100, 21],
    "kmnist": [60000, 10],
    # "oxford_flowers102": [8189, 102], # TODO: n_splits=10 cannot be greater than the number of members in each class.
    "deep_weeds": [17509, 9],
    "eurosat": [27000, 10],
    "mnist": [70000, 10],
    "beans": [1295, 3],

    # ,"stanford_online_products": [59551, 12],  # TODO: check not support  as_supervised=True
    # ,"stanford_dogs": [12000, 120], # TODO: check Unable to allocate 22.5 GiB for an array with shape (3019898880,) and data type float64

}  # "dataset_name": [n_samples, NUM_CLASSES]
MAX_SAMPLES_NUM = 320
# endregion

# region hyper-parameters tuned in this project:
# 1. Learning rate
# 2. Number of dense layers
# 3. Number of nodes for each layers
# 4. Which activation function: 'relu' or sigmoid
# TODO: Change Hyper-Parameters Tuning values
dim_learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform', name='learning_rate')
dim_num_dense_layers = Integer(low=1, high=10, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'], name='activation')

# hold all examnined hyper-parameters dimention i a list
dimensions = [dim_learning_rate, dim_num_dense_layers, dim_num_dense_nodes, dim_activation]
default_parameters = [1e-5, 1, 16, 'relu']


# endregion

# region Config model
def create_model(learning_rate, num_dense_layers, num_dense_nodes, activation):
    '''
    The method builds the model according to the examined hyper-parameters.
    this model was taken from https://github.com/nikitadurasov/masksembles/blob/main/notebooks/MNIST_Masksembles.ipynb
    '''
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="elu"))
    model.add(Masksembles2D(4, 2.0))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="elu"))
    model.add(Masksembles2D(4, 2.0))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(Masksembles1D(4, 2.))
    # Add num_dense_layers dense layers
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i + 1)
        model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name))
    model.add(layers.Dense(n_classes, activation="softmax"))
    # model.summary()

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
'''
Parameter optimization is based on:
https://github.com/mardani72/Hyper-Parameter_optimization/blob/master/Hyper_Param_Facies_tf_final.ipynb
'''


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation):
    """
    This function aims to create and train a network with given hyper-parameters
    and then evaluate model performance with the validation dataset.
    It returns fitness value, negative classification accuracy on the dataset.
    It is negative because skpot performs minimization rather than maximization.
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:', activation)
    print()

    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    average_accuracy = 0
    y_label = np.argmax(Y_train_val, axis=1)
    for train_index, val_index in kf.split(X_train_val, y_label):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = Y_train_val[train_index], Y_train_val[val_index]

        # Create the neural network with these hyper-parameters.
        model = create_model(learning_rate=learning_rate,
                             num_dense_layers=num_dense_layers,
                             num_dense_nodes=num_dense_nodes,
                             activation=activation)
        X_train, y_train = divied_4(X_train, y_train)
        X_val, y_val = divied_4(X_val, y_val)

        # TODO: change number epoch
        history = model.fit(x=X_train,
                            y=y_train,
                            epochs=100,
                            batch_size=32,
                            validation_data=(X_val, y_val),
                            # callbacks=[callback_log]
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
                            dimension_identifier2='num_dense_nodes',
                            levels=50)
    plt.savefig(os.path.join("BayesianOptimization", ds_name, str(index_cv), "Lr_numnods.png"))
    plt.clf()
    # create a list for plotting
    dim_names = ['learning_rate', 'num_dense_layers', 'num_dense_nodes', 'activation']
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


# ================================== start of the script ==================================

# create BayesianOptimization folder
if not os.path.exists("BayesianOptimization"):
    os.mkdir("BayesianOptimization")

all_score = {}

# TODO: create in cluster - to prevent error
# for ds_name in datasets_info:
#     print(f"uploading dataset: {ds_name}")
#     for i in range(10):
#         if not os.path.exists(os.path.join("BayesianOptimization", ds_name)):
#             os.mkdir(os.path.join("BayesianOptimization", ds_name))
#         if not os.path.exists(os.path.join("BayesianOptimization", ds_name, str(i))):
#             os.mkdir(os.path.join("BayesianOptimization", ds_name, str(i)))


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

    # TODO: preprocess data
    X = X / 255
    X = np.resize(X, (X.shape[0], 32, 32, 1))
    print(f"data shape after: {X.shape}")
    input_shape = (X.shape[1], X.shape[2], X.shape[3])

    # perform nested cross validation for hyper-parameter optimization and generalization
    # TODO: change 3 to 10
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
                                        # TODO change to 50
                                        n_calls=50,
                                        x0=default_parameters)

            results[tuple(search_result.x)] = best_accuracy

            opt_par = search_result.x
            learning_rate = opt_par[0]
            num_layers = opt_par[1]
            num_nodes = opt_par[2]
            activation = opt_par[3]
            best_model = create_model(learning_rate, num_layers, num_nodes, activation)
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

# 0. Actually read the paper
# 1. picture dataset - Amit. Prepare the pipeline for training. detail the datasets in the report
# 2. optimization - after reading
# 3. evaluate_on_test
# 4. anova (Statistical significance testing)
# 5. improvement
