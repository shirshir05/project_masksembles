import os
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
# TODO pip install git+http://github.com/nikitadurasov/masksembles
from masksembles.keras import Masksembles1D, Masksembles2D
from sklearn.metrics import confusion_matrix, precision_recall_curve
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib as mpl
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
# TODO: pip install scikit-optimize
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_objective
from skopt.utils import use_named_args
from skopt.plots import plot_objective_2D

from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# TODO: pip install tensorflow_datasets
import tensorflow_datasets as tfds
from keras.utils.np_utils import to_categorical

# regionConfig parameters
# all dataset were taken from: https://www.tensorflow.org/datasets/catalog/overview
datasets_info = {
    "mnist": [70000, 10],
    # "beans": [1295, 3],
    # "binary_alpha_digits": [1404, 36],
    # "cifar10": [60000, 10],
    # "citrus_leaves": [425, 4],
    # "stanford_dogs": [12000, 120],
    # "cassava": [9430, 5],
    # "rock_paper_scissors": [2520, 3],

    # "horses_or_humans": [1280, 2],
    # "dmlab":[65550,6],
    # "food101":[75750,101],
    # "cmaterdb":[5000,10],
    # "stanford_online_products": [59551, 12],
    # "stl10": [5000, 10],
    # "tf_flowers": [2670, 5],
    # "cats_vs_dogs":[23262,2],
    # "uc_merced": [2100, 21],
    # "kmnist": [60000, 10],
    # "oxford_flowers102": [8189, 102],
    # "food101": [75750, 101],
    # "deep_weeds": [17509, 9],
    # "eurosat": [27000, 10],

}  # "dataset_name": [n_samples, NUM_CLASSES]

MAX_SAMPLES_NUM = 320
# path of the best model, contains optimized hyper-parameter.
path_best_model = 'BayesianOptimization/19_best_model.h5'
# define global variable to store accuracy, and random state
random_state = 42
best_accuracy = 0.0

# hyper-parameters tuned in this project:
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

# TODO: change default parameters
default_parameters = [1e-5, 1, 16, 'relu']


def log_dir_name(learning_rate, num_dense_layers, num_dense_nodes, activation):
    '''
    This is a function to log training progress so that can be viewed by TnesorBoard.
    '''
    # The dir-name for the TensorBoard log-dir.
    s = "./BayesianOptimization/19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       activation)
    return log_dir


# endregion

# region Config model
# TODO: change create model and the tuning parameters

def create_model(learning_rate, num_dense_layers, num_dense_nodes, activation):
    '''
    The method builds the model according to the examined hyper-parameters.

    '''
    # TODO: this model was taken from https://github.com/nikitadurasov/masksembles/blob/main/notebooks/MNIST_Masksembles.ipynb
    # TODO: optimized parameters should be considered here.
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),  # input shape changes according to the dataset
            layers.Conv2D(32, kernel_size=(3, 3), activation="elu"),
            Masksembles2D(4, 2.0),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, kernel_size=(3, 3), activation="elu"),
            Masksembles2D(4, 2.0),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            Masksembles1D(4, 2.),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.summary()

    # model = Sequential()
    # model.add(InputLayer(input_shape=(input_shape)))  # input shape changes according to the dataset
    # Add num_dense_layers dense layers
    # for i in range(num_dense_layers):
    #     name = 'layer_dense_{0}'.format(i + 1)
    #     model.add(Dense(num_dense_nodes,
    #                     activation=activation,
    #                     name=name))
    # model.add(Dense(labels.shape[1], activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    # TODO: change metric
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# endregion


# TODO: change tuning parameters
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

    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
    average_accuracy = 0

    for train_index, val_index in kf.split(X_train_val, Y_train_val):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = Y_train_val[train_index], Y_train_val[val_index]

        # Create the neural network with these hyper-parameters.
        model = create_model(learning_rate=learning_rate,
                             num_dense_layers=num_dense_layers,
                             num_dense_nodes=num_dense_nodes,
                             activation=activation)

        # Dir-name for the TensorBoard log-files.
        # log_dir = log_dir_name(learning_rate, num_dense_layers,
        #                        num_dense_nodes, activation)

        # callback_log = TensorBoard(
        #     log_dir=log_dir,
        #     histogram_freq=0,
        #     write_graph=True,
        #     write_grads=False,
        #     write_images=False)

        # TODO: change number epoch
        history = model.fit(x=X_train,
                            y=y_train,
                            epochs=1,
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

    # Update best accuracy rate
    global best_accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy

    # Returns negative because skpot performs minimization rather than maximization.
    return -accuracy


def best_result(search_result, index_cv=0):
    plt.clf()
    if not os.path.exists("BayesianOptimization/" + str(index_cv)):
        os.mkdir("BayesianOptimization/" + str(index_cv))
    plot_convergence(search_result)
    plt.savefig("BayesianOptimization/" + str(index_cv) + "/Converge.png", dpi=400)

    print(f" search result {search_result.x}")
    print(f"The best fitness value associated with these hyper-parameters {search_result.fun}")

    fig = plot_objective_2D(result=search_result,
                            dimension_identifier1='learning_rate',
                            dimension_identifier2='num_dense_nodes',
                            levels=50)
    plt.savefig("BayesianOptimization/" + str(index_cv) + "/Lr_numnods.png", dpi=400)

    # create a list for plotting
    dim_names = ['learning_rate', 'num_dense_layers', 'num_dense_nodes', 'activation']
    plot_objective(result=search_result, dimensions=dim_names)
    plt.savefig("BayesianOptimization/" + str(index_cv) + "/all_dimen.png", dpi=400)
    plot_evaluations(result=search_result, dimensions=dim_names)


def evaluate_on_test(y_true, y_pred):
    scores = {}
    scores['accuracy_score'] = metrics.accuracy_score(y_true, y_pred)
    scores['precision_score'] = metrics.precision_score(y_true, y_pred)
    scores['recall_score'] = metrics.recall_score(y_true, y_pred)
    scores['roc_auc_score'] = metrics.roc_auc_score(y_true, y_pred)
    scores['tn'], scores['fp'], scores['fn'], scores['tp'] = [int(i) for i in
                                                              list(confusion_matrix(y_true, y_pred).ravel())]
    scores['fpr'], scores['tpr'], thresholds = metrics.roc_curve(y_true, y_pred)
    scores['precision'], scores['recall'], scores['thresholds'] = precision_recall_curve(y_true, y_pred)
    return scores


# ================================== start of the script ==================================

# create BayesianOptimization folder
if not os.path.exists("BayesianOptimization"):
    os.mkdir("BayesianOptimization")

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
    input_shape = (X.shape[1], X.shape[2], X.shape[3])
    print(f"data shape: {X.shape}")

    # TODO: preprocess data

    # perform nested cross validation for hyper-parameter optimization and generalization
    # TODO: add time:Training time and Inference time for 1000 instances
    kf_external = KFold(n_splits=10, shuffle=True, random_state=random_state)
    index_cv = 0
    results = {}

    for train_val_index, test_index in kf_external.split(X, Y):
        X_train_val, X_test = X[train_val_index], X[test_index]
        Y_train_val, Y_test = Y[train_val_index], Y[test_index]
        best_accuracy = 0.0
        search_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    acq_func='EI',  # Expected Improvement.
                                    # TODO change to 50
                                    n_calls=11,
                                    x0=default_parameters)
        best_result(search_result, index_cv=index_cv)
        index_cv += 1
        results[tuple(search_result.x)] = best_accuracy

        # best_model = load_model(path_best_model)
        opt_par = search_result.x
        learning_rate = opt_par[0]
        num_layers = opt_par[1]
        num_nodes = opt_par[2]
        activation = opt_par[3]
        best_model = create_model(learning_rate, num_layers, num_nodes, activation)
        history = best_model.fit(X_train_val, Y_train_val)
        y_pred = best_model.predict(X_test)
        # evaluate_on_test(Y_test, y_pred)
    print(results)

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
