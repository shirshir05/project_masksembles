import os

import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_curve
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib as mpl
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# TODO: pip install scikit-optimize
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_objective
from skopt.utils import use_named_args
from skopt.plots import plot_objective_2D

# base on - https://github.com/mardani72/Hyper-Parameter_optimization/blob/master/Hyper_Param_Facies_tf_final.ipynb

# TODO: create folder name: BayesianOptimization
# TODO: replace all region of data
# region remove data
df = pd.read_csv('Data/training_data.csv')
# specify some data types may python concern about
df['Facies'] = df['Facies'].astype('int')
df['Depth'] = df['Depth'].astype('float')
df['Well Name'] = df['Well Name'].astype('category')
df['Formation'] = df['Formation'].astype('category')

# colors
facies_colors = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00',
                 '#1B4F72', '#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D', 'PS', 'BS']
# facies_color_map is a dictionary that maps facies labels to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]


def label_facies(row, labels):
    return labels[row['Facies'] - 1]


# establish facies label str
df.loc[:, 'FaciesLabels'] = df.apply(lambda row: label_facies(row, facies_labels), axis=1)
blind = df[df['Well Name'] == 'SHANKLE']
training_data = df[df['Well Name'] != 'SHANKLE']

dummies = pd.get_dummies(training_data['FaciesLabels'])
Facies_cat = dummies.columns
labels = dummies.values  # target matirx
# select predictors
features = training_data.drop(['Facies', 'Formation', 'Well Name',
                               'Depth', 'FaciesLabels'], axis=1)

from sklearn import preprocessing, metrics

scaler = preprocessing.StandardScaler().fit(features)
scaled_features = scaler.transform(features)
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# endregion


# regionConfig parameters
# TODO: Change Hyper-Parameters Tuning
dim_learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform', name='learning_rate')
dim_num_dense_layers = Integer(low=1, high=10, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'], name='activation')
dimensions = [dim_learning_rate, dim_num_dense_layers, dim_num_dense_nodes, dim_activation]
# TODO: change default parameters
default_parameters = [1e-5, 1, 16, 'relu']
path_best_model = 'BayesianOptimization/19_best_model.h5'
# define global variable to store accuracy
random_state = 42
best_accuracy = 0.0
# This is a function to log traning progress so that can be viewed by TnesorBoard.
def log_dir_name(learning_rate, num_dense_layers,
                 num_dense_nodes, activation):
    # The dir-name for the TensorBoard log-dir.
    s = "./BayesianOptimization/19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       activation)
    return log_dir
#endregion

# region Config model
# TODO: change create model and the tuning parameters
def create_model(learning_rate, num_dense_layers, num_dense_nodes, activation):
    model = Sequential()

    model.add(InputLayer(input_shape=(scaled_features.shape[1])))

    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i + 1)
        model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name))
    model.add(Dense(labels.shape[1], activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    # TODO: change metric
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#endregion


# TODO: change tuning parameters
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation):
    """
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
        X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train, y_val = Y_train_val.iloc[train_index], Y_train_val.iloc[val_index]

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
                            batch_size=128,
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


# TODO: change data
X = pd.DataFrame(scaled_features)
Y = pd.DataFrame(labels)



# TODO: add time:Training time and Inference time for 1000 instance
kf_external = KFold(n_splits=10, shuffle=True, random_state=random_state)
index_cv = 0
results = {}
for train_val_index, test_index in kf_external.split(X, Y):
    X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
    Y_train_val, Y_test = Y.iloc[train_val_index], Y.iloc[test_index]
    best_accuracy = 0.0
    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                #TODO change to 50
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


