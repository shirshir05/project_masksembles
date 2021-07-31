import os
import numpy as np
import scipy.stats
import scikit_posthocs as sp
import pandas as pd


def average_cv(model):
    df = pd.read_csv(os.path.join("Results", model + ".csv"))
    dataset = set(df["Dataset"].values)
    average_df = pd.DataFrame(columns=df.columns)
    for data in list(dataset):
        average_df = average_df.append(
            pd.Series({"Dataset": data}).append(df[df["Dataset"] == data].mean(axis=0)).to_frame().T)
    return average_df, average_df[
        ['Accuracy score', 'FPR', 'TPR', 'pecision score', 'Recall score', 'Auc score', 'Pr auc score']]


average_basic, metrics_basic = average_cv("basic")
average_maskensemble, metrics_maskensemble = average_cv("Masksembles")
average_pruned, metrics_pruned = average_cv("Pruned")

if scipy.stats.friedmanchisquare(metrics_pruned, metrics_maskensemble, metrics_basic).pvalue < 0.05:
    print(scipy.stats.friedmanchisquare(metrics_pruned, metrics_maskensemble, metrics_basic).pvalue)
    for measures in metrics_pruned.columns:
        print(measures)
        col = np.array([metrics_basic[measures], metrics_pruned[measures],  metrics_maskensemble[measures]]).T
        print(sp.posthoc_nemenyi_friedman(col))
