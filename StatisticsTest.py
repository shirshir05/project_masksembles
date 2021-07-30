import os

import scipy.stats
import scikit_posthocs as sp
import pandas as pd


def average_cv(model):
    df = pd.read_csv(os.path.join("Results", model + ".csv"))
    dataset = set(df["Dataset"].values)
    average_df = pd.DataFrame(columns=df.columns)
    for data in list(dataset):
        average_df = average_df.append(pd.Series({"Dataset": data}).append(df[df["Dataset"] == data].mean(axis=0)).to_frame().T)
    return average_df, average_df.iloc[:,5]


average_basic, metrics_basic = average_cv("basic")
average_maskensemble, metrics_maskensemble = average_cv("maskensemble")

alg_1, alg_2, alg_3 = None, None, None
# todo: AUC
if scipy.stats.friedmanchisquare(alg_1, alg_2, alg_3).pvalue < 0.05:
    # TODO:
    df = alg_1 + alg_2 + alg_3
    sp.posthoc_ttest(df, val_col='AUC', group_col='Algorithm', p_adjust='holm')
