import scipy.stats
import scikit_posthocs as sp


# TODO: read data and average on croos validation
alg_1, alg_2, alg_3 = None, None, None
# todo: AUC
if scipy.stats.friedmanchisquare(alg_1, alg_2, alg_3).pvalue < 0.05:
    # TODO:
    df = alg_1 + alg_2 + alg_3
    sp.posthoc_ttest(df, val_col='AUC', group_col='Algorithm', p_adjust='holm')

