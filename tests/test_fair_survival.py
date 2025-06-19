import numpy as np
import pytest
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.ensemble import FairRandomSurvivalForest  # Your custom class
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sksurv.metrics import concordance_index_censored

from itertools import product
from queue import LifoQueue

from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import sparse,stats
from sklearn.pipeline import make_pipeline
from sklearn.tree._tree import TREE_UNDEFINED

from sksurv.compare import compare_survival
from sksurv.datasets import load_breast_cancer, load_veterans_lung_cancer
from sksurv.nonparametric import kaplan_meier_estimator, nelson_aalen_estimator
from sksurv.tree import SurvivalTree, FairSurvivalTree
from sksurv.util import Surv

from collections import OrderedDict
import pandas as pd
from sklearn.utils.validation import check_array

import sys
import os
sys.path.append(os.path.abspath(".."))  # adds the parent directory to the path

from sksurv.util import check_array_survival 

__all__ = ["compare_survival"]


@pytest.fixture
def data():
    
    # Step 1: Load data
    X_raw, y = load_veterans_lung_cancer()

    # Use Celltype as the group for testing purposes
    group_raw = X_raw["Celltype"]
    group_encoder = LabelEncoder()
    group = group_encoder.fit_transform(group_raw).astype(np.intp)

    group = np.asarray(group, dtype=np.intp)

    categorical_cols = ['Celltype','Age_in_years', 'Prior_therapy', 'Treatment']
    encoder = OrdinalEncoder()
    X_encoded = X_raw.copy()
    X_encoded[categorical_cols] = encoder.fit_transform(X_raw[categorical_cols])

    # Step 2: Split into train/test
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(X_encoded, y, group, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, group_train, group_test

    
def test_fair_survival_forest_runs(data):
    X_train, X_test, y_train, y_test, group_train, group_test = data
    
    model = FairRandomSurvivalForest(n_estimators=1, max_depth=3, random_state=0)
    model.fit(X_train, y_train, group_train)

    pred = model.predict(X_test)
    assert pred.shape[0] == X_test.shape[0]
    
def concordance_imparity(risk_scores, times, events, group, num_groups):
    """_summary_

    Args:
        model (_type_): survival model
        X_test (_type_): a pandas dataframe containing the features of the test set
        y_test (_type_): an sksurv ndarray containing status and survival time
        group (_type_): an array containing the label-encoded values of the sensitive group in the test set

    Returns:
        CI (float): The concordance imparity of the model
    """
    
    #risk_scores = model.predict(X_test)
    
    #unique_values_set = set(group)
    #num_groups = len(unique_values_set)
    
    P = [0] * num_groups
    C = [0] * num_groups
    CF = [0] * num_groups
    
    for i in range(len(times)):
        d_i = events[i]
        t_i = times[i]
        r_i = risk_scores[i]
        g_i = group[i]
        
        for j in range(len(times)):
            if j == i:
                continue
            else:
                    d_j = events[j]
                    t_j = times[j]
                    r_j = risk_scores[j]
                    
                    if ((t_i < t_j) and (d_i == 0)) or ((t_j < t_i) and (d_j == 0)) or ((t_i == t_j) and ((d_i == 0) and (d_j == 0))):
                        continue
                    else:
                        P[g_i] += 1
                    
                    if t_i < t_j:
                        if r_i > r_j:
                            C[g_i] += 1
                        elif r_i == r_j:
                            C[g_i] += 0.5
                    elif t_i > t_j:
                        if r_i < r_j:
                            C[g_i] += 1
                        elif r_i == r_j:
                            C[g_i] += 0.5
                    elif t_i == t_j:
                        if d_i == 1 and d_j == 1:
                            if r_i == r_j:
                                C[g_i] += 1
                            else:
                                C[g_i] += 0.5
                        elif d_i == 0 and d_j == 1 and (r_i < r_j):
                            C[g_i] += 1
                        elif d_i == 1 and d_j == 0 and (r_i > r_j):
                            C[g_i] += 1
                        else:
                            C[g_i] += 0.5
                            
    CF = [x/y for x,y in zip(C,P)]
    
    max_diff = 0
    for g1 in range(num_groups):
            for g2 in range(g1 + 1, num_groups):
                diff = abs(CF[g1] - CF[g2])
                if diff > max_diff:
                    max_diff = diff

    return np.abs(max_diff)


def test_concordance_imparity_simple_case():
    # Simulate a toy dataset
    times = np.array([5, 6, 7, 8])
    events = np.array([1, 1, 1, 1])
    risks = np.array([0.2, 0.4, 0.1, 0.1])
    groups = np.array([0, 0, 1, 1])
    num_groups = 2

    # Run the concordance imparity function
    ci = concordance_imparity(risks, times, events, groups, num_groups)

    # Make sure the result is finite and non-negative
    assert np.isfinite(ci)
    assert ci >= 0
    
    C = [4,5]
    P = [6,6]
    CF = [x/y for x,y in zip(C,P)]
    expected_value = np.abs(CF[0] - CF[1])

    # Optional: assert exact value if known (e.g., from hand calculation)
    assert np.isclose(ci, expected_value, atol=1e-5)





def compare_survival(y, group_indicator, return_stats=False):
    """K-sample log-rank hypothesis test of identical survival functions.

    Compares the pooled hazard rate with each group-specific
    hazard rate. The alternative hypothesis is that the hazard
    rate of at least one group differs from the others at some time.

    See [1]_ for more details.

    Parameters
    ----------
    y : structured array, shape = (n_samples,)
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    group_indicator : array-like, shape = (n_samples,)
        Group membership of each sample.

    return_stats : bool, optional, default: False
        Whether to return a data frame with statistics for each group
        and the covariance matrix of the test statistic.

    Returns
    -------
    chisq : float
        Test statistic.
    pvalue : float
        Two-sided p-value with respect to the null hypothesis
        that the hazard rates across all groups are equal.
    stats : pandas.DataFrame
        Summary statistics for each group:  number of samples,
        observed number of events, expected number of events,
        and test statistic.
        Only provided if `return_stats` is True.
    covariance : array, shape=(n_groups, n_groups)
        Covariance matrix of the test statistic.
        Only provided if `return_stats` is True.

    References
    ----------
    .. [1] Fleming, T. R. and Harrington, D. P.
           A Class of Hypothesis Tests for One and Two Samples of Censored Survival Data.
           Communications In Statistics 10 (1981): 763-794.
    """

    event, time = check_array_survival(group_indicator, y)
    group_indicator = check_array(
        group_indicator,
        dtype="O",
        ensure_2d=False,
        estimator="compare_survival",
        input_name="group_indicator",
    )

    n_samples = time.shape[0]
    groups, group_counts = np.unique(group_indicator, return_counts=True)
    n_groups = groups.shape[0]
    if n_groups == 1:
        raise ValueError("At least two groups must be specified, but only one was provided.")

    # sort descending
    o = np.argsort(-time, kind="mergesort")
    x = group_indicator[o]
    event = event[o]
    time = time[o]

    at_risk = np.zeros(n_groups, dtype=int)
    observed = np.zeros(n_groups, dtype=int)
    expected = np.zeros(n_groups, dtype=float)
    covar = np.zeros((n_groups, n_groups), dtype=float)

    covar_indices = np.diag_indices(n_groups)

    k = 0
    while k < n_samples:
        ti = time[k]
        total_events = 0
        while k < n_samples and ti == time[k]:
            idx = np.searchsorted(groups, x[k])
            if event[k]:
                observed[idx] += 1
                total_events += 1
            at_risk[idx] += 1
            k += 1

        if total_events != 0:
            total_at_risk = k
            expected += at_risk * (total_events / total_at_risk)
            if total_at_risk > 1:
                multiplier = total_events * (total_at_risk - total_events) / (total_at_risk * (total_at_risk - 1))
                temp = at_risk * multiplier
                covar[covar_indices] += temp
                covar -= np.outer(temp, at_risk) / total_at_risk

    df = n_groups - 1
    zz = observed[:df] - expected[:df]
    chisq = np.linalg.solve(covar[:df, :df], zz).dot(zz)
    pval = stats.chi2.sf(chisq, df)

    if return_stats:
        table = OrderedDict()
        table["counts"] = group_counts
        table["observed"] = observed
        table["expected"] = expected
        table["statistic"] = observed - expected
        table = pd.DataFrame.from_dict(table)
        table.index = pd.Index(groups, name="group", dtype=groups.dtype)
        return chisq, pval, table, covar

    return chisq, pval


"""
def test_group_split_behavior(data):
    X_train, _, y_train, _, group, = data
    model = FairRandomSurvivalForest(n_estimators=1, max_depth=3, random_state=0)
    model.fit(X_train, y_train, group)

    # Pull the trained tree object
    estimator = model.estimators_[0]
    node_groups = estimator.decision_path(X_train).toarray().argmax(axis=1)

    # Group node assignment check
    group_labels = {}
    for node, grp in zip(node_groups, group):
        if node not in group_labels:
            group_labels[node] = []
        group_labels[node].append(grp)

    # Check at least one node has a mix (only for debugging fairness)
    diverse = any(len(set(grps)) > 1 for grps in group_labels.values())
    assert diverse, "No node has mixed group membership; fairness criterion may be inactive"
   

def test_criterion_accessibility(data):
    X_train, _, y_train, _, group, _ = data
    model = FairRandomSurvivalForest(n_estimators=1, max_depth=2, random_state=42)
    model.fit(X_train, y_train, group)

    criterion = model.estimators_[0].tree_.criterion
    assert hasattr(criterion, "concordance_imparity"), "Criterion missing concordance_imparity method"
""" 

class FairSurvivalDifferenceTreeBuilder:
    def __init__(self, max_depth=4, min_leaf=20):
        self.max_depth = max_depth
        self.min_leaf = min_leaf

    def build(self, X, y):
        val, feat, stat = self._get_best_split(X, y)
        splits = LifoQueue()
        splits.put((val, feat, stat, 0, np.arange(X.shape[0])))

        node_stats = []
        while splits.qsize() > 0:
            val, feat, stat, lvl, idx = splits.get()
            s = {"feature": feat, "threshold": val, "n_node_samples": idx.shape[0], "statistic": stat, "depth": lvl}
            node_stats.append(s)

            if val == TREE_UNDEFINED:
                continue

            left = X[idx, feat] <= val
            right = idx[~left]
            left = idx[left]

            if lvl == self.max_depth - 1:
                splits.put([TREE_UNDEFINED, TREE_UNDEFINED, -np.inf, lvl + 1, right])
                splits.put([TREE_UNDEFINED, TREE_UNDEFINED, -np.inf, lvl + 1, left])
                continue

            X_right = X[right, :]
            y_right = y[right]
            s_right = self._get_best_split(X_right, y_right)
            splits.put(list(s_right) + [lvl + 1, right])

            X_left = X[left, :]
            y_left = y[left]
            s_left = self._get_best_split(X_left, y_left)
            splits.put(list(s_left) + [lvl + 1, left])

        return pd.DataFrame.from_dict(dict(zip(range(len(node_stats)), node_stats)), orient="index")

    def _get_best_split(self, X, y):
        min_leaf = self.min_leaf
        best_val = TREE_UNDEFINED
        best_feat = TREE_UNDEFINED
        best_stat = -np.inf

        if y[y.dtype.names[0]].sum() == 0:
            return best_val, best_feat, best_stat
        for j in range(X.shape[1]):
            vals = X[:, j]
            values = np.unique(vals)
            if len(values) < 2:
                continue

            for i, v in enumerate(values[:-1]):
                t = (v + values[i + 1]) * 0.5
                groups = (vals <= t).astype(int)
                if groups.sum() >= min_leaf and (X.shape[0] - groups.sum()) >= min_leaf:
                    s, _ = compare_survival(y, groups)
                    if s > best_stat:
                        best_feat = j
                        best_val = t
                        best_stat = s
        return best_val, best_feat, best_stat
