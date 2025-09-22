import numpy as np
import csv

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils import check_X_y
import sys
sys.path.append('..')

from aif360.datasets import BinaryLabelDataset

import fairness_intervention as fair

"""
Code related to the measure of Consistency [#zemel13]_ Individual Consistency Score (ICS, c(x)) [#waller25]_ and Balanced Conditioned Consistency (BCC) [#waller25]_

ICS : Given a target label y(x) for dataset E and individual x in E, the ICS c(x) computes the proportion of x's k most similar individuals with the same label as x.
c(x) = 1 - |y(x) - \frac{1}{k} \sum_{x' in knn(x)} y(x')|
    where knn(x) is the set of k nearest neighbors of x

BCC : Sum of the individual consistency scores c(x) above or equal some threshold t divided by the total number N of individuals in dataset E
BCC(E) = \frac{1}{N} \sum_{x \in E} v(x)
    where v(x) = c(x) if c(x) >= t and 0 otherwise

References:
        .. [#zemel13] `R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,
               "Learning Fair Representations," International Conference on Machine
               Learning, 2013. <http://proceedings.mlr.press/v28/zemel13.html>`_

        .. [#waller25] `M. Waller, O. Rodrigues, and O. Cocarascu, 2025.
            "Beyond Consistency: Nuanced Metrics for Individual Fairness."
            In Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency (FAccT '25).
            Association for Computing Machinery, New York, NY, USA, 2087–2097.
            <https://doi.org/10.1145/3715275.3732141>`_

"""

def individual_consistency_scores(dataset: BinaryLabelDataset, k: int = 5) :
    """ Compute individual consistency scores [#waller25] for all individual in dataset, considering k nearest neighbors for proximity
    dataset : BinaryLabelDataset
        dataset containing the individuals for which Individual Consistency will be computed
    k : int
        number of individuals considered for the nearest neighbor algorithm
    returns : Numpy array
        array containing the individual consistency scores for each individual present in 'dataset'
    """
    #preprocess features before measuring proximity : remove sensitive attribute + scale features values
    dataset_mod = dataset.copy()
    dataset_mod.features = fair.blind_features(dataset) #remove sensitice attribute
    min_max_scaler = MaxAbsScaler()
    dataset_mod.features = min_max_scaler.fit_transform(dataset_mod.features) #scale
    #Warning, data_prox.feature_names does not correspond to data_prox.features

    y = dataset_mod.labels

    k = k+1 #because the individual itself is returned by NearestNeighbors but will be removed
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nbrs.fit(dataset_mod.features)
    indices = nbrs.kneighbors(dataset_mod.features, return_distance=False)

    # Code bellow adapted from https://github.com/maddiewaller/MetricsForIndividualFairness
    # remove the index (individual) itself to keep only the neighbors
    sim_indices = []
    for i in range(len(indices)):
        if i in indices[i]:
            sim_indices.append(np.delete(indices[i], np.where(indices[i] == i)))
        else:
            sim_indices.append(indices[i][:-1])

    ind_consistency = 1 - abs(y - y[sim_indices].mean(axis=1))
    return ind_consistency.ravel()


def bcc_from_ind_const(ind_const: np.array, threshold:float = 0.8):
    """ Compute Balanced Conditioned Consistency [#waller2025] based on the individual consistency scores [#waller2025] given in 'ind_const'
    Code taken and slightly adapted from https://github.com/maddiewaller/MetricsForIndividualFairness (functions.py, get_bcc_scores_half)
    ind_const: Numpy array
        1D array containing the ICS
    threshold : float
        threshold under which individual consistency scores are not considered to compute BCC
    returns : float
        BCC value for the set of individuals represented in 'ind_const'
    """
    count = 0
    for cx in ind_const:
        if round(cx,2) >= threshold:
            count = count + cx
    bcc = (count/len(ind_const))
    return bcc

def bcc_penalty_from_ind_const(ind_const: np.array, threshold:float = 0.8, penalty:float = 1):
    """ Compute Balanced Conditioned Consistency with penalty [#waller2025] based on the individual consistency scores [#waller2025] given in 'ind_const'
    Code taken and slightly adapted from https://github.com/maddiewaller/MetricsForIndividualFairness (functions.py, get_bcc_penalty_scores_half)
    ind_const: Numpy array
        1D array containing the ICS
    threshold : float
        threshold under which individual consistency scores are replaced by penalty in BCC sum
    returns : float
        BCC with penalty value for the set of individuals represented in 'ind_const'
    """
    count = 0
    for cx in ind_const:
        if round(cx,2) >= threshold:
            count = count + cx
        else:
            count = count - penalty
    bcc = (count/len(ind_const))
    return bcc

def bcc(dataset: BinaryLabelDataset, k:int = 5, threshold:float = 0.8, penalty:float=0):
    """ Compute Balanced Conditioned Consistency [#waller2025] for 'dataset'
    dataset : BinaryLabelDataset
        dataset for which BCC is computed
    k : int
        number of individuals considered for the k nearest neighbor algorithm used for proximity
    threshold : float
        threshold under which individual consistency scores are not considered to compute BCC
    penalty : foat
        potential penalty added to the BBC score, 0 for no penalty
    returns : float
        BCC score for 'dataset'
    """
    ind_const = individual_consistency_scores(dataset,k)
    #print("Individual consistency scores :"+str(ind_const))
    if penalty == 0 :
        bcc = bcc_from_ind_const(ind_const, threshold=threshold)
    else : 
        bcc = bcc_penalty_from_ind_const(ind_const, threshold=threshold, penalty=penalty)
    return bcc


def consistency(dataset: BinaryLabelDataset, k:int=5):
    """ Compute the Consistency [#zemel13] score of 'dataset' with knn as proximity measure
        For proximity measure : sensitive attribute is excluded, features are scaled, individuals aren't considered as one of their neighbors
    dataset : BinaryLabelDataset
        dataset for which BCC is computed
    k : int
        number of individuals considered for the k nearest neighbor algorithm used for proximity
    """
    ind_const = individual_consistency_scores(dataset,k=k)
    consistency = ind_const.mean()
    return consistency