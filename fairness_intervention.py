"""
    Code to applay bias mitigation methods
"""

import numpy as np
import pandas as pd
import pickle
import math
import gc

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB

from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import LFR
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.metrics import BinaryLabelDatasetMetric

#################
# Preprocessing #
#################

def apply_preproc(nk_datasets, preproc:str, path_start:str = None):
    """ Apply the chosen preprocessing bias mitigation methods all the datasets in 'nk_datasets'
    nk_datasets : Dictionary {float : {int : {'train': train set, ('valid': validation set,) 'test': test set}}}
        Dictionary containing all the splitted datasets the preprocessing should be applied to (validation set is not used and will be discarded)
        nk_datasets[bias][fold]: {'train': train set, ('valid': validation set,) 'test': test set}
     preproc: str
        The chosen postprossing, must be one of the following :
        'reweighting',
        'LFR' for Learning Fair Representation,
        'massaging' for massaging using a bayesian ranker,
        'massagingRF' for massaging using a random forest ranker,
    path_start : String
        Path at which the dictionary will be saved, if None dictionary is not saved

    Returns: Dictionary {float : {int : {'train' : StandardDataset, 'test': StandardDataset}}}
        Dictionary that contains all the datasets with mitigated train and test sets
        preproc_dict[bias][fold]: {'train': train set, 'test': test set}
    """
    if preproc == 'reweighting' :
        preproc_dict = nk_proc(nk_datasets,reweight)
        print("Reweighting was applied")
    elif preproc == 'LFR' :
        #data_train = fair.reweight(nk_folds_dict[0.3][0])
        preproc_dict = nk_proc(nk_datasets,learn_fair_representation)
        print("LFR was applied\n WARNING LFR has not been tuned properly and doesn't give usefull results.")
    elif preproc == 'massaging' :
        preproc_dict = nk_proc(nk_datasets,massage)
        print("Massaging was applied")
    elif preproc == 'massagingRF' :
        preproc_dict = nk_proc(nk_datasets,massage_RF)
        print("MassagingRF was applied")
    else :
        print("WARNING Not a valid preproc name")
        preproc_dict = nk_datasets

    if path_start is not None :
        path = path_start + '_nkdatasets.pkl'
        with open(path,'wb') as file:
            pickle.dump(preproc_dict,file)

    return preproc_dict

def blind_features(data_orig: StandardDataset) :
    """ Fairness Through Unawareness, a.k.a Blinding
    data_orig: StandardDataset
        Dataset on which mitigation is applied
    Returns : np.array
        features of StandardDataset 'data_orig' without the protected attribute
        (if more than one sensitive attribute is listed for data_orig, only the first one of the list 'protected_attribute_names' is remvoved)
    """
    id_sens = data_orig.feature_names.index(data_orig.protected_attribute_names[0])
    features_blind = np.delete(data_orig.features, id_sens, axis=1)
    return features_blind

def reweight(data_orig: StandardDataset, fav_one=True) :
    """ Reweighting of 'data_orig'
    Division by 0 warning comes from RW.fit and leads to an inf value within fit function. Reweighing is still operational.
    data_orig: StandardDataset
        Dataset on which mitigation is applied
    fav_one : Boolean, optional
        Wether the favorable value for the sensitive attribute is 1 (True) or 0 (False)
    Returns : BinaryLabelDataset
      a copy of 'data_orig' with transformed instance weights
    """
    sens_attr = data_orig.protected_attribute_names[0]
    priv = [{sens_attr : int(fav_one)}]
    unpriv = [{sens_attr : 1-int(fav_one)}]
    RWobj = Reweighing(unprivileged_groups=unpriv, privileged_groups=priv)
    RWobj.fit(data_orig)
    dataset_transf = RWobj.transform(data_orig)
    return dataset_transf

def learn_fair_representation(data_orig: StandardDataset, fav_one=True) :
    """ Apply "Learning fair representation" on 'data_orig'
    data_orig: StandardDataset
        Dataset on which mitigation is applied
    fav_one : Boolean, optional
        Wether the favorable value for the sensitive attribute is 1 (True) or 0 (False)
    Returns : BinaryLabelDataset
      a copy of 'data_orig' with new labels
    """
    sens_attr = data_orig.protected_attribute_names[0]
    priv = [{sens_attr : int(fav_one)}]
    unpriv = [{sens_attr : 1-int(fav_one)}]
    LFRobj = LFR(unprivileged_groups=unpriv, privileged_groups=priv,
                 k=10, Ax=0.5, Ay = 1.0, Az=2, 
                 verbose=0,
                 seed = 4242)
    LFRobj.fit(data_orig)
    dataset_transf = LFRobj.transform(data_orig)
    return dataset_transf

def massage(data_orig: StandardDataset, rank_algo:str = 'NaiveBayes', fav_one=True) :
    """ Apply Massaging on 'data_orig'
        data_orig : StandardDataset
            dataset on which massaging is applied
        rank_algo : String
            Ranking algorithm used to determine which labels should be changed
            'NaiveBayes' for naive bayes algo (sklearn CategoricalNB)
            'RF' for random forest (sklearn RandomForestClassifier)
            No significant difference could be observed between the use of Naive Bayes and Random Forest
        path_start : String
            Path at which the dictionary will be saved, if None dictionary is not saved
        fav_one : Boolean, optional
            Wether the favorable value for the sensitive attribute is 1 (True) or 0 (False)
        Returns : BinaryLabelDataset
        a copy of 'data_orig' with some label changes
        WARNING : instance_weight information will be lost
    """
    label = data_orig.label_names[0]
    sens_attr = data_orig.protected_attribute_names[0]
    priv = int(fav_one)
    unpriv = 1-int(fav_one)
    #Learn ranker
    #TODO make sure prob are given for label 1
    X_train = data_orig.features
    Y_train = data_orig.labels.ravel()
    if rank_algo == 'NaiveBayes' :
        ranker = CategoricalNB()
    elif rank_algo == 'RF' :
        ranker = RandomForestClassifier(max_depth=6, min_samples_split=10, min_samples_leaf=10)
    else :
        print("WARNING Not a valid ranking algorithm. Naive Bayes is used")
        ranker = CategoricalNB()
    ranker.fit(X_train, Y_train)
    #Retrieve instances to be ranked (priv-pos and unpriv-neg)
    df_orig = data_orig.convert_to_dataframe()[0]
    df_priv = df_orig.loc[(df_orig[sens_attr]==priv)]
    df_priv_pos = df_priv.loc[(df_priv[label]==data_orig.favorable_label)]
    df_unpriv = df_orig.loc[(df_orig[sens_attr]==unpriv)]
    df_unpriv_neg = df_unpriv.loc[(df_unpriv[label]==data_orig.unfavorable_label)]
    nbr_priv_pos = len(df_priv_pos)
    nbr_unpriv_neg = len(df_unpriv_neg)
    if nbr_priv_pos > 0 :
        #Sort priv instances according to ranker probabilities
        X_priv = df_priv_pos[data_orig.feature_names].values.copy() #priv features
        Y_priv_prob = ranker.predict_proba(X_priv)
        instance_priv = pd.DataFrame(data=Y_priv_prob[:,1], index=df_priv_pos.index)
        instance_priv.sort_values(by=0, ascending=True, inplace=True) #Sorting by prob values, name of column is 0
    if nbr_unpriv_neg > 0 :
        #Sort unpriv instances according to ranker probabilities
        X_unpriv = df_unpriv_neg[data_orig.feature_names].values.copy() #unpriv features
        Y_unpriv_prob = ranker.predict_proba(X_unpriv)
        instance_unpriv = pd.DataFrame(data=Y_unpriv_prob[:,1], index=df_unpriv_neg.index)
        instance_unpriv.sort_values(by=0, ascending=False, inplace=True) #Sorting by prob values, name of column is 0  
    #Get number M of labels to change in each group
    metrics_obj = BinaryLabelDatasetMetric(data_orig, unprivileged_groups=[{sens_attr : unpriv}], privileged_groups=[{sens_attr : priv}])
    disc = -metrics_obj.statistical_parity_difference()
    if math.isnan(disc) :
        M = 0
        print("Massaging error : measured discimination is NaN. Dataset could not be massaged.")
    else :
        M = math.ceil((disc*len(df_priv)*len(df_unpriv))/len(df_orig))
    #Select the candidates for demotion or promotion and relabel
    if nbr_priv_pos > 0 :
        id_dem = instance_priv.index.to_list()[:M]
        df_orig.loc[id_dem,label] = 0
    if nbr_unpriv_neg > 0 :
        id_prom = instance_unpriv.index.to_list()[:M]
        df_orig.loc[id_prom,label] = 1

    massaged_dataset = StandardDataset(df = df_orig,
                                        label_name=label,
                                        protected_attribute_names=[sens_attr],
                                        favorable_classes= [data_orig.favorable_label],
                                        privileged_classes=[[1.]],
                                        categorical_features=[],
                                        metadata=data_orig.metadata)

    return massaged_dataset

def massage_RF(data_orig: StandardDataset) :
    """ Apply Massaging on 'data_orig' using Random Forest as the ranking algorithm
        data_orig : StandardDataset
            dataset on which massaging is applied
        Returns : BinaryLabelDataset
        a copy of 'data_orig' with some label changes
        WARNING : instance_weight information will be lost
    """
    return massage(data_orig,'RF')
    


##################
# Postprocessing #
##################

def apply_postproc_orig(postproc:str, dataset_true: StandardDataset, dataset_pred: StandardDataset, path_start:str = None):
    """ Apply the chosen postprocessing bias mitigation methods on the given predictions
     postproc: str
        The chosen postprossing, must be one of the following :
        'eqOddsProc' for Equalized Odds Postrocessing
        'calEqOddsProc' for Calibrated Equalized Odds Postrocessing
        'ROC-SPD' for Reject Option with Statistical parity difference
        'ROC-EOD' for Reject Option with Equal opportunity difference
        'ROC-AOD' for Reject Option with Average odds difference
    dataset_true: StandardDataset
        Dataset containing labels considered as true
    dataset_pred: StandardDataset
        Dataset containing predicted labels
    path_start : String
        Path at which the dictionary will be saved, if None dictionary is not saved
    Returns: StandardDataset
        Dataset containing the transformed labels
    """
    if postproc == 'eqOddsProc' :
        dataset_transf = eq_odds_postprocess(dataset_true, dataset_pred)
        #print("Equalized Odds Postrocessing was applied")
    elif postproc == 'calEqOddsProc' :
        dataset_transf = calibrated_eq_odds_postprocess(dataset_true, dataset_pred)
        #print("Calibrated Equalized Odds Postrocessing was applied")
    elif postproc == 'ROC-SPD' :
        dataset_transf = reject_option(dataset_true, dataset_pred,'Statistical parity difference')
        #print("Reject Option with Statistical parity difference was applied")
    elif postproc == 'ROC-EOD' :
        dataset_transf = reject_option(dataset_true, dataset_pred,'Equal opportunity difference')
        #print("Reject Option with Equal opportunity difference was applied")
    elif postproc == 'ROC-AOD' :
        dataset_transf = reject_option(dataset_true, dataset_pred,'Average odds difference')
        #print("Reject Option with Average odds difference was applied")
    else :
        print("WARNING Not a valid postproc name\n NO POST PROCESSING APPLIED")
        dataset_transf = dataset_pred

    if path_start is not None :
        path = path_start + '_' + postproc + '.pkl'
        with open(path,'wb') as file:
            pickle.dump(dataset_transf,file)
   
    return dataset_transf

def apply_postproc(postproc:str, dataset_orig: StandardDataset, valid_pred: StandardDataset, test_pred: StandardDataset, path_start:str = None):
    """ Apply the chosen postprocessing bias mitigation methods, fitted with valid_pred, on test_pred
     postproc: str
        The chosen postprossing, must be one of the following :
        'eqOddsProc' for Equalized Odds Postrocessing
        'calEqOddsProc' for Calibrated Equalized Odds Postrocessing
        'ROC-SPD' for Reject Option with Statistical parity difference
        'ROC-EOD' for Reject Option with Equal opportunity difference
        'ROC-AOD' for Reject Option with Average odds difference
    dataset_orig: {'train' : StandardDataset, 'valid':StandardDataset, 'test': StandardDataset}
        Splits of datasets with prior labels
    valid_pred: StandardDataset
        Validation dataset with predicted classification probabilities (scores). Must be the same instances as in dataset_orig['valid']
        Used to fit the post-processing objects
    test_pred: StandardDataset
        Test dataset with predicted classification probabilities (scores). Must be the same instances as in dataset_orig['test']
        Set that will undergo post-processing
    path_start : String
        Path at which the dictionary will be saved, if None dictionary is not saved

    Returns: StandardDataset
        Dataset containing the transformed labels
    """
    if postproc == 'eqOddsProc' :
        dataset_transf = eq_odds_postprocess(dataset_orig, valid_pred, test_pred)
        #print("Equalized Odds Postrocessing was applied")
    elif postproc == 'calEqOddsProc' :
        dataset_transf = calibrated_eq_odds_postprocess(dataset_orig, valid_pred, test_pred)
        #print("Calibrated Equalized Odds Postrocessing was applied")
    elif postproc == 'ROC-SPD' :
        dataset_transf = reject_option(dataset_orig, valid_pred, test_pred,'Statistical parity difference')
        #print("Reject Option with Statistical parity difference was applied")
    elif postproc == 'ROC-EOD' :
        dataset_transf = reject_option(dataset_orig, valid_pred, test_pred,'Equal opportunity difference')
        #print("Reject Option with Equal opportunity difference was applied")
    elif postproc == 'ROC-AOD' :
        dataset_transf = reject_option(dataset_orig, valid_pred, test_pred,'Average odds difference')
        #print("Reject Option with Average odds difference was applied")
    else :
        print("WARNING Not a valid postproc name\n NO POST PROCESSING APPLIED")
        dataset_transf = None

    if path_start is not None :
        path = path_start + '_' + postproc + '.pkl'
        with open(path,'wb') as file:
            pickle.dump(dataset_transf,file)
   
    return dataset_transf

def n_postproc_orig(postproc:str, n_dataset_true, n_dataset_pred, biased_truth:bool = True, path_start:str = None):
    """ Apply the given post-processing method for every bias level and every fold found in n_dataset_pred
        n_dataset_true and n_dataset_pred must contain the same individuals
    postproc: str
        The chosen postprossing, must be one of the following :
        'eqOddsProc' for Equalized Odds Postrocessing
        'calEqOddsProc' for Calibrated Equalized Odds Postrocessing
        'ROC-SPD' for Reject Option with Statistical parity difference
        'ROC-EOD' for Reject Option with Equal opportunity difference
        'ROC-AOD' for Reject Option with Average odds difference
    n_dataset_true: {float : {int : {'train' : StandardDataset, 'test': StandardDataset}}}
        Nested dictionaries holding datasets with labels considered as true. n_dataset_true[b][f] holds the dataset with bias level 'b' and fold nbr 'f'
        (named nk_train_splits in most other functions)
    n_dataset_pred: Dictionary {float: {int: StandardDataset}}
        Nested dictionaries holding datasets with predicted labels. n_dataset_pred[b][f] holds the predictions for bias level 'b' and fold nbr 'f'
    biased_truth:bool
        Whether the labels considered as ground truth for debiasing is that of the biased train set (True) or an unbiased version (False)
    path_start : String
        Path at which the dictionary will be saved, if None dictionary is not saved

    Returns: Dictionary {float: {int: StandardDataset}}
        Nested dictionaries where n_transf[b][f] holds the transformed predictions for fold nbr 'f' and bias level 'b'
    """
    n_transf = {}
    for b in n_dataset_pred.keys() :
        #print("Start of postprocessing method "+postproc+ " for bias level "+str(b))
        n_transf[b] = {}
        for f in n_dataset_pred[b].keys():
            if biased_truth :
                #dataset_true = mt.get_subset(n_biased_dataset[b],n_dataset_pred[b][f])
                dataset_true = n_dataset_true[b][f]['test']
            else : # Use unbiased version as ground truth for debiasing
                dataset_true = n_dataset_true[0][f]['test']
                #dataset_true = mt.get_subset(n_biased_dataset[0],n_dataset_pred[b][f])
            #print("dataset_true :\n"+str(dataset_true.instance_names))
            #print("dataset_pred :\n"+str(n_dataset_pred[b][f].instance_names))
            try :
                n_transf[b][f] = apply_postproc(postproc, dataset_true, n_dataset_pred[b][f])
            except ValueError as err:
                #print("ERROR " + postproc + " not applied for bias level "+str(b)+" and fold "+str(f))
                #print("ValueError: {}".format(err))
                n_transf[b][f] = None
            except IndexError as err:
                #print("ERROR " + postproc + " not applied for bias level "+str(b)+" and fold "+str(f))
                #print("IndexError: {}".format(err))
                n_transf[b][f] = None
        gc.collect()
    print("Postprocessing method "+postproc+" was applied")

    if path_start is not None :
        path = path_start + '_n.pkl'
        with open(path,'wb') as file:
            pickle.dump(n_transf,file)

    return n_transf

def n_postproc(postproc:str, n_valid_pred, n_test_pred, biased_valid:bool, path_start:str = None, queue=None):
    """ Apply the given post-processing method to for every bias level and every fold found in n_test_pred
    postproc: str
        The chosen postprossing, must be one of the following :
        'eqOddsProc' for Equalized Odds Postrocessing
        'calEqOddsProc' for Calibrated Equalized Odds Postrocessing
        'ROC-SPD' for Reject Option with Statistical parity difference
        'ROC-EOD' for Reject Option with Equal opportunity difference
        'ROC-AOD' for Reject Option with Average odds difference
    nk_train_split: {float: int: {'train' : StandardDataset, 'valid':StandardDataset, 'test': StandardDataset}}}
        Nested dictionaries holding datasets splits of datasets with prior labels. nk_train_split[b][f] holds the dataset with bias level 'b' and fold nbr 'f'
    n_valid_pred: Dictionary {float: {int: StandardDataset}}
        Nested dictionaries holding validation datasets with predicted classification probabilities (scores). n_valid_pred[b][f] holds the predictions for bias level 'b' and fold nbr 'f'
        Used to fit the post-processing objects
    n_test_pred: Dictionary {float: {int: StandardDataset}}
        Nested dictionaries holding test datasets with predicted classification probabilities (scores). n_test_pred[b][f] holds the predictions for bias level 'b' and fold nbr 'f'
        Set that will undergo post-processing
    path_start : String
        Path at which the dictionary will be saved, if None dictionary is not saved

    Returns: Dictionary {float: {int: StandardDataset}}
        Nested dictionaries where n_transf[b][f] holds the transformed predictions for fold nbr 'f' and bias level 'b'
    """
    if n_valid_pred['orig'] != n_test_pred['orig'] : # Orig contains the whole datasets and its splits, including valid and test
        print("ERROR The original dataset and it splits are not the same object for n_valid_pred and n_test_pred (n_valid_pred['orig'] != n_test_pred['orig'])")
        exit()
    nk_train_split = n_valid_pred['orig']
    n_transf = {}
    for b in nk_train_split.keys() :
        #print("Start of postprocessing method "+postproc+ " for bias level "+str(b))
        n_transf[b] = {}
        for f in nk_train_split[b].keys():
            try :
                if biased_valid :
                    n_transf[b][f] = apply_postproc(postproc, nk_train_split[b][f], n_valid_pred['pred'][b][f], n_test_pred['pred'][b][f])
                else :
                    n_transf[b][f] = apply_postproc(postproc, nk_train_split[0][f], n_valid_pred['pred'][b][f], n_test_pred['pred'][b][f])
            except (ValueError, IndexError) as err:
                #print("ERROR " + postproc + " not applied for bias level "+str(b)+" and fold "+str(f))
                #print("ValueError: {}".format(err))
                n_transf[b][f] = None
            except Warning as warn :
                #print("Warning: {}".format(warn))
                pass
            #print("bias "+str(b)+"fold "+str(f))
            #print(n_transf[b][f])
        gc.collect()
    print("Postprocessing method "+postproc+" was applied")

    transf_dict = {'pred': n_transf, 'orig': nk_train_split}

    if path_start is not None :
        path = path_start + '_n.pkl'
        with open(path,'wb') as file:
            pickle.dump(transf_dict,file)
    if queue is not None :
        queue.put(transf_dict)

    return transf_dict

def eq_odds_postprocess(dataset_orig, valid_pred: StandardDataset, test_pred: StandardDataset, fav_one=True) :
    """ Apply Equalized Odds Postrocessing, fitted with valid_pred, on test_pred
    dataset_orig: {'train' : StandardDataset, 'valid':StandardDataset, 'test': StandardDataset}
        Dataset with prior labels split into train, valid and test sets
    valid_pred: StandardDataset
        Validation dataset with predicted classification probabilities (scores). Must be the same instances as in dataset_orig['valid']
        Used to fit the post-processing objects
    test_pred: StandardDataset
        Test dataset with predicted classification probabilities (scores). Must be the same instances as in dataset_orig['test']
        Set that will undergo post-processing
    fav_one : Boolean, optional
            Wether the favorable value for the sensitive attribute is 1 (True) or 0 (False)

    Returns: StandardDataset
        Dataset containing transformed labels
    Raises ValueError if dataset_true and dataset_pred don't contain the same instances
    """
    sens_attr = test_pred.protected_attribute_names[0]
    priv = [{sens_attr : int(fav_one)}]
    unpriv = [{sens_attr : 1-int(fav_one)}]
    EOPobj = EqOddsPostprocessing(unprivileged_groups=unpriv, privileged_groups=priv, seed=118712)
    EOPobj.fit(dataset_orig['valid'],valid_pred) #Fit EOP with validation set
    dataset_transf = EOPobj.predict(test_pred) #Apply EOP on test predictions
        
    return dataset_transf

def calibrated_eq_odds_postprocess(dataset_orig, valid_pred: StandardDataset, test_pred: StandardDataset, fav_one=True) :
    """ Apply Calibrated Equalized Odds Postrocessing, fitted with valid_pred, on test_pred
    dataset_orig: {'train' : StandardDataset, 'valid':StandardDataset, 'test': StandardDataset}
        Dataset with prior labels split into train, valid and test sets
    valid_pred: StandardDataset
        Validation dataset with predicted classification probabilities (scores). Must be the same instances as in dataset_orig['valid']
        Used to fit the post-processing objects
    test_pred: StandardDataset
        Test dataset with predicted classification probabilities (scores). Must be the same instances as in dataset_orig['test']
        Set that will undergo post-processing
    fav_one : Boolean, optional
        Wether the favorable value for the sensitive attribute is 1 (True) or 0 (False)

    Returns: StandardDataset
        Dataset containing transformed labels
    """
    sens_attr = test_pred.protected_attribute_names[0]
    priv = [{sens_attr : int(fav_one)}]
    unpriv = [{sens_attr : 1-int(fav_one)}]
    CEOobj = CalibratedEqOddsPostprocessing(unprivileged_groups=unpriv, privileged_groups=priv, cost_constraint='weighted', seed=118712)
    CEOobj.fit(dataset_orig['valid'],valid_pred) #Fit CEO with validation set
    dataset_transf = CEOobj.predict(test_pred) #Apply CEO on test predictions
    return dataset_transf

def reject_option(dataset_orig, valid_pred: StandardDataset, test_pred: StandardDataset, optimisation_metric: str, fav_one=True) :
    """ Apply Reject Option Classification, fitted with valid_pred, on test_pred
    dataset_orig: {'train' : StandardDataset, 'valid':StandardDataset, 'test': StandardDataset}
        Dataset with prior labels split into train, valid and test sets
    valid_pred: StandardDataset
        Validation dataset with predicted classification probabilities (scores). Must be the same instances as in dataset_orig['valid']
        Used to fit the post-processing objects
    test_pred: StandardDataset
        Test dataset with predicted classification probabilities (scores). Must be the same instances as in dataset_orig['test']
        Set that will undergo post-processing
    optimisation_metric: str
        Name of the metric to optimize, must be “Statistical parity difference”, “Average odds difference” or “Equal opportunity difference”
    fav_one : Boolean, optional
        Wether the favorable value for the sensitive attribute is 1 (True) or 0 (False)

    Returns: StandardDataset
        Dataset containing transformed labels
    """
    sens_attr = test_pred.protected_attribute_names[0]
    priv = [{sens_attr : int(fav_one)}]
    unpriv = [{sens_attr : 1-int(fav_one)}]
    ROCobj = RejectOptionClassification(unprivileged_groups=unpriv, privileged_groups=priv, metric_name=optimisation_metric)
    #print("dataset_orig['valid']")
    #print(dataset_orig['valid'].scores)
    #print("valid_pred")
    #print(valid_pred.scores)
    ROCobj.fit(dataset_orig['valid'],valid_pred) #Fit ROC with validation set
    dataset_transf = ROCobj.predict(test_pred) #Apply ROC on test predictions
    return dataset_transf



###################
# Other functions #
###################


def n_proc(data_dict,proc) :
    """ Apply procedure 'proc' on dictionary such that keys in data_dict are bias levels
        Returns : Dictionary {float : any}
        new_dict[bias] = proc(data_dict[b])
    """
    new_dict = {}
    for b in data_dict.keys() :
        new_dict[b] = {}
        new_dict[b] = proc(data_dict[b])
    return new_dict

def nk_proc(data_dict,proc) :
    """ Apply procedure 'proc' on train and test datasets in dictionary data_dict[bias][fold] = {'train': train set, 'test': test set}
        Returns : Dictionary {float : {int : {'train' : StandardDataset, 'test': StandardDataset}}}
        Dictionary with same keys : new_dict[bias][fold]: {'train': train set, 'test': test set}
    """
    new_dict = {}
    for b in data_dict.keys() :
        new_dict[b] = {}
        for f in range(len(data_dict[0])) :
            new_dict[b][f] = {'train': proc(data_dict[b][f]['train']), 'test': proc(data_dict[b][f]['test'])}
    return new_dict
