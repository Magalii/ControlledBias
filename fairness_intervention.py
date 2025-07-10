import numpy as np
import pandas as pd
import pickle
import math

from sklearn.naive_bayes import CategoricalNB

from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import LFR
from aif360.metrics import BinaryLabelDatasetMetric

def apply_preproc(nbias_data_dict, preproc:str, path_start:str = None):
    if preproc == 'reweighting' :
        preproc_dict = nk_proc(nbias_data_dict,reweight)
        print("Reweighting was applied")
    elif preproc == 'LFR' :
        #data_train = fair.reweight(nk_folds_dict[0.3][0])
        preproc_dict = nk_proc(nbias_data_dict,learn_fair_representation)
        print("LFR was applied\n WARNING LFR has not been tuned properly and doesn't give usefull results.")
    elif preproc == 'massaging' :
        preproc_dict = nk_proc(nbias_data_dict,massage)
        print("Massaging was applied")
    else :
        print("WARNING Not a valid preproc name")
        preproc_dict = nbias_data_dict

    if path_start is not None :
        path = path_start + '_nkdatasets.pkl'
        with open(path,'wb') as file:
            pickle.dump(preproc_dict,file)

    return preproc_dict

def blind_features(data_orig: StandardDataset) :
    """ Fairness Through Unawareness, a.k.a Blinding
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

def massage(data_orig: StandardDataset, fav_one=True) :
    """ Apply Massaging on 'data_orig'
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
    NBobj = CategoricalNB()
    NBobj.fit(X_train, Y_train)
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
        Y_priv_prob = NBobj.predict_proba(X_priv)
        instance_priv = pd.DataFrame(data=Y_priv_prob[:,1], index=df_priv_pos.index)
        instance_priv.sort_values(by=0, ascending=True, inplace=True) #Sorting by prob values, name of column is 0
    if nbr_unpriv_neg > 0 :
        #Sort unpriv instances according to ranker probabilities
        X_unpriv = df_unpriv_neg[data_orig.feature_names].values.copy() #unpriv features
        Y_unpriv_prob = NBobj.predict_proba(X_unpriv)
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
