import numpy as np
import pandas as pd
import pickle

from aif360.datasets import StandardDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import sys 
sys.path.append('..')

import fairness_intervention as fair


###########################
### Division into folds ###
###########################

def save_split(dataset: StandardDataset, k: int, path_start: str) :
    """ Creates k partition of dataset and saves it to path_k_splits.pkl
        Returns None
    """
    split_data = dataset.split(k,shuffle=True)
    path = path_start+"_"+str(k)+"splits.pkl"
    with open(path,'wb') as file:
        pickle.dump(split_data,file)

def common_splits(dataset_dict, k: int, path_start: str= None) :
    """ Computes and returns a division of the datasets in 'dataset_dict' in 'k' partitions that is THE SAME for all dataset
        Partitions the datasets in 'dataset_dict' accordingly and saves on disk the partitions for each fold in a dictionary
        Saved dictionary has bias levels (float) as keys and list of the partitions for k fold as objects (list[StandardDataset])
    dataset_dict : Dictionary {float: StandardDataset}
        Dictionary of the different biased versions of the same dataset,
        with the original (unbiased) version at key '0'
    k : int
        number of partitions desired
    path_start: String
        if not None, save dataset dict on disk at address 'path_start' + function postfix
    Returns (Dictionary {float: list[StandardDataset]}, Dictionary {float: list[indexes]})
    -------
    Dictionary with bias level (float) as keys and list with a StandardDataset partition for each fold as objects,
    Dictionary with bias level (float) as keys and list of the folds indexes (list of strings) as objects
    """
    #Split first dataset in dict and save indexes of each partitions
    o = list(dataset_dict.keys())[0] #first key
    dataset_orig: StandardDataset = dataset_dict[o]
    split_orig = dataset_orig.split(k,shuffle=True)
    folds = {} #keys are fold numbers (int), objects are list of the folds indexes (list of strings)
    for i in range(k) :
        folds[i] = list(map(int,split_orig[i].instance_names)) #list of str
    
    #Apply the same slicing to all other datasets
    n_bias_splits = {}
    n_bias_splits[o] = split_orig
    for b in dataset_dict.keys() :
        dataset_list = [None] * k
        #dataset_indices = list(map(int,dataset_dict[b].instance_names))
        for i in range(k) :
            dataset_list[i] = get_subset(dataset_dict[b], split_orig[i])
        n_bias_splits[b] = dataset_list

    if path_start is not None :
        path = path_start + '_nbias_splits_datasets.pkl'
        with open(path,'wb') as file:
            pickle.dump(n_bias_splits,file)
        path = path_start + '_nbias_'+str(k)+'splits_foldsLists.pkl'
        with open(path,'wb') as file:
            pickle.dump(folds,file)
    
    return n_bias_splits, folds

def random_splits(dataset_dict, k: int, path_start: str= None) :
    """ Computes and returns a division of the datasets in 'dataset_dict' in 'k' partitions that is INDEPENDANT for each dataset
        Partitions are determined randomly and are different for each bias level
        They will all have the same size
        Saved dictionary has bias levels as keys (float) as keys and list of the partitions for k fold as objects (list[StandardDataset])
    k : int
        number of partitions desired
    dataset_dict : Dictionary {float: StandardDataset}
        Dictionary of the different biased versions of the same dataset,
    path_start: String
        if not None, save datasets dict on disk at address 'path_start' + function postfix
    Returns (Dictionary {float: list[StandardDataset]}, Dictionary {float: list[indexes]})
    -------
    Dictionary with bias level (float) as keys and list with a StandardDataset partition for each fold as objects
    """
    #folds = {} #keys are fold numbers (int), objects are list of the folds indexes (list of strings)
    n_bias_splits = {}
    for b in dataset_dict.keys() :
        #dataset_indices = list(map(int,dataset_dict[b].instance_names))
        n_bias_splits[b] = dataset_dict[b].split(k,shuffle=True)

    if path_start is not None :
        path = path_start + '_nbias_splits_datasets.pkl'
        with open(path,'wb') as file:
            pickle.dump(n_bias_splits,file)
    
    return n_bias_splits

def get_subset(dataset_orig: StandardDataset, dataset_smaller: StandardDataset) :
    """ Select the subset of instances of dataset_orig that are also in dataset_smaller
    Returns : StandardDataset
        Copy of dataset_orig containing only the instances that are also present in dataset_smaller
    """
    dataset_big_indices = list(map(int,dataset_orig.instance_names))
    dataset_smal_indices = list(map(int,dataset_smaller.instance_names))
    instance_names = set(dataset_big_indices).intersection(dataset_smal_indices) #only consider values present in both datasets
    size = len(instance_names)
    if size < len(dataset_big_indices) : #True if dataset_smaller has been undersampled
        #convert instance name to positional index in undersampled dataset
        to_keep = [0] * size
        p = 0
        for id in instance_names :
            to_keep[p] = dataset_big_indices.index(id)
            p+=1
        to_keep.sort() #Important to allow for correct comparison of dataset content by AIF360 library
        dataset_intersect = dataset_orig.subset(to_keep) #Subset considers index position, not instance names
        return dataset_intersect
    else :
        return dataset_orig
    
def merge_train(split_list: list[StandardDataset], fold:int, valid:bool, path_start: str = None) :
    """ Create a train set that is complimetary to test split = split_list[fold], and optional validation set=split_list[wrap(fold+1)] if valid==true
    #TODO split_list
    fold: int
        index of the dataset partition that will be used as test set, must be a valid index of 'split_list'
    valid:
        whether a validation set should be created (True) or not (False). If True, the validation set will correspond to split_list[wrap(fold+1)]
    path_start: String
        if not None, save dataset dict on disk at address 'path_start' + function postfix
    """
    k = len(split_list)
    df_list = [None] * k
    weights_list = []
    size = 0
    all_folds = range(k)
    train_folds = [x for x in all_folds if x!= fold and (not valid or x!=(fold+1)%5)]
    for i in train_folds :
        df_list[i] = (split_list[i]).convert_to_dataframe()[0]
        weights_list = weights_list + list(split_list[i].instance_weights)
        size += len(weights_list)
    merged_df = pd.concat(df_list)
    merged_df['instance_weights'] = weights_list
    merged_df.sort_index()
    merged_dataset = StandardDataset(merged_df,
                                 label_name=split_list[0].label_names[0],
                                 favorable_classes=[split_list[0].favorable_label],
                                 protected_attribute_names=split_list[0].protected_attribute_names,
                                 privileged_classes=split_list[0].privileged_protected_attributes,
                                 instance_weights_name='instance_weights',
                                 metadata=split_list[0].metadata)
    if path_start is not None :
        with open(path_start+"_train-test.pkl",'wb') as file:
            pickle.dump(merged_dataset,file)

    return merged_dataset

def nk_merge_train_deprecated(split_dict,path_start:str=None) :
    """
    Return : Dictionary {float : {int : {'train' : StandardDataset, 'test': StandardDataset}}}
    train_test_dict[bias][fold]: {'train': train set, 'test': test set}
    """
    keys = split_dict.keys()
    train_test_dict = {}
    k = len(split_dict[list(keys)[0]])
    for b in keys :
        folds = {}
        for i in range(k) :
            folds[i] = {'train': merge_train(split_dict[b],i), 'test': split_dict[b][i]}
        train_test_dict[b] = folds
        
    if path_start is not None :
        with open(path_start+ "_train-test-nk.pkl",'wb') as file:
            pickle.dump(train_test_dict,file)

    return train_test_dict

def nk_merge_train(split_dict, valid:bool, path_start:str=None) :
    """ Creates train set, test set (and validation set if valid=True) for each bias level from list of splits
    #TODO proper comment
    Return : Dictionary {float : {int : {'train' : StandardDataset, 'test': StandardDataset}}}
    train_test_dict[bias][fold]: {'train': train set, 'test': test set}
    """
    keys = split_dict.keys()
    train_test_dict = {}
    k = len(split_dict[list(keys)[0]])
    for b in keys :
        folds = {}
        for i in range(k) :
            if valid :
                folds[i] = {'train': merge_train(split_dict[b],i,valid=True), 'valid': split_dict[b][(i+1)%k], 'test': split_dict[b][i]}
            else :
                folds[i] = {'train': merge_train(split_dict[b],i,valid=False), 'test': split_dict[b][i]}
        train_test_dict[b] = folds
        
    if path_start is not None :
        with open(path_start+ "_train-test-nk.pkl",'wb') as file:
            pickle.dump(train_test_dict,file)

    return train_test_dict
    
def remove_test(data_orig: StandardDataset, data_test: StandardDataset) :
    """Returns a dataset containing the instances of data_orig that are not in data_test
        Returned dataset is meant for training with data_test as test set
        This function resets all instance weights to 1.
    """
    print("Warning : remove_test function resets all instance weights to 1.")
    df_orig = data_orig.convert_to_dataframe()[0]
    df_test = data_test.convert_to_dataframe()[0]
    df_train = df_orig.drop(df_test.index)
    data_train = StandardDataset(df_train,
                                 label_name=data_orig.label_names[0],
                                 favorable_classes=[data_orig.favorable_label],
                                 protected_attribute_names=data_orig.protected_attribute_names,
                                 privileged_classes=data_orig.privileged_protected_attributes,
                                 #categorical_features=[],
                                 metadata=data_orig.metadata)
    return data_train


######################
### Classification ###
######################

def single_classifier(algo: str, dataset_train: StandardDataset, blinding: bool, path_start: str = None) :
    """ Create a classifier trained with the instances of dataset_orig that are not in dataset_test
    Parameters
    ----------
    algo : String ('RF'|'tree'|'neural')
        Type of classifier that should be used
    dataset_train : StandardDataset
        Full dataset, containing both train and test instances
    blinding : Boolean
        Wether the sensitive attribute is used in training (False) or not (True)
        blinding = True is equivalent to applying Fairness Through Unawareness
    path_start: String
        if not None, save classifiers dict on disk at address 'path_start' + function postfix
    Returns
    -------
    Classifier
        Model trained with dataset_train
    """
    if blinding :
        X_train = fair.blind_features(dataset_train)
    else :
        X_train = dataset_train.features
    Y_train = dataset_train.labels.ravel()
    sample_weight = dataset_train.instance_weights
    
    if algo == 'neural' :
        classifier = MLPClassifier(max_iter=1500)
        classifier.fit(X_train, Y_train)
    else :
        if algo == 'tree' :
            classifier = DecisionTreeClassifier(max_depth=6)
        elif algo == 'RF' :
            classifier = RandomForestClassifier(max_depth=6, min_samples_split=10, min_samples_leaf=10)
        
        else :
            print("WARNING : No valid classifier type was given. It must be a string among 'RF', tree' and 'neural' ")
        classifier.fit(X_train, Y_train, sample_weight)

    if path_start is not None :
        path = path_start+"_.pkl"
        with open(path,"wb") as file:
            pickle.dump(classifier,file)

    return classifier

def classifier_kfold(classifier: str, fold_dict, blinding: bool, path_start: str = None) :
    """ Create a classifier for each fold in split_list, trained with the instances of dataset_orig that are not in the corresponding dataset in split_list
    Parameters
    ----------
    algo : String ('RF'|'tree'|'neural')
        Type of classifier that will be trained with each fold in split_list
    blinding : Boolean
        Wether the sensitive attribute is used in training (False) or not (True)
    fold_dict : Dictionary {int: {'train': StandardDataset, 'test: StandardDataset}}
        Dictionary of the train and test sets for each fold (only train is used)
    path_start: String
        if not None, save dataset dict on disk at address 'path_start' + function postfix
    Returns
    -------
    Dictionary
        Dictionary containing the classifier for each fold
        k_models[i] holds the model trained with the dataset held at fold_dict[i]['train']
    """
    k_models = {} #key: fold number, object: sklearn classifier for that fold
    dict_keys = fold_dict.keys()
    k = len(list(dict_keys))
    for i in dict_keys :
        #dataset_train = merge_train(split_list,i)
        #print("classifier_kfold")
        model = single_classifier(classifier, fold_dict[i]['train'], blinding=blinding, path_start=path_start)
        k_models[i] = model

    if path_start is not None :
        path = path_start+"_"+str(k)+"folds.pkl"
        with open(path,"wb") as file:
            pickle.dump(k_models,file)

    return k_models
    
def classifier_nbias(classifier: str, nk_dataset_dict, blinding: bool, path_start: str = None) :
    """ Create a classifier for each bias and each fold in fold accounted for in nk_dataset_dict
    #TODO update comment
    Parameters
    ----------
    algo : String ('RF'|'tree'|'neural')
        Type of classifier that will be trained with each bias level and each fold in dataset_dict
    dataset_dict : Dictionary {float : {int : {'train' : StandardDataset, 'test': StandardDataset}}}
        Nested dictionaries holding train and test sets for each bias and each fold (Only train set is used)
        nk_dataset_dict[bias][fold]: {'train': train set, 'test': test set}
    blinding : Boolean
        Wether the sensitive attribute is used in training (False) or not (True)
    path_start: String
        if not None, save dataset dict on disk at address 'path_start' + function postfix
    Returns
    -------
    Dictionary of dictionary
        Dictionary containing all classifiers for each bias and fold present in dataset_dict
        all_models[bias][fold] holds the model trained with the partitions complementary to dataset_dict[bias][fold], which contains the instances provisioned for the test set
    """
    all_models = {}
    bias = nk_dataset_dict.keys()
    for b in bias :
        #print("training bias "+str(b))
        k_models = classifier_kfold(classifier, nk_dataset_dict[b], blinding=blinding)
        all_models[b] = k_models
        
    if path_start is not None :
        path = path_start+"_all.pkl"
        with open(path,"wb") as file:
            pickle.dump(all_models,file)

    return all_models #all_models[bias][fold] = single classifier

###################
### Predictions ###
###################

def single_prediction(classifier, test_dataset: StandardDataset, pred_type:str, blinding: bool):
    """ Compute the prediction of 'classifier' for the given 'test_dataset
    pred_type : str
        The type of prediction : 'labels' for binary labels and 'scores' for classification probabilities
    blinding : Boolean
        Wether the sensitive attribute has been used to train 'classifier' (True) or not (False)
    Returns
    -------
    StandardDataset
        A dataset with the features in 'test_dataset' and the labels predicted by 'classifier'
    """
    if blinding :
        pred_features = fair.blind_features(test_dataset)
    else :
        pred_features = test_dataset.features
    if pred_type == 'labels':
        pred_array = classifier.predict(pred_features)
    elif pred_type == 'scores':
        pred_array = classifier.predict_proba(pred_features)[:,1].reshape(-1,1)
    else :
        print("ERROR: Not a valid prediction type. Must be 'labels' or 'scores'")
    pred_dataset = pred_to_dataset(test_dataset,pred_array,pred_type)
    return pred_dataset

def prediction_kfold(classifier_dict, split_list: list[StandardDataset], blinding: bool, pred_type:str, path_start: str = None):
    """ Compute the predictions for all k folds given in split_list
    #TODO make sure comment is up to date
    classifier_dict : Dictionary with fold number as keys and corresponding classifier as object
        Dict. of classifiers trained on different folds of the same dataset
    split_list :
        List of the different partitions of the dataset, whith split_list[i] being the test set for classifier_dict[i]
    blinding : Boolean
        Wether the sensitive attribute has been used to train 'classifier' (True) or not (False)
    pred_type : str
        The type of prediction : 'labels' for binary labels and 'scores' for classification probabilities
    path_start: String
        if not None, save predictions on on disk at address 'path_start' + function postfix
    """
    k_pred = {} #key: fold number (= test set number), object: prediction for test set
    k = len(split_list)
    for i in range(k) :
        k_pred[i] = single_prediction(classifier_dict[i], split_list[i], pred_type, blinding=blinding)

    if path_start is not None :
        path = path_start+"_pred_"+str(k)+"fold.pkl"
        with open(path,"wb") as file:
            pickle.dump(k_pred,file)
    
    return k_pred
#Compute the prediction for classifiers with different bias levels and fold found in classifier_dict and given the corresponding test set
    
def prediction_nbias(classifier_dict, nk_train_splits, set:str, pred_type:str, blinding: bool, biased_test = False, path_start: str = None) :
    """ Compute the prediction for classifiers with different bias levels and fold found in classifier_dict and given the corresponding test set
    classifier_dict : Dictionary
        Nested dictionaries holding classifiers, where classifier_dict[b][f] holds the model trained on fold nbr 'f' of dataset with bias level 'b'
    nk_train_splits : Dictionary {float : {int : {'train' : StandardDataset, ('valid' : StandardDataset,) 'test': StandardDataset}}}
        Nested dictionaries holding the train, test and optionnaly validation set for b bias and f folds
        where nk_train_splits[bias][fold] = {'train': train set, ('valid' : validation set,)'test': test set}
    set: str
        One of 'valid' or 'test'. Indicates on which set the predictions should be made
    pred_type : str
        The type of prediction : 'labels' for binary labels and 'scores' for classification probabilities
    blinding : Boolean
        Wether the sensitive attribute has been used to train 'classifier' (True) or not (False) [needed to have same features in test and train]
    biased_test : Boolean
        Wether the test set has bias level 'b' (True) or is a subset of the original (considered unbiased) dataset (False)
        (Doesn't make a difference for label bias)
    path_start: String
        if not None, save predictions dict on disk at address 'path_start' + function postfix
    Returns
    -------
    Dictionary {'pred': {float: {int: StandardDataset}, 'orig': {float: {int: StandardDataset}}
        Nested dictionaries where n_pred['pred'][b][f] holds the prediction given by model trained on fold nbr 'f' of dataset with bias level 'b' and n_pred['orig'][b][f] holds the input data given to make the predictions
    """
    n_pred = {}
    for b in classifier_dict.keys() :
        n_pred[b] = {} #key: fold number (= test set number), object: prediction for test set
        for i in nk_train_splits[0].keys() :
            if biased_test : #test set has the same level of bias as training set
                test_fold = nk_train_splits[b][i][set]
            else : #test set is a subset of the original (considered unbiased) dataset
                test_fold = nk_train_splits[0][i][set]
            n_pred[b][i] = single_prediction(classifier_dict[b][i], test_fold, pred_type, blinding=blinding)

    dict_pred = {'pred':n_pred, 'orig': nk_train_splits}
    #Keeping both the prediction and the original input allows future analyzes of results without having to run the experiment again

    if path_start is not None :
        path = path_start+"_pred_all.pkl"
        with open(path,"wb") as file:
            pickle.dump(dict_pred,file)

    return dict_pred


def pred_to_dataset(dataset_orig : StandardDataset, predictions : np.ndarray, pred_type:str) :
    """ Create a StandardDataset with the predictions (labels or score) and the instances for which they have been predicted
    dataset_orig : StandardDataset
        Dataset containing the instances for which predictions (labels or score) were predicted
    predictions : Numpy ndarray
        1-d array containing predictions, the predictions order must correspond to that of the instances in dataset_orig
    pred_type : str
        The type of prediction : 'labels' for binary labels and 'scores' for classification probabilities
    Returns
    -------
    StandardDataset
        Dataset containing the instances and info in dataset_orig with the labels in predictions
    """
    dataset_pred = dataset_orig.copy()
    if pred_type == 'labels':
        dataset_pred.labels = predictions
    elif pred_type == 'scores':
        dataset_pred.scores = predictions
        size = len(dataset_pred.instance_names)
        pred_labels = np.zeros((size,1))
        for i in range(size):
            label = predictions[i][0] > 0.5
            pred_labels[i] = [label]
        dataset_pred.labels = pred_labels
    else :
        print("ERROR: Not a valid prediction type. Must be 'labels' or 'scores'")
    dataset_pred.validate_dataset()
    return dataset_pred
