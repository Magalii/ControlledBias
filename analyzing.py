import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import numpy as np
import pickle
import math
import gc
import sys
sys.path.append('Code/')
sys.path.append('Code/parent_aif360')

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric #Metrics about 1 dataset
from aif360.metrics.classification_metric import ClassificationMetric #Metrics about 2 datasets
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler

import model_training as mt
import fairness_intervention as fair

def dataset_info(BLDataset: BinaryLabelDataset, fav_one=True) :
    sens_attr = BLDataset.protected_attribute_names[0]
    priv = [{sens_attr : int(fav_one)}]
    unpriv = [{sens_attr : 1-int(fav_one)}]
    metrics_obj = BinaryLabelDatasetMetric(BLDataset, unprivileged_groups=unpriv, privileged_groups=priv)
    metrics = {}
    metrics["num_instances"] = metrics_obj.num_instances()
    metrics["num_privileged"] = metrics_obj.num_instances(privileged=True)
    metrics["num_unprivileged"] = metrics_obj.num_instances(privileged=False)
    metrics["base_rate"] = metrics_obj.base_rate()
    metrics["priv_base_rate"] = metrics_obj.base_rate(privileged=True)
    metrics["unpriv_base_rate"] = metrics_obj.base_rate(privileged=False)
    metrics["priv_num_positive"] = metrics_obj.num_positives(privileged=True)
    metrics["unpriv_num_positive"] = metrics_obj.num_positives(privileged=False)
    metrics["SP_diff"] = metrics_obj.statistical_parity_difference()
    metrics["DP"] = metrics_obj.disparate_impact()

    return metrics

def dataset_info_comparison(ds1: BinaryLabelDataset, ds2: BinaryLabelDataset) :
    info_ds1 = dataset_info(ds1)
    info_ds2 = dataset_info(ds2)
    res = {}
    keys = info_ds1.keys()
    for x in keys :
        diff = info_ds1[x] - info_ds2[x]
        res[x] = {"diff" : diff, "dasaset1" : info_ds1[x], "dasaset2": info_ds2[x]}
    return pd.DataFrame(res).transpose()

def nbr_id_affected(df1: pd.DataFrame, df2: pd.DataFrame) :
    """Return the numbers of instances that have affected by the biasing of df1 that produced df2
        df1 : Original dataset
        df2 : Biased dataset
    """
    res = df1.compare(df2) #.convert_to_dataframe()[0]
    return len(res.index)

def pred_info(test_dataset: StandardDataset, pred_dataset: StandardDataset, fav_one=True) :
    """
        pred_dataset and test_dataset must only different in labels
    """
    if pred_dataset is not None :
        sens_attr = test_dataset.protected_attribute_names[0]
        priv = [{sens_attr : int(fav_one)}]
        unpriv = [{sens_attr : 1-int(fav_one)}]
        #TODO line bellow fails if pred_data is None
        test_data = mt.get_subset(test_dataset, pred_dataset) #only keep test instances that are also in predictions (needed in case of undersampling)
        metrics_object = ClassificationMetric(test_data, pred_dataset, unprivileged_groups=unpriv, privileged_groups=priv)
        pred_BLD_metrics = BinaryLabelDatasetMetric(pred_dataset, unprivileged_groups=unpriv, privileged_groups=priv)
        # Create BinaryLabelDatasetMetric blind to sensitive attribute to compute blind consistency
        min_max_scaler = MaxAbsScaler()
        blind_pred = pred_dataset.copy()
        blind_pred.features = fair.blind_features(pred_dataset)
        blind_pred.features = min_max_scaler.fit_transform(blind_pred.features)
        #Warning, blind_pred.feature_names does not correspond to blind_pred.features
        pred_blind_BLD_metrics = BinaryLabelDatasetMetric(blind_pred, unprivileged_groups=unpriv, privileged_groups=priv)
        results = {}
        #Performances measures
        results['base_rate'] = metrics_object.base_rate()
        results['acc'] = metrics_object.accuracy()
        results['TPR'] = metrics_object.true_positive_rate()
        results['TNR'] = metrics_object.true_negative_rate()
        results['FPR'] = metrics_object.false_positive_rate()
        results['FNR'] = metrics_object.false_negative_rate()
        results['F1'] = f1_score(test_dataset.labels, pred_dataset.labels)
        #Fairness measures
        #results['DP_ratio'] = metrics_object.disparate_impact()
        results['StatParity'] = metrics_object.statistical_parity_difference()
        results['Consistency'] = pred_BLD_metrics.consistency()[0]
        results['BlindCons'] = pred_blind_BLD_metrics.consistency()[0]
        results['EqqOppDiff'] = metrics_object.equal_opportunity_difference() # = TPRunpriv-TPRpriv = FNRpriv-FNRunpriv
        results['EqqOddsDiff'] = metrics_object.equalized_odds_difference() 
        results['FalseDiscRate'] = metrics_object.false_discovery_rate_difference()
        results['FalsePosRateDiff'] = metrics_object.false_positive_rate_difference()
        results['FalseNegRateDiff'] = metrics_object.false_negative_rate_difference()
        results['GenEntropyIndex'] = metrics_object.generalized_entropy_index()

        results['ConfMatrAll'] = metrics_object.binary_confusion_matrix()
        results['ConfMatrUnpriv'] = metrics_object.binary_confusion_matrix(False)
        results['ConfMatrPriv'] = metrics_object.binary_confusion_matrix(True)
        #print(metrics_object.binary_confusion_matrix())
    else :
        results = dict.fromkeys(['base_rate','acc','TPR','TNR','FPR','FNR','F1',
                                 'StatParity','Consistency','BlindCons','EqqOppDiff','EqqOddsDiff','FalseDiscRate','FalsePosRateDiff','FalseNegRateDiff','GenEntropyIndex',
                                 'ConfMatrAll','ConfMatrUnpriv','ConfMatrPriv',]
                                 ,float('nan'))
    #del test_dataset, pred_dataset, metrics_object, pred_BLD_metrics
    return results

def get_all_metrics(nk_pred, nk_train_splits, path_start: str = None, biased_test = False) :
    """
    nk_pred : Dictionary {float: {int: StandardDataset}}
        Nested dictionaries where n_pred[b][f] holds the prediction given by model trained on fold nbr 'f' of dataset with bias level 'b'
    #TODO n_biased_dataset : Dictionary {float: StandardDataset}
        Dictionary with all the biased version of the dataset, n_biased_dataset[b] = StandardDataset with bias level b
    biased_test : Boolean
        Wether the test set has feature bias level 'b' (True) or is a subset of the original (considered unbiased) dataset (False)
    Returns
    -------
    Dictionary {float: {int: {str: float}}}
        all_info[b][f][metric_name] = Value of 'metric_name' obtained for fold nbr 'k' of model trained with bias level 'b'
    """
    #test_nk_dict : Dictionary {float: list[StandardDataset]}
    #    Dictionary where keys are bias levels and object are the lists of datasets used as test sets for the different folds
    
    all_info = {}
    for b in nk_pred.keys() :
        #print("bias : "+str(b))
        all_info[b] = {}
        #for k in nk_pred[b].keys() :
        for k in range(len(nk_pred[b])) :
            if nk_pred[b][k] is not None :
                if biased_test : #test set has the same level of bias as training set
                    test_fold = nk_train_splits[b][k]['test']
                    #test_fold = mt.get_subset(n_biased_dataset[b],nk_pred[b][k])
                    #pred_sorted = mt.get_subset(nk_pred[b][k],test_fold)
                else : #test set is a subset of the original (considered unbiased) dataset
                    #test_fold = mt.get_subset(n_biased_dataset[0],nk_pred[b][k])
                    #pred_sorted = mt.get_subset(nk_pred[b][k],test_fold)
                    test_fold = nk_train_splits[0][k]['test']
            else :
                test_fold = None
            #print("bias : "+str(b) + " fold : "+str(k))
            #print("nk_pred[b][k]")
            #print(nk_pred[b][k].instance_names)
            #print("test fold as subset of n_biased_dataset[0] :")
            #print(test_fold.instance_names)
            #print("predictions sorted:")
            #print(pred_sorted.instance_names)
            try :
                all_info[b][k] = pred_info(test_fold, nk_pred[b][k])
            except ValueError as err :
                print("ValueError: {}".format(err))#print("bias : "+str(b) + " fold : "+str(k))
                print("Error when pred_info is called by function get_all_metrics")
                print("nk_pred[b][k]")
                print(nk_pred[b][k].instance_names)
                print("Size: "+str(len(nk_pred[b][k].instance_names)))
                print("test_fold")
                print(test_fold.instance_names)
                print("Size: "+str(len(test_fold.instance_names)))
                #print("predictions sorted:")
                #print(pred_sorted.instance_names)
                #print("test fold as subset of n_biased_dataset :")
                #print(test_fold)
                exit()
                
    if path_start is not None :
        if biased_test :
            biased = 'Biased'
        else :
            biased = ''
        path = path_start+"_metrics"+biased+"_all.pkl"
        #file = open(path,"wb")
        with open(path,"wb") as file:
            pickle.dump(all_info,file)
        #file.close()

    return all_info

def metrics_for_plot(nk_results_dict, path_start: str = None, memory_save = False, fold: int = None) :
    """
    nk_results_dic : Dictionary {float: {int: {str: float}}}
        Nested dictionaries where nk_results_dic[b][f][metric_name] = Value of 'metric_name' obtained for fold nbr 'k' of model trained with bias level 'b'
    """
    #print("I am in metrics_for_plot")
    #print("here is nk_results_dict")
    #print(nk_results_dict)
    all_bias = list(nk_results_dict.keys())
    n = len(all_bias)
    bias_keys = nk_results_dict.keys()
    k = len(nk_results_dict[0])
    fold_keys = range(k)
    metrics_keys = nk_results_dict[0][0].keys()
    if fold is not None : #Get results for a specific fold only
        metrics_by_bias = {} #key: metric, object : List of metrics values for each bias level 'b'
        #metrics_by_bias[metric][bias] = metric value (for one fold or avg over all folds)
        for metric in metrics_keys :
            i = 0
            metrics_by_bias[metric] =  [0]*n #List of metric values for each bias level 'b'
            for b in bias_keys:
                metrics_by_bias[metric][i] = nk_results_dict[b][fold][metric]
                i+=1
    else : #Get results for all folds and compute mean and stdev 
        metrics_by_bias = {} #key: metric, object : Dict holding mean and stdev for bias level 'b'
        #metrics_by_bias[metric][mean or std][b] = List of avg metrics values for bias level 'b'
        for metric in metrics_keys :
            if type(nk_results_dict[all_bias[0]][list(fold_keys)[0]][metric]) is not dict :
                i = 0
                metrics_by_bias[metric] = {}
                metrics_by_bias[metric]['mean'] = np.zeros(n) #List of mean metric values for each bias level 'b'
                metrics_by_bias[metric]['stdev'] = np.zeros(n) #[0]*n
                for b in bias_keys :
                    values = [0]*k
                    nan_count = 0
                    for f in fold_keys : #k in fold nbr
                        val = nk_results_dict[b][f][metric]
                        values[f]= nk_results_dict[b][f][metric]
                        if math.isnan(val):
                            nan_count += 1
                    #print("bias level :"+str(b))
                    #print(values)
                    if nan_count <= len(fold_keys)/5 :
                        metrics_by_bias[metric]['mean'][i] = np.mean(values)
                        metrics_by_bias[metric]['stdev'][i] = np.std(values)
                    else : #Too much values are nan for usefull results (post-proc has failed for more than 20% of folds)
                        metrics_by_bias[metric]['mean'][i] = float('nan')
                        metrics_by_bias[metric]['stdev'][i] = float('nan')
                    i+=1
            else :
                pass
                #TODO

    if path_start is not None :
        path = path_start+"_metricsForPlot.pkl"
        with open(path,"wb") as file:
            pickle.dump(metrics_by_bias,file)
    del nk_results_dict
    if memory_save:
        del metrics_by_bias, all_bias
        metrics_by_bias, all_bias = None
    
    return metrics_by_bias, all_bias

def metrics_save_all(n_pred, n_pred_bias, nk_train_splits, path) :
    """ Compute and save all metrics in both 'original' format and for plots, with biased and unbiased test set
        Functions created to allow for the memory heavy computation to be done its own process
        Returns None
    """
    print("Unbiased test set")
    all_metrics = get_all_metrics(n_pred, nk_train_splits, path_start=path)
    metrics_for_plot(all_metrics,path_start=path)

    if n_pred_bias is not None :
        print("Biased test set")
        all_metrics_biasedTest = get_all_metrics(n_pred_bias, nk_train_splits, biased_test=True, path_start=path)
        path_biased = path+"_Biased"
        metrics_for_plot(all_metrics_biasedTest,path_start=path_biased)

def metrics_all_format(n_pred, nk_train_splits, path, biased_test:bool) :
    """ Compute and save all metrics in both 'original' format and for plots, with biased and unbiased test set
        Functions created to allow for the memory heavy computation to be done its own process
        Returns None
    """
    #The main goal of this function is to be able to do the computations in a separate process
    all_metrics = get_all_metrics(n_pred, nk_train_splits, path, biased_test)
    metrics_for_plot(all_metrics,path_start=path)


def compute_all(data_list, bias_list, preproc_list, postproc_list, model_list, blinding_list, path_start) :
    """ Compute and save evaluation metrics for all combinations of datasets, bias, preproc, model and blinding given as argument
        Dataset splits and predictions must already be saved and be store at 'path_start'
        Returns None
    """
    #bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #path_start = "Code/ControlledBiasPrivate/data/"
    #datasets = ['OULAD', 'student']  #, 'student' OULAD
    #biases = ['label','selectDouble','selectLow'] #,'selectDouble','selectLow' label
    #preproc_methods = ['', 'reweighting', 'lfr','massaging']
    #ftu corresponds to no preproc + blind model
    #classifiers = ['RF','tree','RF','neural']
    #blind_model = [True,False]
    for ds in data_list :
        for bias in bias_list :
            #Retrieve original (biased) data for test set
            path = path_start+ds+'_'+bias
            #Result of common split
            #with open(path+"_nbias_splits_datasets.pkl","rb") as file:
            #    nk_folds_dict = pickle.load(file)
            #Dictionary with train-test split for each bias level
            with open(path+"_train-test-nk.pkl","rb") as file:
                nk_train_splits = pickle.load(file)
            for preproc in preproc_list :
                for model in model_list :
                    for blind in blinding_list :
                        if blind :
                            visibility = 'Blinded'
                        else :
                            visibility = 'Aware'
                        path = path_start+ds+'_'+bias+'_'+preproc+'_'+model+visibility
                        if len(postproc_list) == 0 :
                            print("\nComputing metrics for "+path)
                            #Retrieve predictions
                            with open(path+"_pred_all.pkl","rb") as file:
                                n_pred = pickle.load(file)
                            path_biased = path+"_biasedTest"
                            with open(path_biased+"_pred_all.pkl","rb") as file:
                                n_pred_bias = pickle.load(file)
                            # Compute in distinct process for memory management
                            print(n_pred.keys())
                            path_res = path_start+"Results/"+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_'
                            proc1 = mp.Process(target = metrics_all_format, args=(n_pred['pred'], n_pred['orig'], path_res, False))
                            proc1.start()
                            proc1.join()
                            path_res_biased = path_res+"_Biased"
                            proc1 = mp.Process(target = metrics_all_format, args=(n_pred_bias['pred'], n_pred_bias['orig'], path_res_biased, True))
                            proc1.start()
                            proc1.join()
                            del n_pred, n_pred_bias
                        else :
                            for post_proc in postproc_list :
                                print("\nComputing metrics for "+path)
                                #Retrieve predictions
                                path = path_start+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_'+post_proc
                                path_biasedValidFairTest = path+"-BiasedValidFairTest"
                                path_biasedValidBiasedTest = path+"-BiasedValidBiasedTest"
                                path_fairValidFairTest = path+"-FairValidFairTest"
                                with open(path_biasedValidFairTest+"_n.pkl","rb") as file:
                                    n_pred_transf_biasedValidFairTest = pickle.load(file)
                                with open(path_biasedValidBiasedTest+"_n.pkl","rb") as file:
                                    n_pred_transf_biasedValidBiasedTest = pickle.load(file)
                                with open(path_fairValidFairTest+"_n.pkl","rb") as file:
                                    n_pred_transf_FairValidFairTest = pickle.load(file)
                                # Compute in distinct process for memory management
                                #TODO add it if I do fairTruth and biasedTest
                                path_res = path_start+"Results/"+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_'+post_proc
                                path_res_biasedValidFairTest = path_res+"-BiasedValidFairTest"
                                path_res_biasedValidBiasedTest = path_res+"-BiasedValidBiasedTest"
                                path_res_fairValidFairTest = path_res+"-FairValidFairTest"
                                #path_biasedPred = path +"_biasedTest"
                                #n_pred, n_pred_bias, n_biased_dataset
                                #print("About to compute for post-proc "+str(post_proc))
                                #print("keys:")
                                #print(n_pred_transf_biasedValidFairTest.keys())
                                #print("end")
                                proc = mp.Process(target = metrics_all_format, args=(n_pred_transf_biasedValidFairTest['pred'], n_pred_transf_biasedValidFairTest['orig'], path_res_biasedValidFairTest, False))
                                proc.start()
                                proc.join()
                                proc = mp.Process(target = metrics_all_format, args=(n_pred_transf_biasedValidBiasedTest['pred'], n_pred_transf_biasedValidBiasedTest['orig'], path_res_biasedValidBiasedTest, True))
                                proc.start()
                                proc.join()
                                proc = mp.Process(target = metrics_all_format, args=(n_pred_transf_FairValidFairTest['pred'], n_pred_transf_FairValidFairTest['orig'], path_res_fairValidFairTest, False))
                                proc.start()
                                proc.join()
                                del n_pred_transf_biasedValidFairTest, n_pred_transf_biasedValidBiasedTest, n_pred_transf_FairValidFairTest
                                gc.collect()
                        gc.collect()
            del nk_train_splits
            gc.collect()


