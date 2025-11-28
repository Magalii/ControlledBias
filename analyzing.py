"""
    Methods the compute evaluation metrics
"""

import multiprocessing as mp
import pandas as pd
import numpy as np
import subprocess
import warnings
import pickle
import math
import gc
import sys
sys.path.append('../')
sys.path.append('parent_aif360')

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric #Metrics about 1 dataset
from aif360.metrics.classification_metric import ClassificationMetric #Metrics about 2 datasets
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text

import model_training as mt
import fairness_intervention as fair
import consistency_metrics as cm

def dataset_info(BLDataset: BinaryLabelDataset, fav_one=True) :
    """
    Compute metrics for dataset 'BLDataset'
    fav_one (bool) : True if the favored group is indicated by 1.0 in the dataset and the unprivileged group by 0.0, False for the opposite
    """
    if BLDataset is  not None :
        sens_attr = BLDataset.protected_attribute_names[0]
        priv = [{sens_attr : int(fav_one)}]
        unpriv = [{sens_attr : 1-int(fav_one)}]
        metrics_obj = BinaryLabelDatasetMetric(BLDataset, unprivileged_groups=unpriv, privileged_groups=priv)
        dataset_mod = BLDataset.copy()
        dataset_mod.features = fair.blind_features(BLDataset) #remove sensitive attribute
        min_max_scaler = MaxAbsScaler()
        dataset_mod.features = min_max_scaler.fit_transform(dataset_mod.features) #scale
        metrics_obj_blind = BinaryLabelDatasetMetric(dataset_mod, unprivileged_groups=unpriv, privileged_groups=priv)
        metrics = {}
        num_inst = metrics_obj.num_instances()
        metrics["num_instances"] = num_inst
        num_priv = metrics_obj.num_instances(privileged=True)
        num_unpriv = num_inst - num_priv
        priv_num_positive = metrics_obj.num_positives(privileged=True)
        unpriv_num_positive = metrics_obj.num_positives(privileged=False)
        metrics["num_privileged"] = num_priv
        metrics["priv_num_positive"] = priv_num_positive
        metrics["priv_num_negative"] = num_priv - priv_num_positive
        metrics["num_unprivileged"] = num_unpriv
        metrics["unpriv_num_positive"] = unpriv_num_positive
        metrics["unpriv_num_negative"] = num_unpriv - unpriv_num_positive
        metrics["base_rate"] = (priv_num_positive+unpriv_num_positive)/num_inst
        priv_base_rate = priv_num_positive/num_priv
        unpriv_base_rate = unpriv_num_positive/num_unpriv
        metrics["priv_base_rate"] = priv_base_rate
        metrics["unpriv_base_rate"] = unpriv_base_rate
        metrics["SP_diff"] = unpriv_base_rate - priv_base_rate
        metrics["DP"] = unpriv_base_rate/priv_base_rate
        try :
            metrics["Consistency"] = metrics_obj.consistency()[0]
            metrics["BlindCons"] = metrics_obj_blind.consistency()[0]
            metrics["Cons_mine"] = cm.consistency(BLDataset)
            metrics["BCC"] = cm.bcc(BLDataset)
            metrics["BCC_penalty"] = cm.bcc(BLDataset,penalty=1)
        except ValueError as err :
            print(str(err) + "\n Consistency related metrics are set to NaN")
            metrics['Consistency'] = np.nan
            metrics["BlindCons"] = np.nan
            metrics["Cons_mine"] = np.nan
            metrics["BCC"] = np.nan
            metrics["BCC_penalty"] = np.nan
    else :
        metrics = dict.fromkeys(['num_instances','num_privileged','priv_num_positive','priv_num_negative','num_unprivileged','unpriv_num_positive','unpriv_num_negative',
                                   'base_rate','priv_base_rate','unpriv_base_rate','SP_diff','DP','Consistency','BlindCons','Cons_mine','BCC','BCC_penalty',]
                                 ,np.nan)

    return metrics

def dataset_info_comparison(ds1: BinaryLabelDataset, ds2: BinaryLabelDataset) :
    """
    Compute metrics for datasets 'ds1' and 'ds2' and display their respective results and the difference between the two values
    """
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
    """ Compute evaluation metrics for the predictions in 'pred_datasets'. Sets all metris to numpy.nan if pred_dataset is None
        test_dataset : StandardDataset
            Dataset holding the labels considered as ground truth for the evaluation
        pred_dataset : StandardDataset
            Dataset with the prediction to be evaluated, may be None if no predictions could be made (e.g, post-proc couldn't be computed)
            Must only different from test_dataset in labels
        fav_one : Boolean
            True if the favored group is indicated by 1.0 in the dataset and the unprivileged group by 0.0, False for the opposite
    """
    #If a new metric is added, it must also be added to the dict for the case where pred_dataset is None
    warnings.simplefilter("ignore", category=RuntimeWarning) #Ignores RuntimeWarning about invalid value encountered in scalar divide
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
        #Warning: blind_pred.feature_names does not correspond to blind_pred.features
        pred_blind_BLD_metrics = BinaryLabelDatasetMetric(blind_pred, unprivileged_groups=unpriv, privileged_groups=priv)
        results = {}
        #Performances measures
        tpr = metrics_object.true_positive_rate()
        tnr = metrics_object.true_negative_rate()
        results['base_rate'] = metrics_object.base_rate()
        results['acc'] = metrics_object.accuracy()
        results['accBal'] = (tpr+tnr)/2
        results['TPR'] = tpr
        results['TNR'] = tnr
        results['FPR'] = metrics_object.false_positive_rate()
        results['FNR'] = metrics_object.false_negative_rate()
        results['F1'] = f1_score(test_dataset.labels, pred_dataset.labels)
        try : 
            results['ROCAUC'] = roc_auc_score(test_dataset.labels, pred_dataset.scores)
        except ValueError as err :
            print(str(err) + "\n ROC-AUC is set to NaN")
            results['ROCAUC'] = np.nan
        #Fairness measures
        #results['DP_ratio'] = metrics_object.disparate_impact()
        results['StatParity'] = metrics_object.statistical_parity_difference()
        try :
            results['Consistency'] = pred_BLD_metrics.consistency()[0] #Out of the box AIF360 metric
            results['BlindCons'] = pred_blind_BLD_metrics.consistency()[0] #Sens. attr. excluded from features for proximity
            results["Cons_mine"] = cm.consistency(pred_dataset) #For proximity measure : sensitive attribute is excluded, features are scaled, individuals aren't considered as one of their neighbors
            results['BCC'] = cm.bcc(pred_dataset)
            results['BCC_penalty'] = cm.bcc(pred_dataset,penalty=1)
        except ValueError as err :
            print(str(err) + "\n Consistency related metrics are set to NaN")
            results['Consistency'] = np.nan
            results['BlindCons'] = np.nan
            results['Cons_mine'] = np.nan
            results['BCC'] = np.nan
            results['BCC_penalty'] = np.nan
        results['EqqOppDiff'] = metrics_object.equal_opportunity_difference() # = TPRunpriv-TPRpriv = FNRpriv-FNRunpriv
        results['EqqOddsDiff'] = metrics_object.equalized_odds_difference() 
        results['AvOddsDiff'] = metrics_object.average_odds_difference()
        results['FalseDiscRate'] = metrics_object.false_discovery_rate_difference()
        results['FalsePosRateDiff'] = metrics_object.false_positive_rate_difference()
        results['FalseNegRateDiff'] = metrics_object.false_negative_rate_difference()
        results['GenEntropyIndex'] = metrics_object.generalized_entropy_index()
        results['ConfMatrAll'] = metrics_object.binary_confusion_matrix()
        results['ConfMatrUnpriv'] = metrics_object.binary_confusion_matrix(False)
        results['ConfMatrPriv'] = metrics_object.binary_confusion_matrix(True)
        #print(metrics_object.binary_confusion_matrix())
    else :
        results = dict.fromkeys(['base_rate','acc','accBal','TPR','TNR','FPR','FNR','F1','ROCAUC',
                                 'StatParity','Consistency','BlindCons','Cons_mine','BCC','BCC_penalty','EqqOppDiff','EqqOddsDiff','AvOddsDiff',
                                 'FalseDiscRate','FalsePosRateDiff','FalseNegRateDiff','GenEntropyIndex',
                                 'ConfMatrAll','ConfMatrUnpriv','ConfMatrPriv',]
                                 ,np.nan)
    #del test_dataset, pred_dataset, metrics_object, pred_BLD_metrics
    return results

def get_all_metrics(nk_pred, nk_train_splits, clas_type:str=None, dataset_name:str=None, bias_name:str=None, path_start: str = None, biased_test = False, to1 = False, recompute = False) :
    """ Compute evaluation metrics for all the prediction datasets in nk_pred
    nk_pred : Dictionary {float: {int: StandardDataset}}
        Nested dictionaries where n_pred[b][f] holds the prediction given by model trained on fold nbr 'f' of dataset with bias level 'b'
        May be None if no predictions could be made (e.g, post-proc couldn't be computed)
    nk_train_splits : Dictionary {float: {int: StandardDataset}}
        Embedded dictionaries containing the datasets holding the ground truth
        nk_train_splits[bias][fold]: {'train': train set, 'test': test set}
    clas_type : String, optional
        if not None, textual representation of trees must be available
        if 'tree' or 'RF', proportion of sens. attribute usage by classifiers is computed and added to results dict
        if 'neural', sens_attr_usage is added to results dict and set to None
        if None, sens_attr_usage is NOT added to results dict
    dataset_name : str, optional
        Necessary if 'clas_type' is not None, name of dataset used to trained the model for which metrics are computed
    bias_name : str, optional
        Necessary if 'clas_type' is not None, type of bias in the data used to trained the model for which metrics are computed
    biased_test : Boolean
        Wether the test set has feature bias level 'b' (True) or is a subset of the original (considered unbiased) dataset (False)
    Returns
    -------
    Dictionary {float: {int: {str: float}}}
        all_info[b][f][metric_name] = Value of 'metric_name' obtained for fold nbr 'k' of model trained with bias level 'b'
    """
    if biased_test : biased = 'Biased'
    else : biased = ''
    path = path_start+"_metrics"+biased+"_all.pkl"
    if not recompute :
        try : #Check if the metrics have already been computed
            with open(path,"rb") as file:
                all_info = pickle.load(file)
        except (Exception, pickle.UnpicklingError) as err :
            recompute = True
    if recompute :
        if clas_type in ['RF'] : #Generate textual representation of trees before calling sens_attr_usage
            txt_nk_trees(clas_type, dataset_name, bias_name, nk_pred[0][0].feature_names,to1=to1)
        all_info = {}
        for b in nk_pred.keys() :
            all_info[b] = {}
            #for k in nk_pred[b].keys() :
            for k in range(len(nk_pred[b])) :
                if nk_pred[b][k] is not None :
                    if biased_test : #test set has the same level of bias as training set
                        test_fold = nk_train_splits[b][k]['test']
                    else : #test set is a subset of the original (considered unbiased) dataset
                        test_fold = nk_train_splits[0][k]['test']
                else :
                    test_fold = None
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
                    exit()
                if clas_type in ['RF'] :
                    if nk_pred[b][k] is not None :
                        all_info[b][k]['sens_attr_usage'] = sens_attr_usage(clas_type, dataset_name, nk_pred[b][k].protected_attribute_names[0], bias_name, b, k)
                    else :
                        all_info[b][k]['sens_attr_usage'] = np.nan
                elif clas_type == 'tree' or clas_type == 'neural':
                    all_info[b][k]['sens_attr_usage'] = np.nan
                elif clas_type is not None :
                    print("WARNING Not a valid classifier type. Must be 'tree', 'RF' or 'neural'")
                #Must keep separate conditions for nk_pred[b][k] and clas_type to be able to raise warning on class type
                
        if path_start is not None :
            #file = open(path,"wb")
            with open(path,"wb") as file:
                pickle.dump(all_info,file)
    return all_info

def metrics_for_plot(nk_results_dict, path_start: str = None, memory_save = False, fold: int = None) :
    """Take a dictionary of metrics computed using get_all_metrics() and returns a dictionary of metrics that is more convenient for plotting results
    nk_results_dic : Dictionary {float: {int: {str: float}}}
        Nested dictionaries where nk_results_dic[b][f][metric_name] = Value of 'metric_name' obtained for fold nbr 'k' of model trained with bias level 'b'
    path_start: String
        if not None, save results dict on disk at address 'path_start' + function postfix
    """
    path = path = path_start+"_metricsForPlot.pkl"
    try : #Check if the metrics have already been computed
        with open(path,"rb") as file:
            metrics_by_bias = pickle.load(file)
            all_bias = None
    except (Exception, pickle.UnpicklingError) as err :
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
                        if nan_count > len(fold_keys)/2 : #Too much values are nan for usefull results (post-proc has failed for more than 50% of folds)
                            metrics_by_bias[metric]['mean'][i] = float('nan')
                            metrics_by_bias[metric]['stdev'][i] = float('nan')
                        else :
                            metrics_by_bias[metric]['mean'][i] = np.mean(values)
                            metrics_by_bias[metric]['stdev'][i] = np.std(values)
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

def metrics_save_all(n_pred, n_pred_bias, nk_train_splits, path:str, clas_type:str=None, dataset_name:str=None, bias_name:str=None, sens_attr:str=None, to1=False, recompute=False) :
    """ Compute and save all metrics in both 'original' format and suitable for plots, with biased and unbiased test set
        Functions created to allow for the memory heavy computation to be done its own process
        clas_type : String, optional
            if 'tree' or 'RF', proportion of sens. attribute usage by classifiers is computed and added to results dict
            if 'neural', sens_attr_usage is added to results dict and set to None
            if None, sens_attr_usage is NOT added to results dict
        dataset_name : str, optional
            Necessary if 'clas_type' is not None, name of dataset used to trained the model for which metrics are computed
        bias_name : str, optional
            Necessary if 'clas_type' is not None, type of bias in the data used to trained the model for which metrics are computed
        Returns None
    """
    print("Unbiased test set")
    all_metrics = get_all_metrics(n_pred, nk_train_splits, clas_type, dataset_name, bias_name, path_start=path, to1=to1, recompute=recompute)
    metrics_for_plot(all_metrics,path_start=path)

    if n_pred_bias is not None :
        print("Biased test set")
        all_metrics_biasedTest = get_all_metrics(n_pred_bias, nk_train_splits, clas_type, dataset_name, bias_name, biased_test=True, path_start=path, recompute=recompute)
        path_biased = path+"_Biased"
        metrics_for_plot(all_metrics_biasedTest,path_start=path_biased)

def metrics_all_format(n_pred, nk_train_splits, path, biased_test:bool, clas_type:str=None, dataset_name:str=None, bias_name:str=None, sens_attr:str=None, to1=False, recompute=False) :
    """ Compute and save all metrics in both 'original' format and suitable for plots, with biased and unbiased test set
        Functions created to allow for the memory heavy computation to be done in a separate processes
        Returns None
    """
    all_metrics = get_all_metrics(n_pred, nk_train_splits, clas_type, dataset_name, bias_name, path, biased_test,to1=to1, recompute=recompute)
    metrics_for_plot(all_metrics,path_start=path)

def txt_nk_trees(clas_type:str, dataset:str, bias_type:str, feature_names:list[str], to1=False, path_retrieval:str=None) :
    """ Generate and saves textual representation of all the decision trees trained on 'dataset' with 'bias_type' (one tree for each bias level and fold)
        dataset : String
            name of dataset on which the trees where trained and which is used in the file name for the pickled dictionary of datasets
        bias_type : String
            name of bias type that was introduced in the data and which is used in the file name for the pickled dictionary of datasets
        feature_names : List[string]
            list of the feature_names listed in the dataset named 'dataset'
        path_retrieval :
            path at which the pickled file containing the dictionary of datasets can be found
    """
    if path_retrieval is None :
        path_retrieval = "data/"
    if to1 : pref = '0to1_'
    else : pref = ''
    path_trees = path_retrieval+pref+dataset+'_'+bias_type+'__'+clas_type+"Aware_all.pkl"
    with open(path_trees,"rb") as file:
        nk_classifier = pickle.load(file)
    for n in nk_classifier.keys() :
        if n == 0 : nn = "b0.0" # Establish consistent naming style for all bias levels #TODO remove this if experiments are rerun with bias=0.0
        else : nn = 'b'+str(n)
        for k in nk_classifier[0].keys() :
            clas = nk_classifier[n][k]
            clas_names = [str(i) for i in clas.classes_]
            file_name = "tree_"+dataset+'_'+bias_type+'_'+str(nn)+'k'+str(k)
            if clas_type == 'tree':
                text = export_text(clas, feature_names=feature_names,class_names=clas_names )
                with open("trees/"+file_name+".txt", "w") as fout:
                    fout.write(text)
            elif clas_type == 'RF':
                for i in range(len(clas.estimators_)) :
                    est = clas.estimators_[i] #one tree composing the RF
                    text = export_text(est, feature_names=feature_names, class_names=clas_names)
                    with open("RFs/"+file_name+'i'+str(i)+".txt", "w") as fout:
                        fout.write(text)
            else :
                print("WARNING Wrong tree-based classifier name. Only 'tree' and 'RF' are supported")

def txt_trees_all(clas_type:str, dataset_list:list[str], bias_list:list[str], path_retrieval:str, to1=False) :
    """ Generate and saves textual representation of all the nk tree-based classifiers for all datasets, biase types and blinding types provided
        clas_type : String
            'tree' to represent the decision tree classifiers
            'RF' to represent the random forest classifiers (generate all trees for each RF)
        dataset_list : list[String]
            list of dataset names for which the textual representation will be generated (dataset on which the trees where trained and which is used in the file name for the pickled dictionaries of datasets)
        bias_list : list[String]
            list of bias types for which the textual representation will be generated (bias introduced in the data and which is used in the file name for the pickled dictionary of datasets)
        path_retrieval :
            path at which the pickled file containing the dictionary of datasets can be found
        to1 : Boolean
            True to consider scenarios where the bias levels used for training go from 0 to 1 (prefix 0to1_ for file paths), False otherwise
    """
    if to1 : pref = '0to1_'
    else : pref = ''
    for dataset in dataset_list :
        path_data = path_retrieval+dataset+"_dataset"
        with open(path_data,"rb") as file:
            ds_obj = pickle.load(file)
        for bias in bias_list :
            txt_nk_trees(clas_type, dataset, bias, ds_obj.feature_names, to1, path_retrieval+pref)


def sens_attr_usage_all(clas_type:str, dataset_list:list[str], bias_list:list[str],
                    bias_levels:list[float], to1=False, recompute_path:str=None) :
    """ Compute the proportion of trees that use the sensitive attribute and save the results in a dict store on disk (update the existing dict or create a new one if none exists)
        clas_type : String
            'tree' to represent the decision tree classifiers
            'RF' to represent the random forest classifiers (generate all trees for each RF)
        dataset_list : list[String]
            list of dataset names for which the sens. attr. usage will be computed
        bias_list : list[String]
            list of bias types for which the sens. attr. usage will be computed
        bias_levels : list[float]

        to1 : Boolean
            True to consider scenarios where the bias levels used for training go from 0 to 1 (prefix 0to1_ for file paths), False otherwise
        recompute_path : String
            None if the textual representation of the classifier to analyze is available on disk,
            otherwise path to pickled files containing the dictionary of the corresponding datasets so the textual representation can be computed
    """
    if clas_type == 'tree':
        num_tree = 1
    elif clas_type == 'RF':
        num_tree = 100
    else : print("WARNING Not a valid type of tree based classifer. clas_type must be 'tree' or 'RF'")
    
    if recompute_path is not None : #Generate textual representation of trees if needed
        txt_trees_all(clas_type, dataset_list, bias_list, recompute_path, to1)

    try :
        with open("data/Results/"+clas_type+"_trees_count.pkl","rb") as file:
            results_dict = pickle.load(file)
    except (Exception, pickle.UnpicklingError) as err:
        results_dict = {}

    for dataset in dataset_list :
        if dataset[0:7] == "student" : sens_attr = "'sex >'"
        elif dataset[0:5] == "OULAD" : sens_attr = "'gender >'"
        else : print("WARNING Not a valid dataset name. Must be 'student', 'OULADsocial' or 'OULADstem'")
        if len(results_dict.keys()) == 0: #Newly created empty dict
            results_dict[dataset] = {}
        for bias in bias_list :
            if bias in ['selectLow','selectDoubleProp','selectRandom','selectPrivNoUnpriv','selectRandomWhole'] : folds = 5
            elif bias in ['label','labelDouble'] : folds = 10
            else : print("WARNING Not a valid bias type")
            results_dict[dataset][bias] = [0]*len(bias_levels)
            i = 0
            for b in bias_levels :
                #for k in range(5) : # 5 folds since we only look at selection bias (file_names = "tree_"+dataset+"_"+bias+'_b'+str(b)+'k'+str(k)+'*.txt')
                file_names = clas_type+"s/tree_"+dataset+"_"+bias+'_b'+str(b)+'*.txt'
                process = subprocess.Popen(["grep -c "+sens_attr+" "+file_names+" | grep -v ':0' | grep -c "+dataset],shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout,stderr = process.communicate()
                results_dict[dataset][bias][i] = int((stdout))/(folds*num_tree) # Save proportion of individual trees that use the sensitive attribute (100 trees per RF)
                i+=1
                # grep - c string @Count number of occurances per file
                # cat *_student_* | grep -c "sex >" @ Count total number of occurances (Combines all files, then give to grep)
                # grep -c "sex >" *_student_* | grep -v ":0" @ Filter out outputs lines with zero occurences of 'sex'
                # grep -c "sex >" *_student_* | grep -v ":0" | grep -c "student" @ Counts the number of file which name contain "student" that contain an occurence of 'sex'
    
    with open("data/Results/"+clas_type+"s_trees_count.pkl","wb") as file:
        pickle.dump(results_dict,file)
    
    return results_dict

def sens_attr_usage(clas_type:str, dataset_name:str, sens_attr:str, bias_name:str, bias_level:float, fold_num:int) :
    """ Compute the proportion of trees that use the sensitive attribute and save the results in a dict store on disk (update the existing dict or create a new one if none exists)
        clas_type : String
            'tree' to represent the decision tree classifiers
            'RF' to represent the random forest classifiers (generate all trees for each RF)
        dataset_name : String
            name of dataset used to trained the model for which the sens. attr. usage will be computed
        sens_attr : String
            name of the designated sens. attr. in 'dataset'
        bias_name : String
           type of bias in the data used to trained the model for which the sens. attr. usage will be computed
        bias_level : Float
            Level of bias in the data used to trained the model for which the sens. attr. usage will be computed
        fold_num : Integer
            Number of the fold for the sens. attr. usage will be computed
        to1 : Boolean
            True to consider scenarios where the bias levels used for training go from 0 to 1 (prefix 0to1_ for file paths), False otherwise
        recompute_path : String
            None if the textual representation of the classifier to analyze is available on disk,
            otherwise path to pickled files containing the dictionary of the corresponding datasets so the textual representation can be computed
    """
    if clas_type == 'tree': num_tree = 1
    elif clas_type == 'RF': num_tree = 100
    else : print("WARNING Not a valid type of tree based classifer. clas_type must be 'tree' or 'RF'")

    sens_attr = "'"+sens_attr+" >'"
    if bias_level == 0 : b = "0.0"
    else : b = str(bias_level)
    file_names = clas_type+"s/tree_"+dataset_name+"_"+bias_name+"_b"+b+"k"+str(fold_num)+"*.txt"
    process = subprocess.Popen(["grep -c "+sens_attr+" "+file_names+" | grep -v ':0' | grep -c "+dataset_name],shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout,stderr = process.communicate()
    result = int((stdout))/num_tree # Save proportion of individual trees that use the sensitive attribute (100 trees per RF)

    # grep - c string @Count number of occurances per file
    # cat *_student_* | grep -c "sex >" @ Count total number of occurances (Combines all files, then give to grep)
    # grep -c "sex >" *_student_* | grep -v ":0" @ Filter out outputs lines with zero occurences of 'sex'
    # grep -c "sex >" *_student_* | grep -v ":0" | grep -c "student" @ Counts the number of file which name contain "student" that contain an occurence of 'sex'

    return result

"""
# WARNING
#Goal to compute the proportion of trees that use the sensitive attribute in their features :
#DOESN'T GIVE THE SAME RESULTS AS sens_attr_usage()
def sens_attr_prop(classifier, sens_attr:str, feature_names:list[str]) : #bias_name:str
    if type(classifier) == RandomForestClassifier : num_tree = 100
    elif type(classifier) == DecisionTreeClassifier : num_tree = 1
    else : print("WARNING Not a valid type of tree based classifer. clas_type must be 'tree' or 'RF'")
    #if bias_name in ['selectLow','selectDoubleProp','selectRandom','selectPrivNoUnpriv'] : folds = 5
    #elif bias_name in ['label','labelDouble'] : folds = 10
    #else : print("WARNING Not a valid bias type")
    if type(classifier) == DecisionTreeClassifier :
       count = _count_feature_use(sens_attr, feature_names, classifier.feature_importances_)
    elif type(classifier) == RandomForestClassifier :
        count = 0
        for est in classifier.estimators_ :
            count += _count_feature_use(sens_attr, feature_names, est.feature_importances_)
    else :
        print("WARNING Wrong tree-based classifier name. Only DecisionTreeClassifier' and RandomForestClassifier are supported")
    result = count/num_tree
    return result

def _count_feature_use(feature, features_list, features_importances):
    count = 0
    used_features = []
    for i in range(len(features_list)) :
        if features_importances[i] > 0 :
            used_features += [features_list[i]]
    #print(used_features)
    #print(feature)
    if feature in used_features :
        count += 1
    return count
"""

def compute_all(data_list, bias_list, preproc_list, postproc_list, model_list, blinding_list, path_start, no_valid = False, to1 = False, recompute=False) :
    """ Compute and save evaluation metrics for all combinations of datasets, bias, preproc, model and blinding given as argument
        Dataset splits and predictions must already be saved and be store at 'path_start'
        no_valid : Boolean
            True if the data used to train models is split between train and test with no validation set, False if there was a validation set 
        to1 : Boolean
            True if the bias levels used for training go from 0 to 1 (prefix 0to1_ for file paths), False otherwise 
        Returns None
    """
    if no_valid : valid = "_noValid"
    else : valid = ''
    if to1 : pref = "0to1_"
    else : pref = ''
    for ds in data_list :
        if ds[0:7] == "student" : sens_attr = 'sex'
        elif ds[0:5] == 'OULAD' : sens_attr = 'gender'
        for bias in bias_list :
            #Retrieve original (biased) data for test set
            path = path_start+pref+ds+'_'+bias+valid
            #Result of common split
            #with open(path+"_nbias_splits_datasets.pkl","rb") as file:
            #    nk_folds_dict = pickle.load(file)
            #Dictionary with train-test split for each bias level
            #with open(path+"_train-test-nk.pkl","rb") as file:
            #    nk_train_splits = pickle.load(file)
            for preproc in preproc_list :
                for model in model_list :
                    for blind in blinding_list :
                        if blind : visibility = 'Blinded'
                        else : visibility = 'Aware'
                        path = path_start+pref+ds+'_'+bias+valid+'_'+preproc+'_'+model+visibility
                        if len(postproc_list) == 0 :
                            print("\nComputing metrics for "+path)
                            #Retrieve predictions
                            with open(path+"_pred_all.pkl","rb") as file:
                                n_pred = pickle.load(file)
                            path_biased = path+"_biasedTest"
                            with open(path_biased+"_pred_all.pkl","rb") as file:
                                n_pred_bias = pickle.load(file)
                            # Compute in distinct process for memory management
                            path_res = path_start+"Results/"+pref+ds+'_'+bias+valid+'_'+preproc+'_'+model+visibility+'_'
                            proc1 = mp.Process(target = metrics_all_format, args=(n_pred['pred'], n_pred['orig'], path_res, False, model, ds, bias,sens_attr,to1,recompute))
                            proc1.start()
                            proc1.join()
                            path_res_biased = path_res+"_Biased"
                            proc1 = mp.Process(target = metrics_all_format, args=(n_pred_bias['pred'], n_pred_bias['orig'], path_res_biased, True, model, ds, bias,sens_attr,to1,recompute))
                            proc1.start()
                            proc1.join()
                            del n_pred, n_pred_bias
                        else :
                            for post_proc in postproc_list :
                                print("\nComputing metrics for "+path)
                                #Retrieve predictions
                                path = path_start+pref+ds+'_'+bias+valid+'_'+preproc+'_'+model+visibility+'_'+post_proc
                                path_biasedValidFairTest = path+"-BiasedValidFairTest"
                                path_biasedValidBiasedTest = path+"-BiasedValidBiasedTest"
                                #path_fairValidFairTest = path+"-FairValidFairTest"
                                with open(path_biasedValidFairTest+"_n.pkl","rb") as file:
                                    n_pred_transf_biasedValidFairTest = pickle.load(file)
                                with open(path_biasedValidBiasedTest+"_n.pkl","rb") as file:
                                    n_pred_transf_biasedValidBiasedTest = pickle.load(file)
                                #with open(path_fairValidFairTest+"_n.pkl","rb") as file:
                                #    n_pred_transf_FairValidFairTest = pickle.load(file)
                                # Compute in distinct process for memory management
                                path_res = path_start+"Results/"+pref+ds+'_'+bias+valid+'_'+preproc+'_'+model+visibility+'_'+post_proc
                                path_res_biasedValidFairTest = path_res+"-BiasedValidFairTest"
                                path_res_biasedValidBiasedTest = path_res+"-BiasedValidBiasedTest"
                                #path_res_fairValidFairTest = path_res+"-FairValidFairTest"
                                proc = mp.Process(target = metrics_all_format, args=(n_pred_transf_biasedValidFairTest['pred'], n_pred_transf_biasedValidFairTest['orig'], path_res_biasedValidFairTest, False, model, ds, bias, recompute))
                                proc.start()
                                proc.join()
                                proc = mp.Process(target = metrics_all_format, args=(n_pred_transf_biasedValidBiasedTest['pred'], n_pred_transf_biasedValidBiasedTest['orig'], path_res_biasedValidBiasedTest, True, model, ds, bias, recompute))
                                proc.start()
                                proc.join()
                                #proc = mp.Process(target = metrics_all_format, args=(n_pred_transf_FairValidFairTest['pred'], n_pred_transf_FairValidFairTest['orig'], path_res_fairValidFairTest, False, model, ds, bias))
                                #proc.start()
                                #proc.join()
                                del n_pred_transf_biasedValidFairTest, n_pred_transf_biasedValidBiasedTest#, n_pred_transf_FairValidFairTest
                                gc.collect()
                        gc.collect()
            #del nk_train_splits
            gc.collect()


