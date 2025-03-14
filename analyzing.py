import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import sys 
sys.path.append('Code/')
sys.path.append('Code/parent_aif360')

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric #Metrics about 1 dataset
from aif360.metrics.classification_metric import ClassificationMetric #Metrics about 2 datasets
from sklearn.metrics import f1_score

#from ControlledBias import dataset_creation as dc
from ControlledBias import model_training as mt

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
    sens_attr = test_dataset.protected_attribute_names[0]
    priv = [{sens_attr : int(fav_one)}]
    unpriv = [{sens_attr : 1-int(fav_one)}]
    test_data = mt.get_subset(test_dataset, pred_dataset) #only keep test instances that are also in predictions (needed in case of undersampling)
    metrics_object = ClassificationMetric(test_data, pred_dataset, unprivileged_groups=unpriv, privileged_groups=priv)
    pred_BLD_metrics = BinaryLabelDatasetMetric(pred_dataset, unprivileged_groups=unpriv, privileged_groups=priv)
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
    results['Consistency'] = pred_BLD_metrics.consistency()
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

    del test_dataset, pred_dataset, metrics_object, pred_BLD_metrics
    return results

def get_all_metrics(test_nk_dict, nk_pred, path_start: str = None, biased_test = False, memory_save = False) :
    """
    test_nk_dict : Dictionary {float: list[StandardDataset]}
        Dictionary where keys are bias levels and object are the lists of datasets used as test sets for the different folds
    nk_pred : Dictionary {float: {int: StandardDataset}}
        Nested dictionaries where n_pred[b][f] holds the prediction given by model trained on fold nbr 'f' of dataset with bias level 'b'
        The folds numbering has to correspond to that of test_nk_dict
    biased_test : Boolean
        Wether the test set has feature bias level 'b' (True) or is a subset of the original (considered unbiased) dataset (False)
    Returns
    -------
    Dictionary {float: {int: {str: float}}}
        all_info[b][f][metric_name] = Value of 'metric_name' obtained for fold nbr 'k' of model trained with bias level 'b'
    """
    all_info = {}
    for b in nk_pred.keys() :
        all_info[b] = {}
        #for k in nk_pred[b].keys() :
        for k in range(len(nk_pred[b])) :
            if biased_test : #test set has the same level of bias as training set
                test_fold = test_nk_dict[b][k]
            else : #test set is a subset of the original (considered unbiased) dataset
                test_fold = test_nk_dict[0][k]
            #print("bias : "+str(b) + " fold : "+str(k))
            all_info[b][k] = pred_info(test_fold, nk_pred[b][k])

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
    
    del test_nk_dict, nk_pred
    if memory_save :
        del all_info
        all_info = None

    return all_info

def compute_all(data_list, bias_list, preproc_list, model_list, blinding_list, path_start) :
    """ Compute and save evaluation metrics for all combinations of datasets, bias, preproc, model and blinding given as argument
        Dataset splits and predictions must already be saved and be store at 'path_start'
        Returns None
        WARNING Uses a lot of RAM. Best to note compute everything in one go
    """

    #bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #path_start = "Code/ControlledBias/data/"
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
            with open(path+"_nbias_splits_datasets.pkl","rb") as file:
                nk_folds_dict = pickle.load(file)
            for preproc in preproc_list :
                for model in model_list :
                    for blind in blinding_list :
                        if blind :
                            visibility = 'Blinded'
                        else :
                            visibility = 'Aware'
                        path = path_start+ds+'_'+bias+'_'+preproc+'_'+model+visibility
                        print("Computing metrics for "+path)
                        #Retrieve predictions
                        with open(path+"_pred_all.pkl","rb") as file:
                            n_pred = pickle.load(file)
                        path_biased = path+"_biasedTest"
                        with open(path_biased+"_pred_all.pkl","rb") as file:
                            n_pred_bias = pickle.load(file)
                        
                        path = path_start+"Results/"+ds+'_'+bias+'_'+preproc+'_'+model+visibility

                        all_metrics = get_all_metrics(nk_folds_dict, n_pred, path_start=path)
                        path_biased = path+"_Biased"
                        all_metrics_biasedTest = get_all_metrics(nk_folds_dict, n_pred_bias, biased_test=True, path_start=path)

                        metr_plots = metrics_for_plot(all_metrics,path_start=path)
                        path_biased = path+"_Biased"
                        bias_metr_plots = metrics_for_plot(all_metrics_biasedTest,path_start=path_biased)
                        
                        #Manage memory
                        del n_pred, all_metrics, all_metrics_biasedTest, metr_plots, bias_metr_plots
            del nk_folds_dict

                        
def plot_all(data_list, bias_list, preproc_list, model_list, blinding_list, all_bias, path_start) :
    """ Plot and save results for all combinations of datasets, bias, preproc, model and blinding given as argument
        Dataset splits metrics for plots must already be saved and be store at 'path_start'
        Returns None
    """
    for ds in data_list :
        for bias in bias_list :
            for preproc in preproc_list :
                for model in model_list :
                    for blind in blinding_list :
                        if blind :
                            visibility = "Blinded"
                        else :
                            visibility = "Aware"
                        path = path_start+"Results/"+ds+'_'+bias+'_'+preproc+'_'+model+visibility
                        with open(path+"_metricsForPlot.pkl","rb") as file:
                            metrics_for_plot = pickle.load(file)
                        with open(path+"_Biased"+"_metricsForPlot.pkl","rb") as file:
                            metricsBiased_for_plot = pickle.load(file)
                        plot_by_bias(metrics_for_plot, all_bias, plot_style='FILLED_STDEV',title='Metric values wrt train set bias level\n ('+bias+' bias, '+preproc+', '+model+visibility+', unbiased test set)', path_start='Code/ControlledBias/plots/'+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_byBias_unbiasedTest', display=False)
                        plot_by_bias(metricsBiased_for_plot, all_bias, plot_style='FILLED_STDEV', title='Metric values wrt train set bias level\n ('+bias+' bias, '+preproc+', '+model+visibility+', biased test set)',path_start='Code/ControlledBias/plots/'+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_byBias_BiasedTest', display=False)
                        #Manage memory
                        metrics_for_plot = None
                        metricsBiased_for_plot = None 

def metrics_for_plot(nk_results_dict, path_start: str = None, memory_save = False, fold: int = None) :
    """
    nk_results_dic : Dictionary {float: {int: {str: float}}}
        Nested dictionaries where nk_results_dic[b][f][metric_name] = Value of 'metric_name' obtained for fold nbr 'k' of model trained with bias level 'b'
    """
    all_bias = list(nk_results_dict.keys())
    n = len(all_bias)
    bias_keys = nk_results_dict.keys()
    k = len(nk_results_dict[0])
    fold_keys = range(k)
    metrics_keys = nk_results_dict[0][0].keys()
    if fold is not None :
        metrics_by_bias = {} #key: metric, object : List of metrics values for each bias level 'b'
        #metrics_by_bias[metric][bias] = metric value (for one fold or avg over all folds)
        for metric in metrics_keys :
            i = 0
            metrics_by_bias[metric] =  [0]*n #List of metric values for each bias level 'b'
            for b in bias_keys:
                metrics_by_bias[metric][i] = nk_results_dict[b][fold][metric]
                i+=1
    else :
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
                    for f in fold_keys : #k in fold nbr
                        values[f]= nk_results_dict[b][f][metric]
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

def plot_by_bias(metrics_by_bias, all_bias: list[float], plot_style: str = 'SIMPLE_PLOT', title: str = '', path_start:str = None, display=True) :
    """
    nk_results_dic : Dictionary {float: {int: {str: float}}}
        Nested dictionaries where nk_results_dic[b][f][metric_name] = Value of 'metric_name' obtained for fold nbr 'k' of model trained with bias level 'b'
    plot_style : string, optional
        'ALL' for basic display of all metrics
        'SIMPLE_PLOT' for choice of metrics displaid without standard deviation
        'FILLED_STDEV' for choice of metrics displaid with standard deviation as colored area arround the curve
    """

    fig, ax = plt.subplots() # (figsize=(length, heigth)) #Scale size of image
    ax.hlines(0,0,1,colors='black')

    if plot_style is None :
        for metric in metrics_keys :
            #if metric != "DP_ratio":
            plt.plot(all_bias,metrics_by_bias[metric],label = str(metric),linestyle="--",marker="o")
            plt.style.use('tableau-colorblind10')
    elif  plot_style == 'SIMPLE_PLOT' or plot_style == 'FILLED_STDEV' :
        #Mean values
        ax.plot(all_bias,metrics_by_bias['Consistency']['mean'], label = 'Consistency', linestyle="--",marker="o", color="#006BA4")#Cerulean/Blue
        ax.plot(all_bias,metrics_by_bias['acc']['mean'], label = 'Accuracy', linestyle="--",marker="o", color='#595959')##Dark gray
        ax.plot(all_bias,metrics_by_bias['FNR']['mean'], label = 'FNR global', linestyle="--",marker="s", color='#ABABAB')##Light gray #TODO FNR
        ax.plot(all_bias,metrics_by_bias['FPR']['mean'], label = 'FPR global', linestyle="--",marker="+", color='#ABABAB')##Light gray #TODO FPR
        #ax.plot(all_bias,metrics_by_bias['F1']['mean'], label = 'F1 score', linestyle="--",marker="o", color='#595959')##Dark gray
        #ax.plot(all_bias,metrics_by_bias['EqqOddsDiff']['mean'], label = 'Eq. Odds', linestyle="--",marker="X", c='#C85200')#Tenne/Dark orange
        #ax.plot(all_bias,metrics_by_bias['EqqOppDiff']['mean'], label = 'EqOpp=TPRdiff', linestyle="--",marker="d", c="#FF800E")#Pumpkin/Bright orange
        ax.plot(all_bias,metrics_by_bias['FalsePosRateDiff']['mean'], label = 'FPRdiff', linestyle="--",marker="X", c='#C85200')#Tenne/Dark orange
        ax.plot(all_bias,metrics_by_bias['FalseNegRateDiff']['mean'], label = 'FNRdiff', linestyle="--",marker="X", c='#FF800E')#Pumpkin/Bright orange
        
        ax.plot(all_bias,metrics_by_bias['StatParity']['mean'], label = 'StatParity', linestyle="--",marker="s", c="#A2C8EC")#Seil/Light blue
        ax.plot(all_bias,metrics_by_bias['GenEntropyIndex']['mean'], label = 'GenEntropyIndex', linestyle="--",marker="^", color="#006BA4")#Cerulean/Blue
        if plot_style == 'FILLED_STDEV':
        #Shade for std values
            ax.fill_between(all_bias,metrics_by_bias['Consistency']['mean'] - metrics_by_bias['Consistency']['stdev'], metrics_by_bias['Consistency']['mean'] + metrics_by_bias['Consistency']['stdev'], edgecolor = None, facecolor='#006BA4', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['acc']['mean'] - metrics_by_bias['acc']['stdev'], metrics_by_bias['acc']['mean'] + metrics_by_bias['acc']['stdev'], edgecolor = None, facecolor='#595959', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['FNR']['mean'] - metrics_by_bias['FNR']['stdev'], metrics_by_bias['FNR']['mean'] + metrics_by_bias['FNR']['stdev'], edgecolor = None, facecolor='#ABABAB', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['FPR']['mean'] - metrics_by_bias['FPR']['stdev'], metrics_by_bias['FPR']['mean'] + metrics_by_bias['FPR']['stdev'], edgecolor = None, facecolor='#ABABAB', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['FalsePosRateDiff']['mean'] - metrics_by_bias['FalsePosRateDiff']['stdev'], metrics_by_bias['FalsePosRateDiff']['mean'] + metrics_by_bias['FalsePosRateDiff']['stdev'], edgecolor = None, facecolor='#C85200', alpha=0.4)
            #ax.fill_between(all_bias,metrics_by_bias['EqqOppDiff']['mean'] - metrics_by_bias['EqqOppDiff']['stdev'], metrics_by_bias['EqqOppDiff']['mean'] + metrics_by_bias['EqqOppDiff']['stdev'], edgecolor = None, facecolor='#FF800E', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['FalseNegRateDiff']['mean'] - metrics_by_bias['FalseNegRateDiff']['stdev'], metrics_by_bias['FalseNegRateDiff']['mean'] + metrics_by_bias['FalseNegRateDiff']['stdev'], edgecolor = None, facecolor='#FF800E', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['StatParity']['mean'] - metrics_by_bias['StatParity']['stdev'], metrics_by_bias['StatParity']['mean'] + metrics_by_bias['StatParity']['stdev'], edgecolor = None, facecolor='#A2C8EC', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['GenEntropyIndex']['mean'] - metrics_by_bias['GenEntropyIndex']['stdev'], metrics_by_bias['GenEntropyIndex']['mean'] + metrics_by_bias['GenEntropyIndex']['stdev'], edgecolor = None, facecolor='#006BA4', alpha=0.4)
    
    ax.tick_params(labelsize = 'large',which='major')
    ax.set_ylim([-1.1,1.1])
    ax.set_xlim([0,1])
    minor_ticks = np.arange(-1,1,0.05)
    #ax.set_yticks(minor_ticks) #, minor=True)
    #ax.set_xlabel(r'$\tau$', size=14)
    
    #ax.grid(which='both',linestyle='-',linewidth=0.5,color='lightgray')
    ax.grid(visible=True)
    ax.grid(which='minor',linestyle=':',linewidth=0.5,color='lightgray')
    ax.minorticks_on()

    #ax.legend(loc='best')
    ax.legend(prop={'size':10}, loc='lower left') #,  bbox_to_anchor=(1, 0.87))
    plt.title(title, fontsize=14)

    if path_start is not None :
            plt.savefig(path_start+".pdf", format="pdf", bbox_inches="tight") # dpi=1000) #dpi changes image quality
    if(display) :
        plt.show()
    plt.close()


def plot_bargraph(retrieval_path:str, dataset_list=['student','OULADstem', 'OULADsocial'], metric_list=['acc','StatParity','EqqOddsDiff','GenEntropyIndex'], bias_list=['label','selectDoubleProp'],preproc_list=['', 'reweighting','massaging'],ylim:list[float]=None, xlim:list[float]=None, all_bias:list[float]=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], plot_style: str = 'SIMPLE_PLOT', title: str = '', path_start:str = None, display=False) :
    """
    plot_style : string, optional
        'SIMPLE_PLOT' for choice of metrics displaid without standard deviation
        'FILLED_STDEV' for choice of metrics displaid with standard deviation as colored area arround the curve
    title : string, optional
        None for automatic title
        '' (empty string) for no title
    """

    metrics_all = {}
    for ds in dataset_list :
        metrics_all[ds] = {}
        for bias in bias_list :
            metrics_all[ds][bias] = {}
            for preproc in preproc_list :
                path = retrieval_path+ds+'_'+bias+'_'+preproc+"_RFAware_metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][bias][preproc] = metrics
            path = retrieval_path+ds+'_'+bias+"__RFBlinded_metricsForPlot.pkl"
            with open(path,"rb") as file:
                metrics = pickle.load(file)
                metrics_all[ds][bias]['FTU'] = metrics
    #metrics_all[bias_type][preproc][metric]['mean']

    for ds in dataset_list :
        #for bias in bias_list:
        for metric in metric_list :
            fig, ax = plt.subplots() # (figsize=(length, heigth)) #Scale size of image
            ax.hlines(y=0,xmin=-0.05,xmax=1,colors='black')

            bar_width = 0.01
            indices = np.array(all_bias)

            if plot_style is None :
                for metric in metrics_keys :
                    #if metric != "DP_ratio":
                    plt.plot(all_bias,metrics_all[metric],label = str(metric),linestyle="--",marker="o")
                    plt.style.use('tableau-colorblind10')
            elif  plot_style == 'SIMPLE_PLOT' or plot_style == 'FILLED_STDEV' :
                ax.bar(indices - 3.6*bar_width,metrics_all[ds]['label'][''][metric]['mean'], label = 'label bias', width=bar_width,color="#595959")#Dark gray
                ax.bar(indices - 2.6*bar_width,metrics_all[ds]['label']['reweighting'][metric]['mean'], label = 'label+reweighing', width=bar_width, color="#006BA4")#Cerulean/Blue
                ax.bar(indices - 1.6*bar_width,metrics_all[ds]['label']['massaging'][metric]['mean'], label = 'label+massaging', width=bar_width, color="#5F9ED1")#Picton blue
                ax.bar(indices - 0.6*bar_width,metrics_all[ds]['label']['FTU'][metric]['mean'], label = 'label+FTU', width=bar_width, color="#A2C8EC")#Seil/Light blue

                ax.bar(indices + 0.6*bar_width,metrics_all[ds]['selectDoubleProp'][''][metric]['mean'], label = 'selection bias', width=bar_width, color="#898989")#Suva Grey
                ax.bar(indices + 1.6*bar_width,metrics_all[ds]['selectDoubleProp']['reweighting'][metric]['mean'], label = 'select+reweighing', width=bar_width, color="#C85200")#Tenne/Dark orange
                ax.bar(indices + 2.6*bar_width,metrics_all[ds]['selectDoubleProp']['massaging'][metric]['mean'], label = 'select+massaging', width=bar_width, color="#FF800E")#Pumpkin/Bright orange
                ax.bar(indices + 3.6*bar_width,metrics_all[ds]['selectDoubleProp']['FTU'][metric]['mean'], label = 'select+FTU', width=bar_width, color="#FFBC79")#Mac and cheese orange
            
            ax.tick_params(labelsize = 'large',which='major')
            if ylim is not None:
                ax.set_ylim(ylim)
            else :
                if metric == 'acc':
                    ax.set_ylim([0.4,1])
            ax.set_xlim([-0.05,0.95])
            ax.set_xlabel('Bias intensity ($\\beta_m$ for label, $p_u$ for selection)', size=14)
            if ds == 'student':
                if metric == 'acc':
                    y_label = "Accuracy"
                elif metric == 'StatParity':
                    y_label = "Statistical Parity Diff."
                elif metric == 'EqqOddsDiff':
                    y_label = "Equalized Odds Diff."
                elif metric == 'GenEntropyIndex':
                    y_label = "Generalized Entropy Ind."
                else :
                    y_label = metric
                ax.set_ylabel(y_label, size = 22)

            ax.grid(which='minor',linestyle=':',linewidth=0.5,color='lightgray')
            ax.minorticks_on()
            
            if metric == 'acc':
                ax.legend(loc='lower left') 
            elif metric == 'GenEntropyIndex' and ds == 'OULADsocial':
                ax.legend(loc='lower left')
            else :
                ax.legend(loc='best')
            #ax.legend(prop={'size':10}, loc='lower left') #,  bbox_to_anchor=(1, 0.87))

            if title is None:
                plt.title(metric+" values wrt train set bias level :\n ("+ds+", unbiased test set)" , fontsize=14)
            elif title != '' :
                plt.title(title, fontsize=14)

            if path_start is not None :
                    plt.savefig(path_start+ds+"_"+metric+".pdf", format="pdf", bbox_inches="tight") # dpi=1000) #dpi changes image quality
            if(display) :
                plt.show()
            plt.close()