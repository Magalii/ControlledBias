"""
    Contains different code snippets for further results analysis
"""

from datetime import timedelta
from pandas import DataFrame
import pickle
import time
import gc
import sys 
sys.path.append('../')
sys.path.append('parent_aif360')

import analyzing as a
import plotting as plot

start = time.perf_counter()
stop = start
bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #, 1]
k = -1 #negative value will create error instead of silent mistake
path_start = "data/"
blind = False

datasets = ['studentBalanced','student','OULADsocial','OULADstem'] 
biases = ['selectLow','selectDoubleProp','label'] #,'selectRandom'] 
preproc_methods = ['','reweighting','massaging']
#ftu corresponds to no preproc + blind model
postproc_methods = ['calEqOddsProc','eqOddsProc','ROC-EOD','ROC-AOD','ROC-SPD'] #For no postproc, [''] for compute_all and [] for plots

classifiers = ['RF','tree','neural']
blind_model =  [False] #,True] #['Aware'] #,'Blinded'] #,'Aware']
biased = [False]

metrics_list = ['acc','StatParity','EqqOddsDiff'] #,'GenEntropyIndex','BCC'] #','BlindCons'
results_path = path_start+"Results/"
plot_path = "plots/"
plot_style = 'FILLED_STDEV'
prefix = '0to1_'

"""
 Place your code snippet here.
"""

exit()

#########################################
# Only code above here will be executed #
#########################################


#a.compute_all(datasets,biases,preproc_methods,postproc_methods,classifiers,blind_model,path_start=path_start)
#plot.plot_all_tradeoff(results_path,biases, metrics_list, datasets, models_list,plot_style=plot_style, path_start=plot_path)
#plot.bargraph(retrieval_path=results_path, dataset_list=datasets, metric_list = metrics_list, bias_list=biases,plot_style=plot_style,path_start=plot_path, title='')


#Retrieve nk_split datasets
path = path_start+'student'+'_'+'label'
with open(path+"_nbias_splits_datasets.pkl","rb") as file:
    nk_folds_dict = pickle.load(file)
for n in nk_folds_dict :
    print(nk_folds_dict[n][0])


#Get info on biased datasets, for each folds
for ds in datasets :
    for bias in biases :
        path = path_start+ds+'_'+bias
        with open(path+"_nbias_splits_datasets.pkl","rb") as file:
            nk_folds_dict = pickle.load(file)
        for b in nk_folds_dict.keys() :
            print("## bias "+str(b)+" :")
            for f in range(len(nk_folds_dict[b])) :
                data_info = a.dataset_info(nk_folds_dict[b][f])
                print("-- fold "+str(f)+" : ")
                print("base_rate        :  "+str(data_info['base_rate']))
                print("priv_base_rate   :  "+str(data_info['priv_base_rate']))
                print("unpriv_base_rate :  "+str(data_info['unpriv_base_rate']))
                """
                print("num_privileged    :  "+str(data_info['num_privileged']))
                print("priv_num_positive :  "+str(data_info['priv_num_positive']))
                print("priv_num_negative :  "+str(data_info['priv_num_negative']))
                print("num_unprivileged  :  "+str(data_info['num_unprivileged']))
                print("unpriv_num_positive :  "+str(data_info['unpriv_num_positive']))
                print("unpriv_num_negative :  "+str(data_info['unpriv_num_negative']))
                """


#Get info on biased datasets, average for all folds
for ds in datasets :
    for bias in biases :
        path = path_start+prefix+ds+'_'+bias
        with open(path+"_nbias_splits_datasets.pkl","rb") as file:
            nk_folds_dict = pickle.load(file)
        for b in nk_folds_dict.keys() :
            k = len(nk_folds_dict[b])
            num_priv = 0
            num_unpriv = 0
            base_rate = 0
            priv_base_rate = 0
            unpriv_base_rate = 0
            spd = 0
            for f in range(k) :
                data_info = a.dataset_info(nk_folds_dict[b][f])
                num_priv += data_info['num_privileged']
                num_unpriv += data_info['num_unprivileged']
                base_rate += data_info['base_rate']
                priv_base_rate += data_info['priv_base_rate']
                unpriv_base_rate += data_info['unpriv_base_rate']
                spd += data_info['SP_diff']
            base_rate = base_rate/k
            priv_base_rate = priv_base_rate/k
            unpriv_base_rate = unpriv_base_rate/k
            spd = spd/k
            num_priv = num_priv/k
            num_unpriv = num_unpriv/k
            print("## bias "+str(b)+" :")
            print("base_rate        :  "+str(base_rate))
            print("priv_base_rate   :  "+str(priv_base_rate))
            print("unpriv_base_rate :  "+str(unpriv_base_rate))
            print("   spd           : "+str(spd))
            print("num_privileged    :  "+str(num_priv))
            print("num_unprivileged    :  "+str(num_unpriv))

#Count number of times post-processing methods fail and give indication of predictions given for validation used
pref = ''
valid = '_'
preproc = ''
model = 'RF'
visibility = 'Aware'
postfix_predvalid = "_validBiasedScores_pred_all"
postfix_postproc = "-BiasedValidFairTest_n"
for ds in datasets :
    for bias in biases :
        for post_proc in postproc_methods :
            path_begin = path_start+pref+ds+'_'+bias+valid+preproc+'_'+model+visibility
            path_pred_postproc = path_begin+'_'+post_proc+postfix_postproc+".pkl"
            path_pred_valid = path_begin+postfix_predvalid+".pkl"
            with open(path_pred_postproc,"rb") as file:
                pred_postproc = pickle.load(file)
            with open(path_pred_valid,"rb") as file:
                pred_valid = pickle.load(file)
            print('\n'+path_pred_postproc + ' :')
            for b in pred_postproc['pred'].keys() :
                print("## bias "+str(b)+" :")
                #info_valid = {}
                for f in pred_postproc['pred'][b].keys() :
                    info_valid = a.pred_info(pred_valid['pred'][b][f],pred_valid['orig'][b][f]['valid']) # Predictions of validation test evaluated on the original validation test 
                    if pred_postproc['pred'][b][f] is None : status = "Failure"
                    else : status = "Success"
                    zero = False
                    for i in info_valid['ConfMatrUnpriv'].keys():
                        if info_valid['ConfMatrUnpriv'][i] == 0 :
                            zero = True
                    if not zero :
                        for i in info_valid['ConfMatrPriv'].keys():
                            if info_valid['ConfMatrPriv'][i] == 0 :
                                zero = True
                    if zero :
                        print("-- fold "+str(f)+" : "+status)
                        print("Confusion matrices :")
                        print("ConfMatrAll :  "+str(info_valid['ConfMatrAll']))
                        print("ConfMatrUnpriv :  "+str(info_valid['ConfMatrUnpriv']))
                        print("ConfMatrPriv: "+str(info_valid['ConfMatrPriv']))

# Look at confusion matrix for a preprocessed model
pref = ''
valid = '_'
model = 'RF'
visibility = 'Aware'
for ds in datasets :
    for bias in biases :
        for preproc in preproc_methods :
            path_preproc = path_start+pref+ds+'_'+bias+valid+preproc+'_'+model+visibility+"_pred_all.pkl"
            with open(path_preproc,"rb") as file:
                pred_preproc = pickle.load(file)
            print('\n'+path_preproc + ' :')
            for b in pred_preproc['pred'].keys():
            #b = 0
                print("## bias "+str(b)+" :")
                #info_valid = {}
                for f in pred_preproc['pred'][b].keys() :
                    info_valid = a.pred_info(pred_preproc['orig'][b][f]['test'],pred_preproc['pred'][b][f]) 
                    print("-- fold "+str(f)+" :")
                    #print("Confusion matrices :")
                    print("ConfMatrAll :  "+str(info_valid['ConfMatrAll']))
                    print("ConfMatrUnpriv :  "+str(info_valid['ConfMatrUnpriv']))
                    print("ConfMatrPriv: "+str(info_valid['ConfMatrPriv']))

#Compare difference of value between BCC and Consistency
blind_model =  ['Aware']
for ds in datasets :
    for bias in biases :
        for preproc in preproc_methods :
            for model in classifiers :
                for visibility in blind_model :
                    path = path_start+"Results/"+ds+'_'+bias+'_'+preproc+'_'+model+visibility+"__metricsForPlot.pkl"
                    with open(path,"rb") as file:
                        pred = pickle.load(file)
                    print(path)
                    for i in range(0,len(pred['acc']['mean'])) :
                        diff = pred['BlindCons']['mean'][i]-pred['BCC']['mean'][i]
                        print(f"{diff:.3f}" + '     ' + path)


# Compute proportion of sens. attr. usage in tree based classifiers
biases = ['selectPrivNoUnpriv'] #['selectRandom','selectDoubleProp','selectLow'] #['label','selectDoubleProp', #['label', 'labelDouble', 'selectDouble','selectLow'] #['label','selectDouble','selectLow'] #,'selectDouble','selectLow' label
datasets = ['student']
preproc_methods = ['']
postproc_methods = []
classifiers = ['RF']
blind_model =  [False]
bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
clas_type = 'RF'
trees_count = a.sens_attr_usage(clas_type,datasets,biases,blind_model,bias_levels,to1=True,recompute_path="data/")


#Create file with textual representations of trees (for either decision trees or RF)
biases = ['selectRandom','selectDoubleProp','selectLow']
blind = [False]
to1 = True
a.txt_trees_all('RF',datasets,biases,path_retrieval="data/", to1=to1)       

# Plot results to analyze effect of different selection bias on unmitigated models
biases = ['selectRandom','selectDoubleProp','selectLow'] #['label','selectDoubleProp', #['label', 'labelDouble', 'selectDouble','selectLow'] #['label','selectDouble','selectLow'] #,'selectDouble','selectLow' label
preproc_methods = ['']
postproc_methods = []
classifiers = ['tree']
blind_model =  [False]
bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
results_path = path_start+"Results/0to1_"
plot.plot_metrics_all(datasets,biases,preproc_methods,postproc_methods,classifiers,blind_model,bias_levels,title=None,path_start=path_start,to1=True)


#Plot metrics for tradeoff analysis
title = ''
plot.plot_all_tradeoff(results_path,biases,metrics_list,datasets,classifiers,plot_style,plot_path)

#Inspect results at specific steps of experiments
path_metricsForPlot = "data/Results/0to1_student_selectRandom__RFAware__metricsForPlot.pkl"
path_metrics_all = "data/Results/0to1_student_selectRandom__RFAware__metrics_all.pkl"
path_pred = "data/0to1_student_selectRandom__RFAware_biasedTest_pred_all.pkl"
path_biasedDataset = "data/0to1_student_selectRandom_nBiasDatasets.pkl"
with open(path_metricsForPlot,"rb") as file:
    data = pickle.load(file)
print(data)

#All bar graphs for paper
plot.bargraph_all_methods(retrieval_path=results_path, dataset_list=datasets, metric_list=metrics_list, bias_list=biases, preproc_list=preproc_methods, postproc_list=postproc_methods,
                                plot_style=plot_style, path_start=plot_path, title='')


# Plot metrics with biased test and unbiased test
title = ''
for bias_type in biases :
    plot.bias_vs_unbiased(results_path, bias_type, datasets, model_list=classifiers, postproc_list = ['eqOddsProc', 'calEqOddsProc','ROC-SPD','ROC-EOD','ROC-AOD'], medium='slide',title=title, path_start=plot_path)
#plot.biasing_unmitig_models(results_path, datasets, metrics_list, biases, classifiers, bias_levels, medium='slide',title=title, path_save=plot_path, to1=True)


#Information on Consistency related metrics for all datasets
for data_name in datasets :
    with open(path_start+data_name+"_dataset","rb") as file:
        dataset = pickle.load(file)
    metrics = a.dataset_info(dataset)
    #metrics = metrics['Consistency','BlindCons','Cons_mine','BCC','BCC_penalty']
    const_metrics = [metrics[key] for key in ['Consistency','BlindCons','Cons_mine','BCC','BCC_penalty']]
    print(data_name + "['Consistency','BlindCons','Cons_mine','BCC','BCC_penalty']")
    print(const_metrics)

#Information on dataset metrics for all datasets
for data_name in datasets :
    with open(path_start+data_name+"_dataset","rb") as file:
        dataset = pickle.load(file)
    metrics = a.dataset_info(dataset)
    #metrics = metrics['Consistency','BlindCons','Cons_mine','BCC','BCC_penalty']
    print(data_name)
    numbers = [metrics[key] for key in ['num_instances','num_privileged','num_unprivileged']]
    base_rates = [metrics[key] for key in ['base_rate','priv_base_rate','unpriv_base_rate']]
    discrim = [metrics[key] for key in ['SP_diff','DP']]
    print("['num_instances','num_privileged','num_unprivileged']")
    print(numbers)
    print("['base_rate','priv_base_rate','unpriv_base_rate']")
    print(base_rates)
    print("['SP_diff','DP']")
    print(discrim)

#Compare massaging with Naive Bayes and with Random Forest
preproc_methods = ['massaging','massagingRF']
plot.bargraph_adaptable(retrieval_path=results_path, dataset_list=datasets, metric_list=metrics_list, bias_list=biases,
                  preproc_list=preproc_methods, postproc_list = postproc_methods, all_bias=bias_levels, title = None, path_start=plot_path)


#def compare_labelBias(datasets,preproc_methods,classifiers,blind_model,path_start):
for ds in datasets :
    #for bias in biases :
    for preproc in preproc_methods :
        for model in classifiers :
            for visibility in blind_model :
                path_label = path_start+"Results/"+ds+'_label_'+preproc+'_'+model+visibility+"_metricsForPlot.pkl"
                path_labelDouble = path_start+"Results/"+ds+'_labelDouble_'+preproc+'_'+model+visibility+"_metricsForPlot.pkl"
                with open(path_label,"rb") as file:
                    pred_label = pickle.load(file)
                with open(path_labelDouble,"rb") as file:
                    pred_labelDouble = pickle.load(file)
                print(ds+'_'+preproc+'_'+model)
                for i in range(0,len(pred_label['acc']['mean'])) :
                    print("         label      labelDouble     ")
                    print("acc    "+f"{pred_label['acc']['mean'][i]:.3f}"+"        "+f"{pred_labelDouble['acc']['mean'][i]:.3f}")
                    print("Cons   "+f"{pred_label['Consistency']['mean'][i]:.3f}"+"        "+f"{pred_labelDouble['Consistency']['mean'][i]:.3f}")
                    print("Entrop "+f"{pred_label['GenEntropyIndex']['mean'][i]:.3f}"+"        "+f"{pred_labelDouble['GenEntropyIndex']['mean'][i]:.3f}")

# Check if the following statement is true : "With self-selection, biased SPD overestimates fairness, while malicious selection leads to underestimated SPD values.
#Retrieve all necessary info
metrics_all_biased = {}
metrics_all_unbiased = {}
biases = ['selectLow','selectDoubleProp']
for bias in biases :
    metrics_all_biased[bias] = {}
    metrics_all_unbiased[bias] = {}
    for ds in datasets :
        metrics_all_biased[bias][ds] = {}
        metrics_all_unbiased[bias][ds] = {}
        for model in classifiers :
            metrics_all_biased[bias][ds][model] = {}
            metrics_all_unbiased[bias][ds][model] = {}
            # retrieve results for active preprocessing methods
            for preproc in preproc_methods :
                path_biased = results_path+ds+'_'+bias+'_'+preproc+'_'+model+"Aware__Biased_metricsForPlot.pkl"
                with open(path_biased,"rb") as file:
                    metrics_biased = pickle.load(file)
                path_unbiased = results_path+ds+'_'+bias+'_'+preproc+'_'+model+"Aware__metricsForPlot.pkl"
                with open(path_unbiased,"rb") as file:
                    metrics_unbiased = pickle.load(file)
                metrics_all_biased[bias][ds][model][preproc] = metrics_biased
                metrics_all_unbiased[bias][ds][model][preproc] = metrics_unbiased
            # retrieve results for FTU
            path_biased = results_path+ds+'_'+bias+'__'+model+"Blinded__Biased_metricsForPlot.pkl"
            with open(path_biased,"rb") as file:
                metrics_biased = pickle.load(file) 
            path_unbiased = results_path+ds+'_'+bias+'__'+model+"Blinded__metricsForPlot.pkl"
            with open(path_unbiased,"rb") as file:
                metrics_unbiased = pickle.load(file)
            metrics_all_biased[bias][ds][model]['FTU'] = metrics_biased
            metrics_all_unbiased[bias][ds][model]['FTU'] = metrics_unbiased
            # retrieve results for postprocessing methods
            for postproc in postproc_methods :
                path_biased = results_path+ds+'_'+bias+'__'+model+"Aware_"+postproc+"-BiasedValidBiasedTest_metricsForPlot.pkl"
                with open(path_biased,"rb") as file:
                    metrics_biased = pickle.load(file)
                path_unbiased = results_path+ds+'_'+bias+'__'+model+"Aware_"+postproc+"-BiasedValidFairTest_metricsForPlot.pkl"
                with open(path_unbiased,"rb") as file:
                    metrics_unbiased = pickle.load(file)
                metrics_all_biased[bias][ds][model][postproc] = metrics_biased
                metrics_all_unbiased[bias][ds][model][postproc] = metrics_unbiased
#Very difference between biased SPD and unbiased SPD for all datasets, classifiers, and type of mitigation
proc_list = preproc_methods + ['FTU'] + postproc_methods
for bias in biases :
    print("--"+bias+"")
    count_eq, count_diff = 0, 0
    for ds in datasets :  
        for model in classifiers :
            for proc in proc_list :
                all_spd_biased = metrics_all_biased[bias][ds][model][postproc]['StatParity']['mean']
                all_spd_unbiased = metrics_all_unbiased[bias][ds][model][postproc]['StatParity']['mean']
                for b in range(len(all_spd_biased)) :
                    spd_biased = all_spd_biased[b]
                    spd_unbiased = all_spd_unbiased[b]
                    diff = spd_unbiased-spd_biased
                    if diff == 0 :
                        count_eq += 1
                    else :
                        if bias == 'selectLow': # Count number of times spd_biased < spd_unbiased
                            if diff > 0 : count_diff += 1
                        else : # Count number of times spd_biased > spd_unbiased
                            if diff < 0 : count_diff += 1
    print("number equality : "+str(count_eq))
    print("number unexpected diff : "+str(count_diff))
