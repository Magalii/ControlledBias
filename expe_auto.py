from datetime import timedelta
import time
import gc
import sys 
sys.path.append('../')

import expe_pipeline as expe
import plotting as plot
import analyzing as a

# Likely nothing to change here
start = time.perf_counter()
stop = start
bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
path_start = "data/"
results_path = path_start+"Results/"
plot_path = "plots/"
save = True
display_plot = False

#Adapt what elements the experiment should be performed on
datasets = ['student'] # Wole list of datasets : ['studentBalanced','student','OULADsocial','OULADstem']
biases = ['label','selectDoubleProp','selectLow']
preproc_methods = ['','massaging', 'reweighting'] # '' for no pre-proc
postproc_methods = ['eqOddsProc','calEqOddsProc','ROC-SPD'] # Other post-procs : 'ROC-EOD','ROC-AOD' #Empty list for no post-proc 
classifiers = ['RF'] #Wole list of training models : ['tree','RF','neural']

#What (sub)part of the experiment should be performed
expe_comp = True # Main experiment
expe_select = True # Expe focused on selection bias
run_expe = True
compute_results = True
plot_results = False # Plotting for expe_comp can only work if results for all postproc_methods have been computed ('eqOddsProc','calEqOddsProc','ROC-SPD','ROC-EOD' and 'ROC-AOD')

start = time.perf_counter()

###########################
### Run all experiments ###
###########################
if expe_comp :
    """
    print("\n ### Pre-proc : Start experiment ###")
    preproc_list = preproc_methods
    postproc_list = []
    blind_model = [False]
    if run_expe :
        expe.run_expe(datasets,biases,bias_levels,preproc_list,postproc_list,classifiers,blind_model,path_start=path_start)
        gc.collect()
    if compute_results :
        print("\n ### Pre-proc : Start computing results ###")
        a.compute_all(datasets,biases,preproc_list,postproc_list,classifiers,blind_model,path_start=path_start)
        gc.collect()
    inter = time.perf_counter()
    print("preproc finished after "+ str(timedelta(seconds=inter-start)))

    print("\n ### FTU : Start experiment ###") #ftu corresponds to no preproc + blind model
    preproc_list = ['']
    postproc_list = []
    blind_model = [True]
    if run_expe :
        expe.run_expe(datasets,biases,bias_levels,preproc_list,postproc_list,classifiers,blind_model,path_start=path_start)
        gc.collect()
    if compute_results :
        print("\n ### FTU : Start computing results ###")
        a.compute_all(datasets,biases,preproc_list,postproc_list,classifiers,blind_model,path_start=path_start)
        gc.collect()
    inter = time.perf_counter()
    print("FTU finished after "+ str(timedelta(seconds=inter-start)))
    """
    print("\n ### Post-proc : Start experiment ###")
    preproc_list = ['']
    postproc_list = postproc_methods
    blind_model = [False]
    if run_expe :
        expe.run_expe(datasets,biases,bias_levels,preproc_list,postproc_list,classifiers,blind_model,path_start=path_start)
        gc.collect()
    if compute_results :
        print("\n ### Post-proc : Start computing results ###")
        a.compute_all(datasets,biases,preproc_list,postproc_list,classifiers,blind_model,path_start=path_start)
        gc.collect()
    inter = time.perf_counter()
    print("postproc finished after "+ str(timedelta(seconds=inter-start)))

    if plot_results :
        print("\n ### Plotting : Start producing bar graphs encompassing all experiment results ###")
        plot_style = 'FILLED_STDEV'
        metrics_list = ['acc','StatParity','EqqOddsDiff','GenEntropyIndex','BCC']
        title = '' # None for default title, '' for no title
        plot.bargraph_all_methods(retrieval_path=results_path, dataset_list=datasets, metric_list=metrics_list, bias_list=biases, preproc_list=preproc_methods, postproc_list=postproc_methods,
                                plot_style=plot_style, path_start=plot_path, title=title)

plot_results = False
if expe_select :
    #Results RF for selection bias 0to1
    bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    biases = ['selectDoubleProp','selectPrivNoUnpriv','selectLow', 'selectRandom']
    preproc_methods = ['']
    postproc_methods = []
    blind_model = [False] 
    classifiers = ['RF'] #['tree','RF','neural']
    path_start = "data/"
    datasets = ['student'] # Wole list of datasets : ['studentBalanced','student','OULADstem','OULADsocial','OULADsocialHarder','OULADstemHarder']
    if run_expe :
        expe.run_expe(datasets,biases,bias_levels,preproc_methods,postproc_methods,classifiers,blind_model,path_start=path_start+"0to1_")
    if compute_results :
        res = a.compute_all(datasets,biases,preproc_methods,postproc_methods,classifiers,blind_model,path_start=path_start,to1=True)
    if plot_results :
        results_path = path_start+"Results/0to1_"
        plot.plot_metrics_all(datasets,biases,preproc_methods,postproc_methods,classifiers,blind_model,bias_levels,title=None,path_start=path_start,to1=True)

inter = time.perf_counter()
print("Total expe finished after "+ str(timedelta(seconds=inter-start)))
