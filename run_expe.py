"""
    Script to run the experiments presented in article "No evaluation without fair representation"
    Allows to control all the experiment settings
"""

from datetime import timedelta
import time
import gc
import sys 
sys.path.append('../')

import expe_pipeline as expe
import plotting as plot
import analyzing as a

#####################################################
# Methods to control different types of experiments #
#####################################################

def run_bias_mitig_comp(datasets:list[str], biases:list[str], preproc_methods:list[str], postproc_methods:list[str], classifiers:list[str], run_expe:bool, compute_results:bool, verbose:bool) :
    """ Run and/or compute the results for the comparison of bias mitigation methods
        WARNING : Only elements that have not already been saved on disk will be recomputed, unless you changed the setting in exep.pipeline()
        run_expe (bool) : True if the experiment should be (re)run. 
        compute_results (bool) : True if the results should be (re)computed
    """
    bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    preproc_list = preproc_methods
    postproc_list = []
    blind_model = [False]
    recompute = True #Recompute results that are already present on disk
    if run_expe :
        print("\n ### Pre-proc : Start experiment ###")
        expe.pipeline(datasets,biases,bias_levels,preproc_list,postproc_list,classifiers,blind_model,path_start,save_intermediate, verbose)
        gc.collect()
    if compute_results :
        print("\n ### Pre-proc : Start computing results ###")
        a.compute_all(datasets,biases,preproc_list,postproc_list,classifiers,blind_model,path_start,recompute=recompute)
        gc.collect()
    inter = time.perf_counter()
    print("preproc finished after "+ str(timedelta(seconds=inter-start)))

    print("\n ### FTU : Start experiment ###") #ftu corresponds to no preproc + blind model
    preproc_list = ['']
    postproc_list = []
    blind_model = [True]
    if run_expe :
        expe.pipeline(datasets,biases,bias_levels,preproc_list,postproc_list,classifiers,blind_model,path_start,save_intermediate, verbose)
        gc.collect()
    if compute_results :
        print("\n ### FTU : Start computing results ###")
        a.compute_all(datasets,biases,preproc_list,postproc_list,classifiers,blind_model,path_start=path_start,recompute=recompute)
        gc.collect()
    inter = time.perf_counter()
    print("FTU finished after "+ str(timedelta(seconds=inter-start)))

    print("\n ### Post-proc : Start experiment ###")
    preproc_list = ['']
    postproc_list = postproc_methods
    blind_model = [False]
    if run_expe :
        expe.pipeline(datasets,biases,bias_levels,preproc_list,postproc_list,classifiers,blind_model,path_start,save_intermediate, verbose)
        gc.collect()
    if compute_results :
        print("\n ### Post-proc : Start computing results ###")
        a.compute_all(datasets,biases,preproc_list,postproc_list,classifiers,blind_model,path_start=path_start,recompute=recompute)
        gc.collect()
    inter = time.perf_counter()
    print("postproc finished after "+ str(timedelta(seconds=inter-start)))

def plot_bias_mitig_comp(metrics_to_plot:list[str], datasets:list[str], biases:list[str], preproc_methods:list[str], postproc_methods:list[str], plot_results:bool):
    """ Plot bargraphs to compare the performance of bias mitigation methods
        plot_results (str) : plot experiment graphs according to desired format
                             None for default, 'article' for article presentation
    """
    print("\n ### Plotting : Start producing bar graphs encompassing all experiment results ###")
    plot_style = 'FILLED_STDEV'
    title = '' # None for default title, '' for no title
    plot.bargraph_all_methods(retrieval_path=results_path, dataset_list=datasets, metric_list=metrics_to_plot, bias_list=biases, preproc_list=preproc_methods, postproc_list=postproc_methods,
                                medium=plot_results, plot_style=plot_style, path_start=plot_path, title=title)

def run_bias_on_model(biases:list[str], datasets:list[str], classifiers, path_start:str, run_expe:bool, compute_results:bool, plot_results:str, verbose:bool) :
    """ Experiment to analyze the impact of bias on unmitigated models
        run_expe (bool) : run experiment. WARNING : Only elements that have not already been saved on disk will be recomputed, unless you changed the setting in exep.pipeline()
        compute_results (bool) : compute the experiment results based on models' predictions
        plot_results (str) : plot experiment graphs according to desired format
                             'no' for no plot, None for default, 'article' for article presentation
    """
    #Results RF for selection bias 0to1
    bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # selectRandomWhole, 
    preproc_methods = ['']
    postproc_methods = []
    blind_model = [False]
    recompute = True
    if run_expe :
        expe.pipeline(datasets,biases,bias_levels,preproc_methods,postproc_methods,classifiers,blind_model,path_start=path_start+"0to1_",save_intermediate=save_intermediate, verbose=verbose)
    if compute_results :
        a.compute_all(datasets,biases,preproc_methods,postproc_methods,classifiers,blind_model,path_start=path_start,to1=True,recompute=recompute)
    if plot_results != 'no' :
        #results_path = path_start+"Results/0to1_"
        plot.plot_metrics_all(datasets,biases,preproc_methods,postproc_methods,classifiers,blind_model,bias_levels,medium=plot_results,plot_biased=False,title=None,path_start=path_start,to1=True)


#################################
# Script to perform experiments #
#################################
"""
    [Bias type] -> [string that represents it]
    Label bias -> 'label'
    Random selection -> 'selectRandom'
    Self-selection -> 'selectLow'
    Malicious selection -> 'selectDoubleProp'
    Malicious selection + exclusion of unprivileged group from train set -> 'selectPrivNoUnpriv'
    Undersampling on whole dataset -> 'selectRandomWhole'
"""
path_start = "data/"
results_path = path_start+"Results/"
plot_path = "plots/"

#What experiment should be performed ?
expe_select = True # Impact of bias on models
expe_comp = True # Comparison of mitigation methods
#What subpart(s) of the above experiments should be performed ?
run_expe = True #Run the experiment (training + predictions)
compute_results = True #Compute metrics from predictions
plot_results = "no" # "default" for default, "no" for no plot, "article" for article presentation
#What additional graphs should be plotted ?
plot_biasVSunbiased = False #Presenting fair and biased measurement of metrics in same graph
plot_tradeoff = False #Presenting fair and biased evaluation of different (un)mitigated models in seperate graphs

#How much intermediate result do you want to save on disk ?
save_intermediate = 'intermediate'
# Saving all the results requires around 75Gb of disk space
# 'no' -> only save final predictions (Not recommened, will require to repeat a lot of computation)
# 'dataset_only' -> only save original datasets but not their biased version (~37Mb)
# 'minimal' -> save the original and biased version of the datasets (~1,3Gb)
# 'intermediate' -> save all datasets versions and their splits.(RECOMMENDED) (~6Gb) (train-test 4,5)
# 'all' -> save all intermadiate results (WARNING requires over 150GB disk space)

verbose = True

start = time.perf_counter()
stop = start

# Impact of bias on models
if expe_select :
    print("#### Experiment bias in unmitigated models ####")
    classifiers = ['RF']
    # All bias types for original datasets
    biases =   ['label','selectRandom','selectLow','selectDoubleProp','selectPrivNoUnpriv']
    datasets = ['OULADstem','OULADsocial','student']
    run_bias_on_model(biases, datasets, classifiers, path_start, run_expe, compute_results, plot_results, verbose)
    # Explore selection bias in other circumstances
    biases =   ['selectRandom','selectLow','selectDoubleProp','selectPrivNoUnpriv']
    datasets = ['studentBalanced','OULADstemHarder','OULADsocialHarder']
    run_bias_on_model(biases, datasets, classifiers, path_start, run_expe, compute_results, plot_results, verbose)
    # Effect of mere size reduction on StudentBalanced
    biases = ['selectRandomWhole']
    datasets = ['studentBalanced']
    run_bias_on_model(biases, datasets, classifiers, path_start, run_expe, compute_results, plot_results, verbose)

#Comparison of mitigation methods performance
if expe_comp :
    print("#### Experiment bias mitigation comparison ####")
    datasets = ['OULADsocial','OULADstem','studentBalanced']
    biases = ['label','selectDoubleProp','selectLow']
    preproc_methods = ['','massaging', 'reweighting'] # '' for no pre-proc
    postproc_methods = ['eqOddsProc','calEqOddsProc','ROC-SPD','ROC-EOD','ROC-AOD'] #Empty list for no post-proc 
    classifiers =  ['tree','RF','neural']
    if run_expe or compute_results :
        run_bias_mitig_comp(datasets, biases, preproc_methods, postproc_methods, classifiers, run_expe, compute_results, verbose)
    if plot_results != "no":
        metrics_to_plot = ['acc','StatParity','EqqOddsDiff','GenEntropyIndex','BCC'] 
        plot_bias_mitig_comp(metrics_to_plot, datasets, biases, preproc_methods, postproc_methods, plot_results)

# Plot comparison of metric results in biased vs fair evaluation
if plot_biasVSunbiased :
    print("#### Plot metric results in biased vs fair evaluation ####")
    biases = ['label','selectDoubleProp','selectLow','selectRandom']
    datasets = ['OULADsocial','OULADstem','studentBalanced', 'student']
    classifiers =  ['tree','RF','neural']
    title = ''
    medium='article'
    for bias_type in biases :
        plot.bias_vs_unbiased(results_path, bias_type, datasets, model_list=classifiers, medium=plot_results,title=title, path_start=plot_path)
        #by default, all mitigated + unmitigated models are included

#Plot results of (un)mitigated models on graphs with fair evaluation and equivalent graph with biased evaluation
if plot_tradeoff :
    print("#### Plot (un)mitigated models in biased vs fair evaluation ####")
    title = ''
    datasets =  ['studentBalanced'] #['OULADsocial','OULADstem', 'studentBalanced']
    classifiers =  ['RF']
    #label bias
    biases = ['label']
    metrics_to_plot = ['acc','EqqOddsDiff']
    plot.plot_all_tradeoff(results_path,biases,metrics_to_plot,datasets,classifiers,plot_path=plot_path)
    #selection bias
    biases = ['selectDoubleProp','selectLow','selectRandom']
    metrics_to_plot = ['acc','StatParity']
    plot.plot_all_tradeoff(results_path,biases,metrics_to_plot,datasets,classifiers,plot_path=plot_path)


inter = time.perf_counter()
print("All experiments finished after "+ str(timedelta(seconds=inter-start)))

exit()
