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

def run_bias_on_model(biases:list[str], datasets:list[str], classifiers, path_start:str, run_expe:bool, plot_results:str, verbose:bool) :
    """ Experiment to analyze the impact of bias on unmitigated models
        path_start (String) : Relative path at which (intermediate) results are saved
        run_expe (bool) : True if the experiment should be (re)run. WARNING : Only elements that have not already been saved on disk will be recomputed, unless you changed the setting in expe.pipeline()
        compute_results (bool) : compute the experiment results based on models' predictions
        plot_results (str) : plot experiment graphs according to desired format
                             'no' for no plot, 'default' for default, 'article' for article presentation
    """
    #Results RF for selection bias 0to1
    bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # selectRandomWhole, 
    preproc_methods = ['']
    postproc_methods = []
    blind_model = [False]
    if run_expe :
        pref = '0to1_'
        expe.pipeline(datasets,biases,bias_levels,preproc_methods,postproc_methods,classifiers,blind_model,path_start,pref,save_intermediate=save_intermediate, verbose=verbose)
    if plot_results != 'no' :
        pref = '' #pref is handled by to1=True
        plot.plot_metrics_all(datasets,biases,preproc_methods,postproc_methods,classifiers,blind_model,bias_levels,medium=plot_results,plot_biased=False,title=None,path_start=path_start,to1=True)

def run_bias_mitig_comp(datasets:list[str], biases:list[str], preproc_methods:list[str], postproc_methods:list[str], classifiers:list[str], path_start:str, verbose:bool) :
    """ Run and/or compute the results for the comparison of bias mitigation methods
        WARNING : Only elements that have not already been saved on disk will be recomputed, unless you changed the setting in expe.pipeline()
        path_start (String) : Relative path at which (intermediate) results are saved
    """
    bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    preproc_list = preproc_methods
    postproc_list = []
    blind_model = [False]
    pref = '' #prefix for file names
    
    print("\n### Pre-proc : Start experiment ###")
    expe.pipeline(datasets,biases,bias_levels,preproc_list,postproc_list,classifiers,blind_model,path_start,pref,save_intermediate, verbose)
    gc.collect()
    inter = time.perf_counter()
    print("preproc finished after "+ str(timedelta(seconds=inter-start)))
    
    print("\n### FTU : Start experiment ###") #ftu corresponds to no preproc + blind model
    preproc_list = ['']
    postproc_list = []
    blind_model = [True]
    expe.pipeline(datasets,biases,bias_levels,preproc_list,postproc_list,classifiers,blind_model,path_start,pref,save_intermediate, verbose)
    gc.collect()
    inter = time.perf_counter()
    print("FTU finished after "+ str(timedelta(seconds=inter-start)))

    print("\n### Post-proc : Start experiment ###")
    preproc_list = ['']
    postproc_list = postproc_methods
    blind_model = [False]
    expe.pipeline(datasets,biases,bias_levels,preproc_list,postproc_list,classifiers,blind_model,path_start,pref,save_intermediate, verbose)
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
expe_select = False # Impact of bias on models
expe_comp = True # Comparison of mitigation methods
run_expe = True # False to avoid running expe (usefull to merely plot results)
#What graphs should be plotted ?
plot_results = "default" # Graphs for expe_select and expe_comp. "default" for default, "no" for no plot, "article" for article presentation
plot_biasVSunbiased = True #Presenting fair and biased measurement of metrics in same graph
plot_eval_comp = True #Presenting fair and biased evaluation of different (un)mitigated models in seperate graphs

#How much intermediate result do you want to save on disk ?
save_intermediate = 'intermediate'
# 'no' -> only save final results (Not recommened, will require to repeat a lot of computation + doesn't allow to compute the sensitive attribute usage)
# 'minimal' -> save the original and biased version of the datasets + unmitigated trained models (~3,5Gb)
# 'intermediate' -> save all datasets versions and their splits + unmitigated trained models(RECOMMENDED) (~8Gb)
# 'all' -> save all intermadiate results (WARNING requires over 150GB disk space)

verbose = False

start = time.perf_counter()

# Impact of bias on models
if expe_select :
    print("\n\n#### Experiment bias in unmitigated models ####")
    classifiers = ['RF','tree','neural']
    # All bias types for original datasets
    biases =   ['label','selectRandom','selectLow','selectDoubleProp','selectPrivNoUnpriv']
    datasets = ['student','OULADstem','OULADsocial']
    run_bias_on_model(biases, datasets, classifiers, path_start, run_expe, plot_results, verbose)
    # Explore selection bias in other circumstances
    biases =   ['selectRandom','selectLow','selectDoubleProp','selectPrivNoUnpriv']
    datasets = ['studentBalanced','OULADstemHarder','OULADsocialHarder']
    run_bias_on_model(biases, datasets, classifiers, path_start, run_expe, plot_results, verbose)
    # Effect of mere size reduction on StudentBalanced
    biases = ['selectRandomWhole']
    datasets = ['studentBalanced']
    run_bias_on_model(biases, datasets, classifiers, path_start, run_expe, plot_results, verbose)
    
#Comparison of mitigation methods performance
if expe_comp :
    print("\n\n#### Experiment bias mitigation comparison ####")
    datasets = ['studentBalanced', 'OULADstem', 'OULADsocial', 'student']
    biases = ['label','selectDoubleProp','selectLow','selectRandom']
    preproc_methods = ['','massaging', 'reweighting'] # '' for no pre-proc
    postproc_methods = ['eqOddsProc','calEqOddsProc','ROC-SPD','ROC-EOD','ROC-AOD'] #Empty list for no post-proc 
    classifiers = ['tree','RF','neural']
    if run_expe :
        run_bias_mitig_comp(datasets, biases, preproc_methods, postproc_methods, classifiers, path_start, verbose)
    if plot_results != "no":
        datasets = ['studentBalanced', 'OULADstem', 'OULADsocial']
        biases = ['label','selectDoubleProp','selectLow']
        metrics_to_plot = ['acc','StatParity','EqqOddsDiff','GenEntropyIndex','BCC'] 
        plot_bias_mitig_comp(metrics_to_plot, datasets, biases, preproc_methods, postproc_methods, path_start, plot_results)

# Plot comparison of metric results in biased vs fair evaluation
if plot_biasVSunbiased :
    print("\n\n#### Plot metric results in biased vs fair evaluation ####")
    biases = ['label','selectDoubleProp','selectLow','selectRandom']
    datasets = ['studentBalanced','OULADsocial','OULADstem', 'student']
    classifiers =  ['tree','RF','neural']
    preproc_methods = ['','massaging', 'reweighting'] # '' for no pre-proc
    postproc_methods = ['eqOddsProc','calEqOddsProc','ROC-SPD','ROC-EOD','ROC-AOD'] #Empty list for no post-proc 
    title = ''
    medium='article'
    for bias_type in biases :
        plot.bias_vs_unbiased(results_path, bias_type, datasets, classifiers, preproc_methods, postproc_methods, medium=plot_results,title=title, path_start=plot_path)

#Plot results of (un)mitigated models on graphs with fair evaluation and equivalent graph with biased evaluation
if plot_eval_comp :
    print("\n\n#### Plot (un)mitigated models in biased vs fair evaluation ####")
    title = ''
    datasets =  ['OULADsocial','OULADstem', 'studentBalanced']
    classifiers =  ['RF','tree','neural']
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
