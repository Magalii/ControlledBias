import matplotlib.pyplot as plt
import numpy as np
import pickle
import gc
import sys
sys.path.append('../')

from aif360.datasets import BinaryLabelDataset
from sklearn import tree
colors = {'':'black', 'reweighting': "#C85200", 'massaging': "#FF800E", 'FTU':"#FFBC79",
                'eqOddsProc':"#595959", 'calEqOddsProc':"#898989", 'ROC-SPD':"#006BA4", 'ROC-EOD':"#5F9ED1", 'ROC-AOD':"#A2C8EC"}
        # Color-blind friendly color scheme :
        #'#FFBC79' Light orange/Mac and cheese; '#898989'#Suva Grey/Light Gray; '#ABABAB'#Dark gray; '#595959'#Mortar/Darker Grey
        #https://stackoverflow.com/questions/74830439/list-of-color-names-for-matplotlib-style-tableau-colorblind10      
legend = {'':'Unmitigated', 'reweighting': "Reweighting", 'massaging': "Massaging", 'FTU':"FTU",
            'eqOddsProc':"EOP", 'calEqOddsProc':"CEO", 'ROC-SPD':"ROC-SPD", 'ROC-EOD':"ROC-EOD"}
colorblindIBM = {'ultramarine': "#648FFF", 'indigo':"785EF0", 'magenta':"#DC267F", 'orange':"#FE6100", 'gold':"#FFB000",
                 'black':"#000000", 'white':"#FFFFFF"}

def plot_tree(classifier: tree.DecisionTreeClassifier, dataset: BinaryLabelDataset,save_path:str=None) :
    fig = plt.figure
    tree.plot_tree(classifier, feature_names=dataset.feature_names, class_names=dataset.label_names)
    if save_path is not None :
        plt.savefig(save_path+".pdf", format="pdf", bbox_inches="tight")

""" Draft for potential future function
def plot_nkTrees(nk_classifier, dataset, bias_type, path_start:str=None) :
#dataset.feature_names should be passed as arg instead of a BinaryLabelDataset object
    path = path_start+"Results/"+pref+ds+'_'+bias+'_'+preproc+'_'+model+visibility
    path_trees = path_start+ds+'_'+bias+'__tree'+blind+"_all.pkl"
    with open(path_trees,"rb") as file:
        nk_classifier = pickle.load(file)
    path_data = path_start+ds+"_dataset"
    with open(path_data,"rb") as file:
        dataset = pickle.load(file)
    clas = nk_classifier[n][k]
    plot_tree(clas,dataset)
"""

def plot_metrics_all(data_list, bias_list, preproc_list, postproc_list, model_list, blinding_list, all_bias, title='', path_start=None, to1=False) :
    """ Apply plot-by_bias for all combinations of datasets, bias, preproc, model and blinding given as argument
        Each graph represent one mitigation situation, plotting evolution of divers metrics with the increasing intensity of the bias
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
                        if to1 :
                            pref = "0to1_"
                        else :
                            pref = ''
                        path = path_start+"Results/"+pref+ds+'_'+bias+'_'+preproc+'_'+model+visibility
                        if len(postproc_list) == 0 :
                            with open(path+"__metricsForPlot.pkl","rb") as file:
                                metrics_for_plot = pickle.load(file)
                            with open(path+"__Biased"+"_metricsForPlot.pkl","rb") as file:
                                metricsBiased_for_plot = pickle.load(file)
                            if title == None :
                                title_biased ='Metric values wrt train set bias level\n ('+bias+' bias, '+preproc+', '+model+visibility+', biased test set)'
                                title_unbiased = 'Metric values wrt train set bias level\n ('+bias+' bias, '+preproc+', '+model+visibility+', unbiased test set)'
                            else :
                                title_biased, title_unbiased = title, title
                            plot_metrics(metricsBiased_for_plot, all_bias, bias, biased_test=True, plot_style='FILLED_STDEV', title=title_biased,path_start='plots/'+pref+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_byBias_BiasedTest', display=False)
                            plot_metrics(metrics_for_plot, all_bias, bias, biased_test=False, plot_style='FILLED_STDEV',title=title_unbiased , path_start='plots/'+pref+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_byBias_unbiasedTest', display=False)
                            
                            del metrics_for_plot, metricsBiased_for_plot
                            gc.collect()
                        else : #Need to take post-proc and there specificity into account
                            for postproc in postproc_list :
                                path = path_start+"Results/"+pref+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_'+postproc
                                print(path)
                                path_biasedValidFairTest = path+"-BiasedValidFairTest"
                                path_biasedValidBiasedTest = path+"-BiasedValidBiasedTest"
                                path_fairValidFairTest = path+"-FairValidFairTest"
                                with open(path_biasedValidFairTest+"_metricsForPlot.pkl","rb") as file: 
                                    metrics_BVFT = pickle.load(file)
                                with open(path_biasedValidBiasedTest+"_metricsForPlot.pkl","rb") as file:
                                    metrics_BVBT = pickle.load(file)
                                with open(path_fairValidFairTest+"_metricsForPlot.pkl","rb") as file:
                                    metrics_FVFT = pickle.load(file)
                                
                                if title == None :
                                    title_BVFT = title='Metric values wrt train set bias level\n ('+bias+' bias, '+model+visibility+', '+postproc+', biased postproc & fair test set)'
                                    title_BVBT = 'Metric values wrt train set bias level\n ('+bias+' bias, '+model+visibility+', '+postproc+', biased postproc & biased test set)'
                                    title_FVFT = 'Metric values wrt train set bias level\n ('+bias+' bias, '+model+visibility+', '+postproc+', fair postproc & fair test set)'
                                else :
                                    title_BVFT, title_BVBT, title_FVFT  = title, title, title
                                plot_metrics(metrics_BVFT, all_bias, bias, biased_test=False, plot_style='FILLED_STDEV', title=title_BVFT,path_start='plots/'+pref+ds+'_'+bias+'__'+model+visibility+'_'+postproc+'-BiasedValidFairTest_byBias_', display=False)
                                plot_metrics(metrics_BVBT, all_bias, bias, biased_test=True, plot_style='FILLED_STDEV',title=title_BVBT, path_start='plots/'+pref+ds+'_'+bias+'__'+model+visibility+'_'+postproc+'-BiasedValidBiasedTest_byBias_', display=False)
                                plot_metrics(metrics_FVFT, all_bias, bias, biased_test=False, plot_style='FILLED_STDEV',title=title_FVFT, path_start='plots/'+pref+ds+'_'+bias+'__'+model+visibility+'_'+postproc+'-FairValidFairTest_byBias_', display=False)
                                
                                #TODO add Ideal ground truth + fair test if studied
                                
                                #Manage memory
                                del metrics_BVFT, metrics_BVBT, metrics_FVFT
                                gc.collect()


def plot_metrics(metrics_for_plot, all_bias: list[float], bias_type:str, biased_test:bool, plot_style: str = 'SIMPLE_PLOT', title: str = '', path_start:str = None, display=True) :
    """ For one mitigation situation, plot evolution of divers metrics with the increasing intensity of the bias
    metrics_for_plot : Dictionary {str:{'mean': list[float], 'stdev':list[float]}}
        dictionary containing the metrics for that scenario, such that metrics_for_plot[metric] = {'mean': list of values, 'stdev': list of standard variation}
    all_bias : list[float]
        list of bias levels to be used as x-values, must be the same length as the list of metric values in 'metrics_for_plot'
    bias_type : str
        Whether the bias is of type 'label' or 'selection'. Used for x-axis label
    biased_test : TODO remove
    plot_style : string, optional
        None for automatic display of all metrics
        'SIMPLE_PLOT' for choice of metrics displaid without standard deviation
        'FILLED_STDEV' for choice of metrics displaid with standard deviation as colored area arround the curve
    title : str
        '' (empty string) for no title
        None for automatic title
        non-empty string to for custom title
    """

    fig, ax = plt.subplots() # (figsize=(length, heigth)) #Scale size of image
    ax.hlines(0,0,1,colors='black')

    if plot_style is None :
        for metric in metrics_keys :
            #if metric != "DP_ratio":
            plt.plot(all_bias,metrics_for_plot[metric],label = str(metric),linestyle="--",marker="o")
            plt.style.use('tableau-colorblind10')
    elif  plot_style == 'SIMPLE_PLOT' or plot_style == 'FILLED_STDEV' :
        #Mean values
        #print("all_bias :")
        #print(all_bias)
        #print("metrics_by_bias['acc']['mean'] :")
        #print(metrics_by_bias['acc']['mean'])
        ax.plot(all_bias,metrics_for_plot['acc']['mean'], label = 'Accuracy', linestyle="--",marker="o", color='#595959')##Dark gray
        ax.plot(all_bias,metrics_for_plot['accBal']['mean'], label = 'Balanced Acc.', linestyle="--",marker="^", color='#595959')##Dark gray
        ax.plot(all_bias,metrics_for_plot['ROCAUC']['mean'], label = 'ROCAUC', linestyle="--",marker="v", color='#595959')##Dark gray
        #ax.plot(all_bias,metrics_by_bias['BlindCons']['mean'], label = 'Consistency', linestyle="--",marker="p", color="#5F9ED1")#Picton blue
        ax.plot(all_bias,metrics_for_plot['BCC']['mean'], label = 'BCC', linestyle="--",marker="s", color="#5F9ED1")#Picton blue
        ax.plot(all_bias,metrics_for_plot['FPR']['mean'], label = 'FPR global', linestyle="--",marker="+", color='#ABABAB')##Light gray
        ax.plot(all_bias,metrics_for_plot['FNR']['mean'], label = 'FNR global', linestyle="--",marker="x", color='#ABABAB')##Light gray
        #ax.plot(all_bias,metrics_by_bias['F1']['mean'], label = 'F1 score', linestyle="--",marker="o", color='#595959')##Dark gray
        #ax.plot(all_bias,metrics_by_bias['EqqOddsDiff']['mean'], label = 'Eq. Odds', linestyle="--",marker="X", c="#A2C8EC")#Seil/Light blue
        #ax.plot(all_bias,metrics_by_bias['EqqOppDiff']['mean'], label = 'EqOpp=TPRdiff', linestyle="--",marker="d", c="#FF800E")#Pumpkin/Bright orange
        ax.plot(all_bias,metrics_for_plot['GenEntropyIndex']['mean'], label = 'GEI', linestyle="--",marker="p", c="#A2C8EC")#Seil/Light blue
        ax.plot(all_bias,metrics_for_plot['FalsePosRateDiff']['mean'], label = 'FPR diff.', linestyle="--",marker="P", color="#FFBC79")#Mac and cheese orange
        ax.plot(all_bias,metrics_for_plot['FalseNegRateDiff']['mean'], label = 'FNR diff.', linestyle="--",marker="X", c='#FF800E')#Pumpkin/Bright orange
        ax.plot(all_bias,metrics_for_plot['StatParity']['mean'], label = 'SPD', linestyle="--",marker="d", c='#C85200')#Tenne/Dark orange
        ax.plot(all_bias,metrics_for_plot['sens_attr_usage']['mean'], label = 'Sens. attr. usage', linestyle="--",marker=".", c='#000000')#Tenne/Dark orange

        #if trees_count is not None :
        #    ax.plot(all_bias, trees_count, label = 'Sens.attr. use', linestyle="--",marker=".", c='#000000')

        if plot_style == 'FILLED_STDEV':
        #Shade for std values
            ax.fill_between(all_bias,metrics_for_plot['acc']['mean'] - metrics_for_plot['acc']['stdev'], metrics_for_plot['acc']['mean'] + metrics_for_plot['acc']['stdev'], edgecolor = None, facecolor='#595959', alpha=0.4)
            ax.fill_between(all_bias,metrics_for_plot['accBal']['mean'] - metrics_for_plot['accBal']['stdev'], metrics_for_plot['accBal']['mean'] + metrics_for_plot['accBal']['stdev'], edgecolor = None, facecolor='#595959', alpha=0.4)
            ax.fill_between(all_bias,metrics_for_plot['ROCAUC']['mean'] - metrics_for_plot['ROCAUC']['stdev'], metrics_for_plot['ROCAUC']['mean'] + metrics_for_plot['ROCAUC']['stdev'], edgecolor = None, facecolor='#595959', alpha=0.4)
            #ax.fill_between(all_bias,metrics_by_bias['BlindCons']['mean'] - metrics_by_bias['BlindCons']['stdev'], metrics_by_bias['BlindCons']['mean'] + metrics_by_bias['BlindCons']['stdev'], edgecolor = None, facecolor='#5F9ED1', alpha=0.4)
            ax.fill_between(all_bias,metrics_for_plot['BCC']['mean'] - metrics_for_plot['BCC']['stdev'], metrics_for_plot['BCC']['mean'] + metrics_for_plot['BCC']['stdev'], edgecolor = None, facecolor='#5F9ED1', alpha=0.4)
            ax.fill_between(all_bias,metrics_for_plot['FPR']['mean'] - metrics_for_plot['FPR']['stdev'], metrics_for_plot['FPR']['mean'] + metrics_for_plot['FPR']['stdev'], edgecolor = None, facecolor='#ABABAB', alpha=0.4)
            ax.fill_between(all_bias,metrics_for_plot['FNR']['mean'] - metrics_for_plot['FNR']['stdev'], metrics_for_plot['FNR']['mean'] + metrics_for_plot['FNR']['stdev'], edgecolor = None, facecolor='#ABABAB', alpha=0.4)
            ax.fill_between(all_bias,metrics_for_plot['FalsePosRateDiff']['mean'] - metrics_for_plot['FalsePosRateDiff']['stdev'], metrics_for_plot['FalsePosRateDiff']['mean'] + metrics_for_plot['FalsePosRateDiff']['stdev'], edgecolor = None, facecolor='#FFBC79', alpha=0.4)
            #ax.fill_between(all_bias,metrics_by_bias['EqqOppDiff']['mean'] - metrics_by_bias['EqqOppDiff']['stdev'], metrics_by_bias['EqqOppDiff']['mean'] + metrics_by_bias['EqqOppDiff']['stdev'], edgecolor = None, facecolor='#FF800E', alpha=0.4)
            ax.fill_between(all_bias,metrics_for_plot['FalseNegRateDiff']['mean'] - metrics_for_plot['FalseNegRateDiff']['stdev'], metrics_for_plot['FalseNegRateDiff']['mean'] + metrics_for_plot['FalseNegRateDiff']['stdev'], edgecolor = None, facecolor='#FF800E', alpha=0.4)
            ax.fill_between(all_bias,metrics_for_plot['GenEntropyIndex']['mean'] - metrics_for_plot['GenEntropyIndex']['stdev'], metrics_for_plot['GenEntropyIndex']['mean'] + metrics_for_plot['GenEntropyIndex']['stdev'], edgecolor = None, facecolor='#A2C8EC', alpha=0.4)
            ax.fill_between(all_bias,metrics_for_plot['StatParity']['mean'] - metrics_for_plot['StatParity']['stdev'], metrics_for_plot['StatParity']['mean'] + metrics_for_plot['StatParity']['stdev'], edgecolor = None, facecolor='#C85200', alpha=0.4)
            ax.fill_between(all_bias,metrics_for_plot['sens_attr_usage']['mean'] - metrics_for_plot['sens_attr_usage']['stdev'], metrics_for_plot['sens_attr_usage']['mean'] + metrics_for_plot['sens_attr_usage']['stdev'], edgecolor = None, facecolor='#000000', alpha=0.4)

    ax.tick_params(labelsize = 'large',which='major')
    ax.set_ylim([-1.1,1.1])
    ax.set_xlim([0,1])
    #minor_ticks = np.arange(-1,1,0.05)
    #ax.set_yticks(minor_ticks) #, minor=True)
    
    if bias_type == 'label': 
        ax.set_xlabel('Bias intensity $\\beta_l$', size=14)
    else : # selection bias
        ax.set_xlabel('Bias intensity $p_u$', size=14)
    
    #ax.grid(which='both',linestyle='-',linewidth=0.5,color='lightgray')
    ax.grid(visible=True)
    ax.grid(which='minor',linestyle=':',linewidth=0.3,color='lightgray')
    ax.minorticks_on()

    #ax.legend(loc='best')
    ax.legend(prop={'size':7}, loc='best') #lower right

    #if biased_test :
    #    ax.legend(prop={'size':9}, loc='lower left') #,  bbox_to_anchor=(1, 0.87))

    if title != '':
        plt.title(title, fontsize=14)

    if path_start is not None :
            plt.savefig(path_start+".pdf", format="pdf", bbox_inches="tight") # dpi=1000) #dpi changes image quality
    if(display) :
        plt.show()
    plt.close()

#    nk_results_dic : Dictionary {float: {int: {str: float}}}
#        Nested dictionaries where nk_results_dic[b][f][metric_name] = Value of 'metric_name' obtained for fold nbr 'k' of model trained with bias level 'b'

def plot_tradeoff(retrieval_path:str, metric:str, dataset_list:list[str], model_list:list[str], bias:str,  all_bias:list[float], biased_test:bool,
                  preproc_list=['','reweighting','massaging'], postproc_list = ['eqOddsProc', 'calEqOddsProc','ROC-SPD','ROC-EOD'],
                  plot_style:str = 'FILLED_STDEV', title: str = '', path_start:str = None, display=True) :
    """ Plot evolution of several bias mitigation methods for a given metric
    bias_type : str
        Whether the bias is of type 'label' or a selection type. Used for x-axis label
    plot_style : string, optional
        'ALL' for basic display of all metrics
        'SIMPLE_PLOT' for choice of metrics displaid without standard deviation
        'FILLED_STDEV' for choice of metrics displaid with standard deviation as colored area arround the curve
    """
    # Retrieve all necessary info
    metrics_all = {}
    
    for ds in dataset_list :
        metrics_all[ds] = {}
        for model in model_list :
            metrics_all[ds][model] = {}
            # retrieve results for active preprocessing methods
            for preproc in preproc_list :
                if biased_test : biased = 'Biased_'
                else : biased = ''
                path = retrieval_path+ds+'_'+bias+'_'+preproc+'_'+model+"Aware__"+biased+"metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][model][preproc] = metrics
            # retrieve results for FTU
            path = retrieval_path+ds+'_'+bias+'__'+model+"Blinded__"+biased+"metricsForPlot.pkl"
            with open(path,"rb") as file:
                metrics = pickle.load(file)
                metrics_all[ds][model]['FTU'] = metrics
            # retrieve results for postprocessing methods
            for postproc in postproc_list :
                if biased_test : biased = "Biased"
                else : biased = "Fair"
                path = retrieval_path+ds+'_'+bias+'__'+model+"Aware_"+postproc+"-BiasedValid"+biased+"Test_metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    postproc = postproc
                    metrics_all[ds][model][postproc] = metrics
    #metrics_all[ds][model][proc][metric]['mean']
    
    fig, ax = plt.subplots() # (figsize=(length, heigth)) #Scale size of image
    #ax.hlines(0,-0.05,1,colors="#595959")

    colors = {'':'black', 'reweighting': "#C85200", 'massaging': "#FF800E", 'FTU':"#FFBC79",
              'eqOddsProc':"#006BA4", 'calEqOddsProc':"#5F9ED1", 'ROC-SPD':"#A2C8EC", 'ROC-EOD':"#595959"}
    indices = np.array(all_bias)
    #fuzzy_start = -0.0035
    fuzzy = 0.0025
    for ds in dataset_list :
        for model in model_list :
            if plot_style == 'FILLED_STDEV':
                ax.fill_between(indices,metrics_all[ds][model][''][metric]['mean'] - metrics_all[ds][model][''][metric]['stdev'], metrics_all[ds][model][''][metric]['mean'] + metrics_all[ds][model][''][metric]['stdev'], edgecolor = None, facecolor='black', alpha=0.3)
                ax.fill_between(indices,metrics_all[ds][model]['reweighting'][metric]['mean'] - metrics_all[ds][model]['reweighting'][metric]['stdev'], metrics_all[ds][model]['reweighting'][metric]['mean'] + metrics_all[ds][model]['reweighting'][metric]['stdev'], edgecolor = None, facecolor=colors['reweighting'], alpha=0.3)
                ax.fill_between(indices,metrics_all[ds][model]['massaging'][metric]['mean'] - metrics_all[ds][model]['massaging'][metric]['stdev'], metrics_all[ds][model]['massaging'][metric]['mean'] + metrics_all[ds][model]['massaging'][metric]['stdev'], edgecolor = None, facecolor=colors['massaging'], alpha=0.3)
                ax.fill_between(indices,metrics_all[ds][model]['FTU'][metric]['mean'] - metrics_all[ds][model]['FTU'][metric]['stdev'], metrics_all[ds][model]['FTU'][metric]['mean'] + metrics_all[ds][model]['FTU'][metric]['stdev'], edgecolor = None, facecolor=colors['FTU'], alpha=0.3)
                ax.fill_between(indices,metrics_all[ds][model]['eqOddsProc'][metric]['mean'] - metrics_all[ds][model]['eqOddsProc'][metric]['stdev'], metrics_all[ds][model]['eqOddsProc'][metric]['mean'] + metrics_all[ds][model]['eqOddsProc'][metric]['stdev'], edgecolor = None, facecolor=colors['eqOddsProc'], alpha=0.3)
                ax.fill_between(indices,metrics_all[ds][model]['calEqOddsProc'][metric]['mean'] - metrics_all[ds][model]['calEqOddsProc'][metric]['stdev'], metrics_all[ds][model]['calEqOddsProc'][metric]['mean'] + metrics_all[ds][model]['calEqOddsProc'][metric]['stdev'], edgecolor = None, facecolor=colors['calEqOddsProc'], alpha=0.3)
                ax.fill_between(indices,metrics_all[ds][model]['ROC-SPD'][metric]['mean'] - metrics_all[ds][model]['ROC-SPD'][metric]['stdev'], metrics_all[ds][model]['ROC-SPD'][metric]['mean'] + metrics_all[ds][model]['ROC-SPD'][metric]['stdev'], edgecolor = None, facecolor=colors['ROC-SPD'], alpha=0.3)
                ax.fill_between(indices,metrics_all[ds][model]['ROC-EOD'][metric]['mean'] - metrics_all[ds][model]['ROC-EOD'][metric]['stdev'], metrics_all[ds][model]['ROC-EOD'][metric]['mean'] + metrics_all[ds][model]['ROC-EOD'][metric]['stdev'], edgecolor = None, facecolor=colors['ROC-EOD'], alpha=0.3)
                ax.fill_between(indices,metrics_all[ds][model]['ROC-AOD'][metric]['mean'] - metrics_all[ds][model]['ROC-AOD'][metric]['stdev'], metrics_all[ds][model]['ROC-AOD'][metric]['mean'] + metrics_all[ds][model]['ROC-AOD'][metric]['stdev'], edgecolor = None, facecolor=colors['ROC-AOD'], alpha=0.3)
                
            ax.plot(indices,metrics_all[ds][model][''][metric]['mean'], label = "Unmitigated", linestyle="--", marker="o", markeredgewidth=0, c='black', alpha=1)
            ax.plot(indices-4*fuzzy,metrics_all[ds][model]['reweighting'][metric]['mean'], label = "Reweighing", linestyle="--", marker="s", markeredgewidth=0, c=colors['reweighting'], alpha=1)
            ax.plot(indices-3*fuzzy,metrics_all[ds][model]['massaging'][metric]['mean'], label = "Massaging", linestyle="--", marker="D", markeredgewidth=0, c=colors['massaging'], alpha=1)
            ax.plot(indices-2*fuzzy,metrics_all[ds][model]['FTU'][metric]['mean'], label = "FTU", linestyle="--", marker="p", markeredgewidth=0, c=colors['FTU'], alpha=1)
            ax.plot(indices-1*fuzzy,metrics_all[ds][model]['eqOddsProc'][metric]['mean'], label = "EOP", linestyle="--", marker="X", markeredgewidth=0, c=colors['eqOddsProc'], alpha=1)
            ax.plot(indices+1*fuzzy,metrics_all[ds][model]['calEqOddsProc'][metric]['mean'], label = "CEO", linestyle="--", marker="h", markeredgewidth=0, c=colors['calEqOddsProc'], alpha=1)
            ax.plot(indices+2*fuzzy,metrics_all[ds][model]['ROC-SPD'][metric]['mean'], label = "ROC-SPD", linestyle="--", marker="d", markeredgewidth=0, c=colors['ROC-SPD'], alpha=1)
            ax.plot(indices+3*fuzzy,metrics_all[ds][model]['ROC-EOD'][metric]['mean'], label = "ROC-EOD", linestyle="--", marker="^", markeredgewidth=0, c=colors['ROC-EOD'], alpha=1)
            ax.plot(indices+4*fuzzy,metrics_all[ds][model]['ROC-AOD'][metric]['mean'], label = "ROC-AOD", linestyle="--", marker="v", markeredgewidth=0, c=colors['ROC-AOD'], alpha=1)
            
            """
            for proc in metrics_all[ds][model].keys():
                if proc != '':
                    ax.plot(indices+fuzzy,metrics_all[ds][model][proc][metric]['mean'], label = proc, linestyle="--", marker="o", markeredgewidth=0, c=colors[proc], alpha=0.8)
                fuzzy_start += fuzzy
            """
                
       
    ax.tick_params(labelsize = 'large',which='major')
    #minor_ticks = np.arange(-1,1,0.05)
    #ax.set_yticks(minor_ticks) #, minor=True)
    """
    if metric == 'acc':
        ax.set_ylim([0.4,1])
    elif metric == 'StatParity':
        if bias == 'selectLow':
            ax.set_ylim([-0.1,1])
        elif bias == 'selectDoubleProp':
            ax.set_ylim([-0.8,0.5])
        elif bias == 'label':
            ax.set_ylim([-0.5,1.1])
    else : 
        if bias == 'selectLow':
            ax.set_ylim([-0.05,1])
        elif bias == 'selectDoubleProp':
            ax.set_ylim([-0.05,1])
        elif bias == 'label':
            ax.set_ylim([-0.05,1])
        #ax.set_xlim([0,1])
    """
    if metric != 'acc':
        if bias == 'label': 
            ax.set_xlabel('Bias intensity $\\beta_l$', size=14)
        else : # selection bias
            ax.set_xlabel('Bias intensity $p_u$', size=14)
    
    # Name of metric on leftmost graph in paper
    if biased_test :
        if metric == 'acc':
            y_label = "Accuracy"
        elif metric == 'StatParity':
            y_label = "Statistical Parity Diff."
        elif metric == 'EqqOddsDiff':
            y_label = "Equalized Odds Diff."
        else :
            y_label = metric
        ax.set_ylabel(y_label, size = 16)
    else :
        ax.set_ylabel(" ", size = 16)
    
    #ax.grid(which='both',linestyle='-',linewidth=0.5,color='lightgray')
    ax.grid(visible=True)
    ax.grid(which='minor',linestyle=':',linewidth=0.3,color='lightgray')
    ax.minorticks_on()

    #ax.legend(loc='best')
    #if biased_test :
    ax.legend(prop={'size':9}, loc='best') #, 'lower left' bbox_to_anchor=(1, 0.87))

    if title != '':
        plt.title(title, fontsize=14)

    if path_start is not None :
            if biased_test : test = "BiasedTest"
            else : test = "FairTest"
            path=path_start+"tradeoff_"+bias+'_'+metric+'_'+ds+'_'+model+'_'+test+".pdf"
            plt.savefig(path, format="pdf", bbox_inches="tight") # dpi=1000) #dpi changes image quality
    if(display) :
        plt.show()
    plt.close()

def plot_all_tradeoff(retrieval_path:str, bias_list:list[str], metric_list:list[str], dataset_list:list[str], model_list:list[str], plot_style:str = 'FILLED_STDEV',  path_start:str = None) :
    bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for bias_type in bias_list :
        for ds in dataset_list :
            for metric in metric_list :
                plot_tradeoff(retrieval_path,metric,[ds],model_list,bias_type,bias_levels,biased_test=False,plot_style=plot_style,path_start=path_start,display = False)
                plot_tradeoff(retrieval_path,metric,[ds],model_list,bias_type,bias_levels,biased_test=True,plot_style=plot_style,path_start=path_start,display = False)


def plot_by_metric(retrieval_path:str, dataset_list=['student','OULADstem', 'OULADsocial'], metric_list=['acc','StatParity','EqqOddsDiff','GenEntropyIndex'], bias_list=['label','selectDoubleProp'],preproc_list=['', 'reweighting','massaging'],ylim:list[float]=None, xlim:list[float]=None, all_bias:list[float]=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], plot_style: str = 'SIMPLE_PLOT', title: str = '', path_start:str = None, display=False) :
    """ Has not been used
    plot_style : string, optional
        'SIMPLE_PLOT' for choice of metrics displaid without standard deviation
        'FILLED_STDEV' for choice of metrics displaid with standard deviation as colored area arround the curve
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
        for metric in metric_list :
            fig, ax = plt.subplots() # (figsize=(length, heigth)) #Scale size of image
            ax.hlines(0,0,1,colors='black')

            if plot_style is None :
                for metric in metrics_keys :
                    #if metric != "DP_ratio":
                    plt.plot(all_bias,metrics_all[metric],label = str(metric),linestyle="--",marker="o")
                    plt.style.use('tableau-colorblind10')
            elif  plot_style == 'SIMPLE_PLOT' or plot_style == 'FILLED_STDEV' :
                #Mean values
                ax.plot(all_bias,metrics_all[ds]['label'][''][metric]['mean'], label = 'label bias', linestyle="--",marker="o", color="#595959")#Dark gray
                ax.plot(all_bias,metrics_all[ds]['label']['reweighting'][metric]['mean'], label = 'label+reweighing', linestyle="--",marker="X", color="#006BA4")#Cerulean/Blue
                ax.plot(all_bias,metrics_all[ds]['label']['massaging'][metric]['mean'], label = 'label+massaging', linestyle="--",marker="P", color="#A2C8EC")#Seil/Light blue
                ax.plot(all_bias,metrics_all[ds]['label']['FTU'][metric]['mean'], label = 'label+FTU', linestyle="--",marker="D", color="#A2C8EC")#Seil/Light blue
                ax.plot(all_bias,metrics_all[ds]['selectDoubleProp'][''][metric]['mean'], label = 'selection bias', linestyle="--",marker="s", color="#ABABAB")#Cerulean/Blue
                ax.plot(all_bias,metrics_all[ds]['selectDoubleProp']['reweighting'][metric]['mean'], label = 'select+reweighing', linestyle="--",marker="^", color="#C85200")#Tenne/Dark orange
                ax.plot(all_bias,metrics_all[ds]['selectDoubleProp']['massaging'][metric]['mean'], label = 'select+massaging', linestyle="--",marker="v", color="#FF800E")#Pumpkin/Bright orange
                ax.plot(all_bias,metrics_all[ds]['selectDoubleProp']['FTU'][metric]['mean'], label = 'select+FTU', linestyle="--",marker=">", color="#FF800E")#Pumpkin/Bright orange
                
                if plot_style == 'FILLED_STDEV':
                #Shade for std values
                    ax.fill_between(all_bias,metrics_all[ds]['label'][''][metric]['mean'] - metrics_all[ds]['label'][''][metric]['stdev'], metrics_all[ds]['label'][''][metric]['mean'] + metrics_all[ds]['label'][''][metric]['stdev'], edgecolor = None, facecolor='#006BA4', alpha=0.4)
                    ax.fill_between(all_bias,metrics_all[ds]['label']['reweighting'][metric]['mean'] - metrics_all[ds]['label']['reweighting'][metric]['stdev'], metrics_all[ds]['label']['reweighting'][metric]['mean'] + metrics_all[ds]['label']['reweighting'][metric]['stdev'], edgecolor = None, facecolor='#595959', alpha=0.4)
                    ax.fill_between(all_bias,metrics_all[ds]['label']['massaging'][metric]['mean'] - metrics_all[ds]['label']['massaging'][metric]['stdev'], metrics_all[ds]['label']['massaging'][metric]['mean'] + metrics_all[ds]['label']['massaging'][metric]['stdev'], edgecolor = None, facecolor='#ABABAB', alpha=0.4)
                    ax.fill_between(all_bias,metrics_all[ds]['selectDoubleProp'][''][metric]['mean'] - metrics_all[ds]['selectDoubleProp'][''][metric]['stdev'], metrics_all[ds]['selectDoubleProp'][''][metric]['mean'] + metrics_all[ds]['selectDoubleProp'][''][metric]['stdev'], edgecolor = None, facecolor='#ABABAB', alpha=0.4)
                    ax.fill_between(all_bias,metrics_all[ds]['selectDoubleProp']['reweighting'][metric]['mean'] - metrics_all[ds]['selectDoubleProp']['reweighting'][metric]['stdev'], metrics_all[ds]['selectDoubleProp']['reweighting'][metric]['mean'] + metrics_all[ds]['selectDoubleProp']['reweighting'][metric]['stdev'], edgecolor = None, facecolor='#C85200', alpha=0.4)
                    ax.fill_between(all_bias,metrics_all[ds]['selectDoubleProp']['massaging'][metric]['mean'] - metrics_all[ds]['selectDoubleProp']['massaging'][metric]['stdev'], metrics_all[ds]['selectDoubleProp']['massaging'][metric]['mean'] + metrics_all[ds]['selectDoubleProp']['massaging'][metric]['stdev'], edgecolor = None, facecolor='#FF800E', alpha=0.4)
                    
            ax.tick_params(labelsize = 'large',which='major')
            if ylim is not None:
                ax.set_ylim(ylim)
            if xlim is not None :
                ax.set_xlim(xlim)
            #minor_ticks = np.arange(-1,1,0.05)
            #ax.set_yticks(minor_ticks) #, minor=True)
            #ax.set_xlabel(r'$\tau$', size=14)
            
            #ax.grid(which='both',linestyle='-',linewidth=0.5,color='lightgray')
            ax.grid(visible=True)
            ax.grid(which='minor',linestyle=':',linewidth=0.5,color='lightgray')
            ax.minorticks_on()

            #ax.legend(loc='best')
            ax.legend(prop={'size':10}, loc='lower left') #,  bbox_to_anchor=(1, 0.87))
            if title == '':
                plt.title(metric+" values wrt train set bias level :\n ("+ds+", unbiased test set)" , fontsize=14)
            else :
                plt.title(title, fontsize=14)

            if path_start is not None :
                    plt.savefig(path_start+ds+"_"+metric+".pdf", format="pdf", bbox_inches="tight") # dpi=1000) #dpi changes image quality
            if(display) :
                plt.show()
            plt.close()

def bargraph_adaptable(retrieval_path:str, dataset_list=['student','OULADstem', 'OULADsocial'], metric_list=['acc','StatParity','EqqOddsDiff','GenEntropyIndex'],
                  bias_list=['label','selectDoubleProp','selectLow'],preproc_list=['', 'reweighting','massaging'],
                  postproc_list = ['eqOddsProc', 'calEqOddsProc','ROC-SPD','ROC-EOD'],
                  ylim:list[float]=None, xlim:list[float]=None, all_bias:list[float]=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  plot_style:str = 'FILLED_STDEV', title: str = '', path_start:str = None, display=False) :
    """ Plot comparison of several mitigation methods in a bar graph
    plot_style : string, optional
        'FILLED_DEV' for display of standard deviation
        'SIMPLE_PLOT' for choice of metrics displaid without standard deviation
    title : string, optional
        None for automatic title
        '' (empty string) for no title
    """

    metrics_all = {}
    for ds in dataset_list :
        metrics_all[ds] = {}
        for bias in bias_list :
            metrics_all[ds][bias] = {}
            # retrieve results for active preprocessing methods
            for preproc in preproc_list :
                path = retrieval_path+ds+'_'+bias+'_'+preproc+"_RFAware__metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][bias][preproc] = metrics
            # retrieve results for FTU
            path = retrieval_path+ds+'_'+bias+"__RFBlinded__metricsForPlot.pkl"
            with open(path,"rb") as file:
                metrics = pickle.load(file)
                metrics_all[ds][bias]['FTU'] = metrics
            # retrieve results for postprocessing methods
            for postproc in postproc_list :
                path = retrieval_path+ds+'_'+bias+"__RFAware_"+postproc+"-BiasedValidFairTest_metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][bias][postproc] = metrics
    #metrics_all[bias_type][proc][metric]['mean']

    methods_list = preproc_list + postproc_list
    

    for ds in dataset_list :
        for metric in metric_list :
            for bias in bias_list :
                fig, ax = plt.subplots() # (figsize=(length, heigth)) #Scale size of image
                #ax.hlines(y=0,xmin=-0.05,xmax=1,colors='black')

                bar_width = 0.01
                indices = np.array(all_bias)

                ax.axhline(0, linewidth=0.5, color=colors[''])#Black
                
                dev = len(methods_list)/2
                for method in methods_list :
                    ax.bar(indices - dev*bar_width,metrics_all[ds][bias][method][metric]['mean'], label = 'biased model', width=bar_width)
                    if plot_style == 'FILLED_STDEV':
                        ax.errorbar(indices - dev*bar_width, metrics_all[ds][bias][method][metric]['mean'], yerr=metrics_all[ds][bias][method][metric]['stdev'], fmt = 'none', elinewidth=0.5, capsize=1, capthick=0.5)
                    plt.style.use('tableau-colorblind10')
                    dev +=1

                ax.tick_params(labelsize = 'large',which='major')
                
                ax.set_xlim([-0.05,0.95])

                #Indicate NaN by X
                """
                y_bottom = max(0,ax.get_ylim()[0])
                nan_EOP = np.isnan(metrics_all[ds][bias]['eqOddsProc'][metric]['mean'])
                for index in np.where(nan_EOP)[0]:
                    ax.text(index/10+0.005, y_bottom, 'X', ha='center', va='bottom', color='black', fontsize=7)
                nan_ROCEOD = np.isnan(metrics_all[ds][bias]['ROC-EOD'][metric]['mean'])
                for index in np.where(nan_ROCEOD)[0]:
                    ax.text(index/10+0.035, y_bottom, 'X', ha='center', va='bottom', color='black', fontsize=7)
                """
                
                #ax.grid(which='both',linestyle='-',linewidth=0.5,color='lightgray')
                #ax.grid(visible=True)
                ax.grid(which='minor',linestyle=':',linewidth=0.5,color='lightgray')
                ax.minorticks_on()
                
                ax.legend(loc='best')

                #ax.legend(prop={'size':10}, loc='lower left') #,  bbox_to_anchor=(1, 0.87))
                if title is None:
                    plt.title(metric+" values wrt train set "+bias+" bias level :\n ("+ds+", unbiased test set)" , fontsize=14)
                elif title != '' :
                    plt.title(title, fontsize=14)
                else : #title == ''
                    if metric == 'acc':
                        plt.title(ds, fontsize=20)

                if path_start is not None :
                        plt.savefig(path_start+ds+"_"+bias+"_"+metric+"_study.pdf", format="pdf", bbox_inches="tight") # dpi=1000) #dpi changes image quality
                if(display) :
                    plt.show()
                plt.close()

def bargraph_all_methods(retrieval_path:str, dataset_list=['student','OULADstem', 'OULADsocial'], metric_list=['acc','StatParity','EqqOddsDiff','GenEntropyIndex'],
                  bias_list=['label','selectDoubleProp','selectLow'], preproc_list=['', 'reweighting','massaging'], postproc_list = ['eqOddsProc', 'calEqOddsProc','ROC-SPD','ROC-EOD', 'ROC-AOD'],
                  ylim:list[float]=None, xlim:list[float]=None,
                  all_bias:list[float]=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  plot_style:str = 'FILLED_STDEV', title: str = '', path_start:str = None, display=False) :
    """ Plot comparison of all mitigation methods in a bar graph.
    plot_style : string, optional
        'FILLED_DEV' for display of standard deviation
        'SIMPLE_PLOT' for choice of metrics displaid without standard deviation
    title : string, optional
        None for automatic title
        '' (empty string) for no title
    """

    metrics_all = {}
    for ds in dataset_list :
        metrics_all[ds] = {}
        for bias in bias_list :
            metrics_all[ds][bias] = {}
            # retrieve results for active preprocessing methods
            for preproc in preproc_list :
                path = retrieval_path+ds+'_'+bias+'_'+preproc+"_RFAware__metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][bias][preproc] = metrics
            # retrieve results for FTU
            path = retrieval_path+ds+'_'+bias+"__RFBlinded__metricsForPlot.pkl"
            with open(path,"rb") as file:
                metrics = pickle.load(file)
                metrics_all[ds][bias]['FTU'] = metrics
            # retrieve results for postprocessing methods
            for postproc in postproc_list :
                path = retrieval_path+ds+'_'+bias+"__RFAware_"+postproc+"-BiasedValidFairTest_metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][bias][postproc] = metrics
    #metrics_all[bias_type][proc][metric]['mean']

    for ds in dataset_list :
        for metric in metric_list :
            for bias in bias_list :
                fig, ax = plt.subplots() # (figsize=(length, heigth)) #Scale size of image
                #ax.hlines(y=0,xmin=-0.05,xmax=1,colors='black')

                bar_width = 0.009
                indices = np.array(all_bias)
                #colors = {'':'black', 'reweighting': "#C85200", 'massaging': "#FF800E", 'FTU':"#FFBC79",
                #          'eqOddsProc':"#006BA4", 'calEqOddsProc':"#5F9ED1", 'ROC-SPD':"#A2C8EC", 'ROC-EOD':"#595959"}

                ax.axhline(0, linewidth=0.5, color=colors[''])#Black
                #Reference - mitigation
                ax.bar(indices - 4*bar_width,metrics_all[ds][bias][''][metric]['mean'], label = 'biased model', width=bar_width,color=colors[''])#Black
                #plt.axhline(metrics_all[''][fair_metric]['mean'][0], color='black',linestyle=':',linewidth=1,alpha = 0.5)
                ax.axhline(metrics_all[ds][bias][''][metric]['mean'][0], linestyle = ':', linewidth=0.5, color=colors[''], alpha=0.6)#Black
                #Preprocessingf
                ax.bar(indices - 3*bar_width,metrics_all[ds][bias]['reweighting'][metric]['mean'], label = 'reweighing', width=bar_width, color=colors['reweighting'])#Tenne/Dark orange
                ax.bar(indices - 2*bar_width,metrics_all[ds][bias]['massaging'][metric]['mean'], label = 'massaging', width=bar_width, color=colors['massaging'])#Pumpkin/Bright orange
                ax.bar(indices - 1*bar_width,metrics_all[ds][bias]['FTU'][metric]['mean'], label = 'FTU', width=bar_width, color=colors['FTU'])#Mac and cheese orange
                #ax.bar(indices + 0.5*bar_width,metrics_all[ds][bias]['eqOddsProc-FGT'][metric]['mean'], label = 'eqOddsProc-FGT', width=bar_width, color="#C85200")#Tenne/Dark orange
                ax.bar(indices + 0*bar_width,metrics_all[ds][bias]['eqOddsProc'][metric]['mean'], label = 'EOP', width=bar_width, color=colors['eqOddsProc'])
                #ax.bar(indices + 1.5*bar_width,metrics_all[ds][bias]['calEqOddsProc-FGT'][metric]['mean'], label = 'calEqOddsProc-FGT', width=bar_width, color="#FF800E")#Pumpkin/Bright orange
                ax.bar(indices + 1*bar_width,metrics_all[ds][bias]['calEqOddsProc'][metric]['mean'], label = 'CEO', width=bar_width, color=colors['calEqOddsProc'])
                #ax.bar(indices + 2.5*bar_width,metrics_all[ds][bias]['ROC-SPD-FGT'][metric]['mean'], label = 'ROC-SPD-FGT', width=bar_width, color="#FFBC79")#Mac and cheese orange
                ax.bar(indices + 2*bar_width,metrics_all[ds][bias]['ROC-SPD'][metric]['mean'], label = 'ROC-SPD', width=bar_width, color=colors['ROC-SPD'])
                #ax.bar(indices + 3.5*bar_width,metrics_all[ds][bias]['ROC-EOD-FGT'][metric]['mean'], label = 'ROC-EOD-FGT', width=bar_width, color="#FF800E")#Pumpkin/Bright orange
                ax.bar(indices + 3*bar_width,metrics_all[ds][bias]['ROC-EOD'][metric]['mean'], label = 'ROC-EOD', width=bar_width, color=colors['ROC-EOD'])
                ax.bar(indices + 4*bar_width,metrics_all[ds][bias]['ROC-AOD'][metric]['mean'], label = 'ROC-AOD', width=bar_width, color=colors['ROC-AOD'])
                
                if plot_style == 'FILLED_STDEV':
                    ax.errorbar(indices - 4*bar_width, metrics_all[ds][bias][''][metric]['mean'], yerr=metrics_all[ds][bias][''][metric]['stdev'], fmt = 'none', elinewidth=0.5, capsize=1, capthick=0.5, color=colors[''])
                    ax.errorbar(indices - 3*bar_width, metrics_all[ds][bias]['reweighting'][metric]['mean'], yerr=metrics_all[ds][bias]['reweighting'][metric]['stdev'], fmt = 'none', elinewidth=0.5, capsize=1, capthick=0.5, color=colors['reweighting'])
                    ax.errorbar(indices - 2*bar_width, metrics_all[ds][bias]['massaging'][metric]['mean'], yerr=metrics_all[ds][bias]['massaging'][metric]['stdev'], fmt = 'none', elinewidth=0.5, capsize=1, capthick=0.5, color=colors['massaging'])
                    ax.errorbar(indices - 1*bar_width, metrics_all[ds][bias]['FTU'][metric]['mean'], yerr=metrics_all[ds][bias]['FTU'][metric]['stdev'], fmt = 'none', elinewidth=0.5, capsize=1, capthick=0.5, color=colors['FTU'])
                    ax.errorbar(indices + 0*bar_width, metrics_all[ds][bias]['eqOddsProc'][metric]['mean'], yerr=metrics_all[ds][bias]['eqOddsProc'][metric]['stdev'], fmt = 'none', elinewidth=0.5, capsize=1, capthick=0.5, color=colors['eqOddsProc'])
                    ax.errorbar(indices + 1*bar_width, metrics_all[ds][bias]['calEqOddsProc'][metric]['mean'], yerr=metrics_all[ds][bias]['calEqOddsProc'][metric]['stdev'], fmt = 'none', elinewidth=0.5, capsize=1, capthick=0.5, color=colors['calEqOddsProc'])
                    ax.errorbar(indices + 2*bar_width, metrics_all[ds][bias]['ROC-SPD'][metric]['mean'], yerr=metrics_all[ds][bias]['ROC-SPD'][metric]['stdev'], fmt = 'none', elinewidth=0.5, capsize=1, capthick=0.5, color=colors['ROC-SPD'])
                    ax.errorbar(indices + 3*bar_width, metrics_all[ds][bias]['ROC-EOD'][metric]['mean'], yerr=metrics_all[ds][bias]['ROC-EOD'][metric]['stdev'], fmt = 'none', elinewidth=0.5, capsize=1, capthick=0.5, color=colors['ROC-EOD'])
                    ax.errorbar(indices + 4*bar_width, metrics_all[ds][bias]['ROC-AOD'][metric]['mean'], yerr=metrics_all[ds][bias]['ROC-AOD'][metric]['stdev'], fmt = 'none', elinewidth=0.5, capsize=1, capthick=0.5, color=colors['ROC-AOD'])
                    
                # Color-blind friendly color scheme :
                #'#FFBC79' Light orange/Mac and chesse '#898989'#Suva Grey/Light Gray '#ABABAB'#Dark gray '#595959'#Mortar/Darker Grey
                #https://stackoverflow.com/questions/74830439/list-of-color-names-for-matplotlib-style-tableau-colorblind10          
                

                ax.tick_params(labelsize = 'large',which='major')
                if ylim is not None:
                    ax.set_ylim(ylim)
                else :
                    if metric == 'acc':
                        #if bias != 'selectLow':
                        #    ax.set_ylim([0.3,1])
                        pass
                    elif metric == 'EqqOddsDiff':
                        ax.set_ylim(0,1.1)
                    elif metric == 'BlindCons':
                        ax.set_ylim([0.4,1.2])
                    elif metric == 'BCC':
                        ax.set_ylim([0.2,1.05])
                    elif metric == 'GenEntropyIndex':
                        if ds[0:7] == 'student':
                            ax.set_ylim(bottom = 0, top = 1.5)
                            """
                            if bias == 'label':
                                ax.set_ylim(bottom = 0, top = 1.5)
                            elif bias == 'selectDoubleProp':
                                ax.set_ylim(bottom = 0, top = 4)
                            else :
                                ax.set_ylim(bottom = 0, top = None)
                            """
                        else :
                            if bias == 'selectDoubleProp':
                                ax.set_ylim(0.0,0.5) 
                            else:
                                ax.set_ylim(0.0,0.6) 
                            #ax.set_ybound([0.0,0.5])
                    elif metric == 'StatParity':
                        if bias == 'label':
                             ax.set_ylim(-0.82,0.25)
                        elif bias == 'selectDoubleProp':
                            ax.set_ylim(-0.61,0.5)
                        elif bias == 'selectLow':
                            ax.set_ylim(-0.2,1.05)

                ax.set_xlim([-0.05,0.95])

                #Indicate NaN by X
                y_bottom = max(0,ax.get_ylim()[0])
                nan_EOP = np.isnan(metrics_all[ds][bias]['eqOddsProc'][metric]['mean'])
                for index in np.where(nan_EOP)[0]:
                    ax.text(index/10+0, y_bottom, 'X', ha='center', va='bottom', color='black', fontsize=6)
                nan_ROCEOD = np.isnan(metrics_all[ds][bias]['ROC-EOD'][metric]['mean'])
                for index in np.where(nan_ROCEOD)[0]:
                    ax.text(index/10+3*bar_width, y_bottom, 'X', ha='center', va='bottom', color='black', fontsize=6)
                nan_ROCAOD = np.isnan(metrics_all[ds][bias]['ROC-AOD'][metric]['mean'])
                for index in np.where(nan_ROCAOD)[0]:
                    ax.text(index/10+4*bar_width, y_bottom, 'X', ha='center', va='bottom', color='black', fontsize=6)
                
                #ax.set_xlabel("Bias intensity ("+r'$\beta_m$'+" for label, "+r"$p_u"+" for selection)", size=14)
                if metric == 'GenEntropyIndex':
                    if bias == 'label': 
                        ax.set_xlabel('Bias intensity $\\beta_l$', size=18)
                    else : # selection bias
                        ax.set_xlabel('Bias intensity $p_u$', size=18)
               
                # Name of metric on leftmost graph in paper
                if ds == 'studentBalanced' : #'student':
                    if metric == 'acc':
                        y_label = "Accuracy"
                    elif metric == 'StatParity':
                        y_label = "Statistical Parity Diff."
                    elif metric == 'EqqOddsDiff':
                        y_label = "Equalized Odds Diff."
                    elif metric == 'BlindCons':
                        y_label = "Consistency"
                    elif metric == 'GenEntropyIndex':
                        y_label = "Generalized Entropy Ind."
                    elif metric == 'Consistency':
                        y_label = "Consistency-with sens. attr."
                    elif metric == 'BCC':
                        y_label = "Balanced Cond. Consistency"
                    else :
                        y_label = metric
                    ax.set_ylabel(y_label, size = 20)
                
                #minor_ticks = np.arange(-1,1,0.05)
                #ax.set_yticks(minor_ticks) #, minor=True)
                #ax.set_xticks(all_bias)  # Set x-ticks to all_bias values
                #ax.set_xticklabels(['0','0.2','0.4','0.6','0.8'])
                
                #ax.grid(which='both',linestyle='-',linewidth=0.5,color='lightgray')
                #ax.grid(visible=True)
                ax.grid(which='minor',linestyle=':',linewidth=0.5,color='lightgray')
                ax.minorticks_on()
                
                if metric == 'acc' and ds == 'studentBalanced':
                    ax.legend(loc='lower left')
                """
                elif metric == 'GenEntropyIndex' and ds == 'OULADsocial':
                    ax.legend(loc='lower left')
                else :
                    ax.legend(loc='best')
                """

                #ax.legend(prop={'size':10}, loc='lower left') #,  bbox_to_anchor=(1, 0.87))
                if title is None:
                    plt.title(metric+" values wrt train set "+bias+" bias level :\n ("+ds+", unbiased test set)" , fontsize=14)
                elif title != '' :
                    plt.title(title, fontsize=14)
                else : #title == ''
                    if metric == 'acc':
                        plt.title(ds, fontsize=20)

                if path_start is not None :
                        plt.savefig(path_start+ds+"_"+bias+"_"+metric+".pdf", format="pdf", bbox_inches="tight") # dpi=1000) #dpi changes image quality
                if(display) :
                    plt.show()
                plt.close()

def bargraph_EWAF2025(retrieval_path:str, dataset_list=['student','OULADstem', 'OULADsocial'], metric_list=['acc','StatParity','EqqOddsDiff','GenEntropyIndex'],
                  bias_list=['label','selectDoubleProp'],preproc_list=['', 'reweighting','massaging'], postproc_list = [],
                  ylim:list[float]=None, xlim:list[float]=None, all_bias:list[float]=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  plot_style:str = 'FILLED_STDEV', title: str = '', path_start:str = None, display=False) :
    """ Plot comparison of several mitigation methods in a bar graph
    plot_style : string, optional
        'SIMPLE_PLOT' for choice of metrics displaid without standard deviation
    title : string, optional
        None for automatic title
        '' (empty string) for no title
    """
    valid = "_noValid_"
    metrics_all = {}
    for ds in dataset_list :
        metrics_all[ds] = {}
        for bias in bias_list :
            metrics_all[ds][bias] = {}
            # retrieve results for active preprocessing methods
            for preproc in preproc_list :
                path = retrieval_path+ds+'_'+bias+valid+preproc+"_RFAware__metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][bias][preproc] = metrics
            # retrieve results for FTU
            path = retrieval_path+ds+'_'+bias+valid+"_RFBlinded__metricsForPlot.pkl"
            with open(path,"rb") as file:
                metrics = pickle.load(file)
                metrics_all[ds][bias]['FTU'] = metrics
            # retrieve results for postprocessing methods
            for postproc in postproc_list :
                path = retrieval_path+ds+'_'+bias+valid+"_RFAware_"+postproc+"-BiasedValidFairTest_metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][bias][postproc] = metrics
    #metrics_all[bias_type][proc][metric]['mean']

    for ds in dataset_list :
        for metric in metric_list :
            fig, ax = plt.subplots() # (figsize=(length, heigth)) #Scale size of image

            bar_width = 0.01
            indices = np.array(all_bias)
        
            ax.axhline(0, linewidth=0.5, color='black')#Black

            ax.bar(indices - 3.6*bar_width,metrics_all[ds]['label'][''][metric]['mean'], label = 'label bias', width=bar_width,color="#595959")#Dark gray
            ax.bar(indices - 2.6*bar_width,metrics_all[ds]['label']['reweighting'][metric]['mean'], label = 'label+reweighing', width=bar_width, color="#006BA4")#Cerulean/Blue
            ax.bar(indices - 1.6*bar_width,metrics_all[ds]['label']['massaging'][metric]['mean'], label = 'label+massaging', width=bar_width, color="#5F9ED1")#Picton blue
            ax.bar(indices - 0.6*bar_width,metrics_all[ds]['label']['FTU'][metric]['mean'], label = 'label+FTU', width=bar_width, color="#A2C8EC")#Seil/Light blue

            ax.bar(indices + 0.6*bar_width,metrics_all[ds]['selectDoubleProp'][''][metric]['mean'], label = 'selection bias', width=bar_width, color="#898989")#Suva Grey
            ax.bar(indices + 1.6*bar_width,metrics_all[ds]['selectDoubleProp']['reweighting'][metric]['mean'], label = 'select+reweighing', width=bar_width, color="#C85200")#Tenne/Dark orange
            ax.bar(indices + 2.6*bar_width,metrics_all[ds]['selectDoubleProp']['massaging'][metric]['mean'], label = 'select+massaging', width=bar_width, color="#FF800E")#Pumpkin/Bright orange
            ax.bar(indices + 3.6*bar_width,metrics_all[ds]['selectDoubleProp']['FTU'][metric]['mean'], label = 'select+FTU', width=bar_width, color="#FFBC79")#Mac and cheese orange
            
            # Color-blind friendly color scheme :
            #'#FFBC79' Light orange/Mac and chesse '#898989'#Suva Grey/Light Gray '#ABABAB'#Dark gray '#595959'#Mortar/Darker Grey
            #https://stackoverflow.com/questions/74830439/list-of-color-names-for-matplotlib-style-tableau-colorblind10          
            
            ax.tick_params(labelsize = 'large',which='major')
            if ylim is not None:
                ax.set_ylim(ylim)
            else :
                if metric == 'acc':
                    ax.set_ylim([0.4,1])
            ax.set_xlim([-0.05,0.95])
            ax.set_xlabel('Bias intensity ($\\beta_l$ for label, $p_u$ for selection)', size=14)
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
            
            if metric == 'acc' : #and ds == 'student':
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

def presentation_EWAF2025(retrieval_path:str, dataset_list=['OULADstem'], metric_list=['acc','StatParity'],
                  bias_list=['label','selectDoubleProp'],preproc_list=['', 'reweighting','massaging'], postproc_list = [],
                  style = 'slide', all_bias:list[float]=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  title: str = '', path_start:str = None, display=False) :
    """ Plot comparison of several mitigation methods in a bar graph with distinct graphs for selection and label bias
    style : 'slide' ou 'poster'
    title : string, optional
        None for automatic title
        '' (empty string) for no title
    """
    valid = "_noValid_"
    metrics_all = {}
    for ds in dataset_list :
        metrics_all[ds] = {}
        for bias in bias_list :
            metrics_all[ds][bias] = {}
            # retrieve results for active preprocessing methods
            for preproc in preproc_list :
                path = retrieval_path+ds+'_'+bias+valid+preproc+"_RFAware__metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][bias][preproc] = metrics
            # retrieve results for FTU
            path = retrieval_path+ds+'_'+bias+valid+"_RFBlinded__metricsForPlot.pkl"
            with open(path,"rb") as file:
                metrics = pickle.load(file)
                metrics_all[ds][bias]['FTU'] = metrics
            # retrieve results for postprocessing methods
            for postproc in postproc_list :
                path = retrieval_path+ds+'_'+bias+valid+"_RFAware_"+postproc+"-BiasedValidFairTest_metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][bias][postproc] = metrics
    #metrics_all[bias_type][proc][metric]['mean']

    for ds in dataset_list :
        for metric in metric_list :
            for bias in bias_list :
                fig, ax = plt.subplots() # (figsize=(length, heigth)) #Scale size of image
                #ax.hlines(y=0,xmin=-0.05,xmax=1,colors='black')

                bar_width = 0.015
                indices = np.array(all_bias)
                
                #colorblindIBM = {'ultramarine': "#648FFF", 'indigo':"785EF0", 'magenta':"#DC267F", 'orange':"#FE6100", 'gold':"#FFB000",
                # 'black':"#000000", 'white':"#FFFFFF"}
                colors = {'':'black', 'reweighting': "#FE6100", 'massaging': "#648FFF", 'FTU':"#785EF0"}

                ax.axhline(0, linewidth=0.5, color=colors[''])#Black
                ax.axhline(metrics_all[ds][bias][''][metric]['mean'][0], linestyle = ':', linewidth=1.5, color=colors[''], alpha=0.6)#Black

                ax.bar(indices - 1.5*bar_width,metrics_all[ds][bias][''][metric]['mean'], label = 'Unmitigated', width=bar_width,color=colors[''])
                ax.bar(indices - 0.5*bar_width,metrics_all[ds][bias]['reweighting'][metric]['mean'], label = 'Reweighing', width=bar_width, color=colors['reweighting'])
                ax.bar(indices + 0.5*bar_width,metrics_all[ds][bias]['massaging'][metric]['mean'], label = 'Massaging', width=bar_width, color=colors['massaging'])
                ax.bar(indices + 1.5*bar_width,metrics_all[ds][bias]['FTU'][metric]['mean'], label = 'FTU', width=bar_width, color=colors['FTU'])
 
                # Color-blind friendly color scheme :
                #'#FFBC79' Light orange/Mac and chesse '#898989'#Suva Grey/Light Gray '#ABABAB'#Dark gray '#595959'#Mortar/Darker Grey
                #https://stackoverflow.com/questions/74830439/list-of-color-names-for-matplotlib-style-tableau-colorblind10          
                
                if style == 'slide' :
                    labelsize = '16'
                else :
                    labelsize = 'large'
                ax.tick_params(labelsize = labelsize,which='major')
                if metric == 'acc':
                    ax.set_ylim([0.45,1])
                elif metric == 'StatParity':
                    ax.set_ylim([-0.45,0.4])
                ax.set_xlim([-0.05,0.95])
                """
                ax.set_xlabel('Bias intensity ($\\beta_l$ for label, $p_u$ for selection)', size=14)
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
                """
                
                ax.grid(which='minor',linestyle=':',linewidth=0.5,color='lightgray')
                ax.minorticks_on()
                
                if metric == 'acc' : #and ds == 'student':
                    position = 'lower left'
                elif metric == 'StatParity' :
                    position = 'upper left'
                elif metric == 'GenEntropyIndex' and ds == 'OULADsocial':
                    position = 'lower left'
                else :
                    position = 'best'
                if style == 'slide' :
                    ax.legend(loc=position,prop={'size': 16})
                else :
                    ax.legend(loc=position) #,prop={'size': 16})

                #ax.legend(prop={'size':10}, loc='lower left') #,  bbox_to_anchor=(1, 0.87))
                if title is None:
                    plt.title(metric+" values wrt train set bias level :\n ("+ds+", unbiased test set)" , fontsize=14)
                elif title != '' :
                    plt.title(title, fontsize=14)

                if path_start is not None :
                        plt.savefig(path_start+ds+"_"+bias+"_"+metric+"_ewaf-"+style+".png", format="png", bbox_inches="tight", dpi=1000) #dpi changes image quality
                if(display) :
                    plt.show()
                plt.close()

def all_scatter(dataset_list=['student','OULADstem', 'OULADsocial'], metric_list=['acc','StatParity','EqqOddsDiff','GenEntropyIndex'],
                  bias_list=['label','selectDoubleProp','selectLow']):
    for ds in dataset_list :
        for metric in metric_list :
            for bias in bias_list :
                pass

def fair_acc(retrieval_path:str, dataset, bias_type, fair_metric, preproc_list=['', 'reweighting','massaging'],
                  postproc_list = ['eqOddsProc', 'calEqOddsProc','ROC-SPD','ROC-EOD'], all_bias:list[float]=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  title: str = '', path_start:str = None, display=False) :
    """ Plot comparison of several mitigation methods in a bar graph
    plot_style : string, optional
        'SIMPLE_PLOT' for choice of metrics displaid without standard deviation
    title : string, optional
        None for automatic title
        '' (empty string) for no title
    """
    metrics_all = {}
    # retrieve results for active preprocessing methods
    for preproc in preproc_list :
        path = retrieval_path+dataset+'_'+bias_type+'_'+preproc+"_RFAware__metricsForPlot.pkl"
        with open(path,"rb") as file:
            metrics = pickle.load(file)
            metrics_all[preproc] = metrics
    # retrieve results for FTU
    path = retrieval_path+dataset+'_'+bias_type+"__RFBlinded__metricsForPlot.pkl"
    with open(path,"rb") as file:
        metrics = pickle.load(file)
        metrics_all['FTU'] = metrics
    # retrieve results for postprocessing methods
    for postproc in postproc_list :
        path = retrieval_path+dataset+'_'+bias_type+"__RFAware_"+postproc+"-BiasedValidFairTest_metricsForPlot.pkl"
        with open(path,"rb") as file:
            metrics = pickle.load(file)
            metrics_all[postproc] = metrics
    #metrics_all[proc][metric]['mean']

    fig, ax = plt.subplots() # (figsize=(length, heigth)) #Scale size of image
    plt.axhline(metrics_all[''][fair_metric]['mean'][0], color='black',linestyle=':',linewidth=1,alpha = 0.5)
    plt.axvline(metrics_all['']['acc']['mean'][0], color='black',linestyle=':',linewidth=1,alpha = 0.5)

    ax.grid(which='both',linestyle=':',linewidth=0.5,color='lightgray')
    ax.minorticks_on()
    #ax.grid(which='both',linestyle='-',linewidth=0.5,color='lightgray')
    #ax.grid(visible=True)

    ax.plot(metrics_all['']['acc']['mean'],metrics_all[''][fair_metric]['mean'], c=colors[''], label=legend[''], alpha=0.8)
    ax.scatter(metrics_all['reweighting']['acc']['mean'],metrics_all['reweighting'][fair_metric]['mean'], c=colors['reweighting'], label=legend['reweighting'], alpha=0.8)
    ax.scatter(metrics_all['massaging']['acc']['mean'],metrics_all['massaging'][fair_metric]['mean'], c=colors['massaging'], label=legend['massaging'], alpha=0.8)
    ax.scatter(metrics_all['FTU']['acc']['mean'],metrics_all['FTU'][fair_metric]['mean'], c=colors['FTU'], label=legend['FTU'], alpha=0.8)
    ax.scatter(metrics_all['eqOddsProc']['acc']['mean'],metrics_all['eqOddsProc'][fair_metric]['mean'], c=colors['eqOddsProc'], label=legend['eqOddsProc'], alpha=0.8)
    ax.scatter(metrics_all['calEqOddsProc']['acc']['mean'],metrics_all['calEqOddsProc'][fair_metric]['mean'], c=colors['calEqOddsProc'], label=legend['calEqOddsProc'], alpha=0.8)
    ax.scatter(metrics_all['ROC-SPD']['acc']['mean'],metrics_all['ROC-SPD'][fair_metric]['mean'], c=colors['ROC-SPD'], label=legend['ROC-SPD'], alpha=0.8)
    ax.scatter(metrics_all['ROC-EOD']['acc']['mean'],metrics_all['ROC-EOD'][fair_metric]['mean'], c=colors['ROC-EOD'], label=legend['ROC-EOD'], alpha=0.8)
    
    #ax.plot(metrics_all['']['acc']['mean'],)
    #min_val = min(min(metrics_all[proc]['acc']['mean']), min(metrics_all[proc][fair_metric]['mean']))
    #max_val = max(max(metrics_all[proc]['acc']['mean']), max(metrics_all[proc][fair_metric]['mean']))
    #ax.plot([min_val, max_val], [min_val, max_val], linestyle='--')
    
    #ax.tick_params(labelsize = 'large',which='major')
    """
    if ylim is not None:
        ax.set_ylim(ylim)
    else :
        if metric == 'acc':
            if bias != 'selectLow':
                ax.set_ylim([0.4,1])
        elif metric == 'BlindCons':
            ax.set_ylim([0.6,1.2])
        elif metric == 'GenEntropyIndex':
            if ds != 'student':
                ax.set_ylim(0.0,0.5) #,auto=False)
                #ax.set_ybound([0.0,0.5])
    ax.set_xlim([-0.05,0.95])
    """
    
    ax.set_xlabel("Accuracy", size=18)
    
    if fair_metric == 'StatParity':
        y_label = 'Statistical Parity'
    ax.set_ylabel(y_label, size = 20)
    
    #minor_ticks = np.arange(-1,1,0.05)
    #ax.set_yticks(minor_ticks) #, minor=True)
    #ax.set_xticks(all_bias)  # Set x-ticks to all_bias values
    #ax.set_xticklabels(['0','0.2','0.4','0.6','0.8'])

    ax.legend(loc='best')
    #ax.legend(prop={'size':10}, loc='lower left') #,  bbox_to_anchor=(1, 0.87))

    if title is None:
        plt.title("Fairness ("+fair_metric+") accuracy relationship\n("+dataset+", "+bias_type+", unbiased test set)" , fontsize=14)
    elif title != '' :
        plt.title(title, fontsize=14)
    #No title for title == ''


    if path_start is not None :
            plt.savefig(path_start+"fair-acc_"+dataset+"_"+bias_type+"_"+fair_metric+".pdf", format="pdf", bbox_inches="tight") # dpi=1000) #dpi changes image quality
    if(display) :
        plt.show()
    plt.close()

def bias_vs_unbiased(retrieval_path:str, bias_type, dataset_list:list[str], model_list=list[str], preproc_list=['', 'reweighting','massaging'],
                  postproc_list = ['eqOddsProc', 'calEqOddsProc','ROC-SPD','ROC-EOD'], title: str = '', path_start:str = None, display=False) :
    """ Plot comparison of several mitigation methods in a bar graph
    
    title : string, optional
        None for automatic title
        '' (empty string) for no title
    """
    metrics_all = {}
    for ds in dataset_list :
        metrics_all[ds] = {}
        for model in model_list :
            metrics_all[ds][model] = {}
            # retrieve results for active preprocessing methods
            for preproc in preproc_list :
                metrics_all[ds][model][preproc] = {}
                path = retrieval_path+ds+'_'+bias_type+'_'+preproc+"_"+model+"Aware__metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][model][preproc]['fair'] = metrics
                path = retrieval_path+ds+'_'+bias_type+'_'+preproc+"_"+model+"Aware__Biased_metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][model][preproc]['biased'] = metrics
            # retrieve results for FTU
            metrics_all[ds][model]['FTU'] = {}
            path = retrieval_path+ds+'_'+bias_type+"__"+model+"Blinded__metricsForPlot.pkl"
            with open(path,"rb") as file:
                metrics = pickle.load(file)
                metrics_all[ds][model]['FTU']['fair'] = metrics
            path = retrieval_path+ds+'_'+bias_type+"__"+model+"Blinded__Biased_metricsForPlot.pkl"
            with open(path,"rb") as file:
                metrics = pickle.load(file)
                metrics_all[ds][model]['FTU']['biased'] = metrics
            # retrieve results for postprocessing methods
            for postproc in postproc_list :
                metrics_all[ds][model][postproc] = {}
                path = retrieval_path+ds+'_'+bias_type+"__"+model+"Aware_"+postproc+"-BiasedValidFairTest_metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][model][postproc]['fair'] = metrics
                path = retrieval_path+ds+'_'+bias_type+"__"+model+"Aware_"+postproc+"-BiasedValidBiasedTest_metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics = pickle.load(file)
                    metrics_all[ds][model][postproc]['biased'] = metrics
            #metrics_all[proc][metric]['mean']

    fig, ax = plt.subplots() # (figsize=(length, heigth)) #Scale size of image
    ax.grid(which='major',linestyle=':',linewidth=0.5,color='lightgray')
    ax.minorticks_on()
    #ax.grid(which='both',linestyle='-',linewidth=0.5,color='lightgray')
    #ax.grid(visible=True)
    if bias_type == 'label':
        ax.plot([-1, 2.4], [-1, 2.4], linestyle=':', color='black', zorder=1)
    elif bias_type == 'selectDoubleProp':
        ax.plot([-1, 3.6], [-1, 3.6], linestyle=':', color='black', zorder=1)
    elif bias_type == 'selectLow':
        ax.plot([-0.25, 3.9], [-0.25, 3.9], linestyle=':', color='black', zorder=1)

    #min_val = min(min(metrics_all[proc]['acc']['mean']), min(metrics_all[proc][fair_metric]['mean']))
    #max_val = max(max(metrics_all[proc]['acc']['mean']), max(metrics_all[proc][fair_metric]['mean']))

    colorblindIBM = {'ultramarine': "#648FFF", 'indigo':"785EF0", 'magenta':"#DC267F", 'orange':"#FE6100", 'gold':"#FFB000",
                 'black':"#000000", 'white':"#FFFFFF"}
    m_colors = {'acc':"#785EF0", 'BlindCons':"#DC267F", 'BCC':"#DC267F", 'StatParity': "#FE6100", 'EqqOddsDiff': "#FFB000", 'GenEntropyIndex':"#648FFF"}
    m_legend = {'StatParity': "SPD", 'EqqOddsDiff': "EOD", 'acc':"Accuracy", 'BlindCons':"Consistency", 'BCC':"BCC", 'GenEntropyIndex':"GEI"} 
    m_marker = {'acc':"o", 'BlindCons':"p", 'BCC':"p", 'StatParity': "d", 'EqqOddsDiff': "P", 'GenEntropyIndex':"p"}
    alpha = 0.4
    label = True
    for ds in dataset_list:
        for model in metrics_all[ds].keys() :
            for proc in metrics_all[ds][model].keys():
                if label :
                    ax.scatter(metrics_all[ds][model][proc]['fair']['acc']['mean'][1:],metrics_all[ds][model][proc]['biased']['acc']['mean'][1:], facecolors=m_colors['acc'], edgecolors = m_colors['acc'], label=m_legend['acc'], marker=m_marker['acc'], alpha=alpha)
                    #ax.scatter(metrics_all[ds][model][proc]['fair']['BlindCons']['mean'][1:],metrics_all[ds][model][proc]['biased']['BlindCons']['mean'][1:], facecolors=m_colors['BlindCons'], edgecolors = m_colors['BlindCons'], label=m_legend['BlindCons'], marker=m_marker['BlindCons'], alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['BCC']['mean'][1:],metrics_all[ds][model][proc]['biased']['BCC']['mean'][1:], facecolors=m_colors['BCC'], edgecolors = m_colors['BCC'], label=m_legend['BCC'], marker=m_marker['BCC'], alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['StatParity']['mean'][1:],metrics_all[ds][model][proc]['biased']['StatParity']['mean'][1:], facecolors=m_colors['StatParity'], edgecolors = m_colors['StatParity'], label=m_legend['StatParity'], marker=m_marker['StatParity'], alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['GenEntropyIndex']['mean'][1:],metrics_all[ds][model][proc]['biased']['GenEntropyIndex']['mean'][1:], facecolors=m_colors['GenEntropyIndex'], edgecolors = m_colors['GenEntropyIndex'], label=m_legend['GenEntropyIndex'], marker=m_marker['GenEntropyIndex'], alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['EqqOddsDiff']['mean'][1:],metrics_all[ds][model][proc]['biased']['EqqOddsDiff']['mean'][1:], facecolors=m_colors['EqqOddsDiff'], edgecolors = m_colors['EqqOddsDiff'], label=m_legend['EqqOddsDiff'], marker=m_marker['EqqOddsDiff'], alpha=alpha)
                else :
                    ax.scatter(metrics_all[ds][model][proc]['fair']['acc']['mean'][1:],metrics_all[ds][model][proc]['biased']['acc']['mean'][1:], facecolors=m_colors['acc'], edgecolors = m_colors['acc'], label=None, marker=m_marker['acc'], alpha=alpha)
                    #ax.scatter(metrics_all[ds][model][proc]['fair']['BlindCons']['mean'][1:],metrics_all[ds][model][proc]['biased']['BlindCons']['mean'][1:], facecolors=m_colors['BlindCons'], edgecolors = m_colors['BlindCons'], label=None, marker=m_marker['BlindCons'], alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['BCC']['mean'][1:],metrics_all[ds][model][proc]['biased']['BCC']['mean'][1:], facecolors=m_colors['BCC'], edgecolors = m_colors['BCC'], label=None, marker=m_marker['BCC'], alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['StatParity']['mean'][1:],metrics_all[ds][model][proc]['biased']['StatParity']['mean'][1:], facecolors=m_colors['StatParity'], edgecolors = m_colors['StatParity'], label=None, marker=m_marker['StatParity'], alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['GenEntropyIndex']['mean'][1:],metrics_all[ds][model][proc]['biased']['GenEntropyIndex']['mean'][1:], facecolors=m_colors['GenEntropyIndex'], edgecolors = m_colors['GenEntropyIndex'], label=None, marker=m_marker['GenEntropyIndex'], alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['EqqOddsDiff']['mean'][1:],metrics_all[ds][model][proc]['biased']['EqqOddsDiff']['mean'][1:], facecolors=m_colors['EqqOddsDiff'], edgecolors = m_colors['EqqOddsDiff'], label=None, marker=m_marker['EqqOddsDiff'], alpha=alpha)
                label = False

    #ax.tick_params(labelsize = 'large',which='major')
    
    if bias_type == 'label':
        ax.set_ylim([-1,2.5])
        ax.set_xlim([-1,2.5])
    elif bias_type == 'selectDoubleProp':
        ax.set_ylim([-1,3.7])
        ax.set_xlim([-1,3.7])
    elif bias_type == 'selectLow':
        ax.set_ylim([-0.25,4])
        ax.set_xlim([-0.25,4])
    plt.axis('scaled')

    #if bias_type == 'label':
    ax.set_ylabel("Biased evaluation", size = 20)
    ax.set_xlabel("Fair evaluation", size=20)
    
    
    #minor_ticks = np.arange(-1,1,0.05)
    #ax.set_yticks(minor_ticks) #, minor=True)
    #ax.set_xticks(all_bias)  # Set x-ticks to all_bias values
    #ax.set_xticklabels(['0','0.2','0.4','0.6','0.8'])

    ax.legend(loc='best')
    #ax.legend(prop={'size':10}, loc='lower left') #,  bbox_to_anchor=(1, 0.87))

    if title is None:
        plt.title("Fair versus biased evaluation of metrics for "+bias_type , fontsize=14)
    elif title != '' :
        plt.title(title, fontsize=14)
    #No title for title == ''

    if path_start is not None :
        plt.savefig(path_start+"biasVSunbiased_"+bias_type+".pdf", format="pdf", bbox_inches="tight") # dpi=1000) #dpi changes image quality
    if(display) :
        plt.show()
    plt.close()