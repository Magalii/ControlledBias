##############################################################################
# Contains all the methods for plotting results, like in article(s) and more #
##############################################################################

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
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

legend_ds = {'student': "Student", 'studentBalanced': "StudentBalanced", 'OULADstem':"OULADstem", "OULADsocial":"OULADsocial", 'OULADsocialHarder': "OULADsocial-Complex",'OULADstemHarder': "OULADstem-Complex"}
legend_bias_abrev = {'label':"Label bias", 'selectRandom': "Random select.", 'selectLow': "Self-select.", 'selectDoubleProp':"Malicious select.", 'selectPrivNoUnpriv':"Malicious - Priv. only"}
legend_bias_long = {'label':"Label bias", 'selectRandom': "Random selection", 'selectLow': "Self-selection", 'selectDoubleProp':"Malicious selection", 'selectPrivNoUnpriv':"Malicious - Priv. only"}
legend_method = {'':'Unmitigated', 'reweighting': "Reweighting", 'massaging': "Massaging", 'FTU':"FTU",
                 'eqOddsProc':"EOP", 'calEqOddsProc':"CEO", 'ROC-SPD':"ROC-SPD", 'ROC-EOD':"ROC-EqOp", 'ROC-AOD':"ROC-AvOd"}
legend_metric = {'acc':"Accuracy", 'accBal':"Balanced acc.", 'ROCAUC':"ROC-AUC", 'sens_attr_usage':"Sens. attr. usage",
                 'FPR':"FPR global",'FNR':"FNR global", 'FalsePosRateDiff': "FPR diff.", 'FalseNegRateDiff':"FNR diff.",
                 'StatParity':"Stat. Parity Diff.", 'EqqOddsDiff':"Equalized Odds Diff.", 'GenEntropyIndex':"Generalized Entropy Ind.", 
                 'BlindCons':"Consistency", 'Consistency':"Consistency-with sens. attr.", 'BCC': "Balanced Cond. Consistency"}       
legend_metric_short = {'acc':"Accuracy", 'accBal':"Balanced acc.", 'ROCAUC':"ROC-AUC", 'sens_attr_usage':"Sens. attr. usage",
                 'FPR':"FPR global",'FNR':"FNR global", 'FalsePosRateDiff': "FPR diff.", 'FalseNegRateDiff':"FNR diff.",
                 'StatParity':"SPD", 'EqqOddsDiff':"EqOd", 'GenEntropyIndex':"GEI", 
                 'BlindCons':"Consistency", 'Consistency':"Consistency-with sens. attr.", 'BCC': "BCC"}

colorblindIBM = {'ultramarine': "#648FFF", 'indigo':"#785EF0", 'magenta':"#DC267F", 'orange':"#FE6100", 'gold':"#FFB000",
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

def plot_metrics_all(data_list, bias_list, preproc_list, postproc_list, model_list, blinding_list, all_bias, title='', path_start=None, plot_biased=True, medium=None, to1=False) :
    """ Apply plot-by_bias for all combinations of datasets, bias, preproc, model and blinding given as argument
        Each graph represent one mitigation situation, plotting evolution of divers metrics with the increasing intensity of the bias
        Dataset splits metrics for plots must already be saved and be store at 'path_start'
        medium : Adapts the plot parameters to specific medium, like 'article'. None gives the default
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
                            if title == None :
                                title = 'Metric values wrt train set bias level\n ('+bias+' bias, '+preproc+', '+model+visibility+', unbiased test set)'
                            plot_metrics(metrics_for_plot, all_bias, bias, ds, plot_style='FILLED_STDEV',title=title, medium=medium, path_start='plots/'+pref+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_byBias_unbiasedTest', display=False)
                            del metrics_for_plot
                            if plot_biased :
                                with open(path+"__Biased"+"_metricsForPlot.pkl","rb") as file:
                                    metricsBiased_for_plot = pickle.load(file)
                                if title == None :
                                    title ='Metric values wrt train set bias level\n ('+bias+' bias, '+preproc+', '+model+visibility+', biased test set)'
                                plot_metrics(metricsBiased_for_plot, all_bias, bias, ds, plot_style='FILLED_STDEV', title=title, medium=medium, path_start='plots/'+pref+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_byBias_BiasedTest', display=False)
                                del metricsBiased_for_plot
                            gc.collect()
                        else : #Need to take post-proc and there specificity into account
                            for postproc in postproc_list :
                                path = path_start+"Results/"+pref+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_'+postproc
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
                                    title_BVFT = 'Metric values wrt train set bias level\n ('+bias+' bias, '+model+visibility+', '+postproc+', biased postproc & fair test set)'
                                    title_BVBT = 'Metric values wrt train set bias level\n ('+bias+' bias, '+model+visibility+', '+postproc+', biased postproc & biased test set)'
                                    title_FVFT = 'Metric values wrt train set bias level\n ('+bias+' bias, '+model+visibility+', '+postproc+', fair postproc & fair test set)'
                                else :
                                    title_BVFT, title_BVBT, title_FVFT  = title, title, title
                                plot_metrics(metrics_BVFT, all_bias, bias, ds, plot_style='FILLED_STDEV', title=title_BVFT, medium=medium, path_start='plots/'+pref+ds+'_'+bias+'__'+model+visibility+'_'+postproc+'-BiasedValidFairTest_byBias_', display=False)
                                plot_metrics(metrics_BVBT, all_bias, bias, ds, plot_style='FILLED_STDEV',title=title_BVBT, medium=medium, path_start='plots/'+pref+ds+'_'+bias+'__'+model+visibility+'_'+postproc+'-BiasedValidBiasedTest_byBias_', display=False)
                                plot_metrics(metrics_FVFT, all_bias, bias, ds, plot_style='FILLED_STDEV',title=title_FVFT, medium=medium, path_start='plots/'+pref+ds+'_'+bias+'__'+model+visibility+'_'+postproc+'-FairValidFairTest_byBias_', display=False)
                                
                                #TODO add Ideal ground truth + fair test if studied
                                
                                #Manage memory
                                del metrics_BVFT, metrics_BVBT, metrics_FVFT
                                gc.collect()


def plot_metrics(metrics_for_plot, all_bias: list[float], bias_type:str, dataset:str, plot_style: str = 'FILLED_STDEV', title: str = '', path_start:str = None, medium=None, display=False) :
    """ For one mitigation situation, plot evolution of divers metrics with the increasing intensity of the bias
    metrics_for_plot : Dictionary {str:{'mean': list[float], 'stdev':list[float]}}
        dictionary containing the metrics for that scenario, such that metrics_for_plot[metric] = {'mean': list of values, 'stdev': list of standard variation}
    all_bias : list[float]
        list of bias levels to be used as x-values, must be the same length as the list of metric values in 'metrics_for_plot'
    bias_type : str
        Whether the bias is of type 'label' or 'selection'. Used for x-axis label
    plot_style : string, optional
        None for automatic display of all metrics
        'SIMPLE_PLOT' for choice of metrics displaid without standard deviation
        'FILLED_STDEV' for choice of metrics displaid with standard deviation as colored area arround the curve
    title : str
        '' (empty string) for no title
        None for automatic title
        non-empty string to for custom title
    medium : str
        Adapts the plot parameters to specific medium, like 'article'
        None gives the default
    """
    def_length, def_height = 6.4, 4.8
    def_ymin, def_ymax = -1.1, 1.1
    legend_in_plot = True
    if medium == 'article':
        if bias_type in ['selectRandom','selectLow', 'selectPrivNoUnpriv'] :
            ymin, ymax = -0.85, def_ymax
            height = def_height*((ymax-ymin)/(def_ymax-def_ymin))
            figsize=(def_length, height)
            legend_in_plot = True
    else :
        ymin, ymax = def_ymin, def_ymax
        figsize=(def_length, def_height)
        legend_in_plot = True
        
    fig, ax = plt.subplots(figsize=figsize) # (figsize=(length, heigth)) #Scale size of image (default: [6.4, 4.8])
    ax.hlines(0,0,1,colors='black')

    if plot_style is None :
        for metric in metrics_keys :
            #if metric != "DP_ratio":
            plt.plot(all_bias,metrics_for_plot[metric],label = str(metric),linestyle="--",marker="o")
            plt.style.use('tableau-colorblind10')
    elif  plot_style == 'SIMPLE_PLOT' or plot_style == 'FILLED_STDEV' :
        if medium[0:7] == 'article' :
            if bias_type == 'selectPrivNoUnpriv':
                metric_list = ['accBal','GenEntropyIndex','BCC', 'FPR', 'FalseNegRateDiff', 'FalsePosRateDiff', 'StatParity']
            else :
                metric_list = ['accBal','GenEntropyIndex','BCC', 'FPR', 'FalseNegRateDiff', 'FalsePosRateDiff', 'StatParity','sens_attr_usage']
        elif medium == 'legend':
            metric_list = ['accBal','BCC' ,'sens_attr_usage','FPR', 'FalseNegRateDiff', 'FalsePosRateDiff', 'GenEntropyIndex', 'StatParity']
        else :
            metric_list = ['acc','accBal','ROCAUC','BCC','GenEntropyIndex','FPR','FNR', 'FalsePosRateDiff', 'FalseNegRateDiff', 'StatParity','sens_attr_usage']
        metric_colors = {'acc':'#595959','accBal':'#595959','ROCAUC':'#595959','sens_attr_usage':'#000000','BCC':"#FFBC79",'GenEntropyIndex':"#5F9ED1",'FPR':"#A2C8EC",'FNR':"#A2C8EC", 'FalsePosRateDiff':'#FF800E', 'FalseNegRateDiff':"#C85200", 'StatParity':"#006BA4"}
        metric_markers = {'acc':"+",'accBal':"o",'ROCAUC':"x",'sens_attr_usage':".",'BCC':"p",'GenEntropyIndex':"s",'FPR':"v",'FNR':"^", 'FalsePosRateDiff':"P", 'FalseNegRateDiff':"X", 'StatParity':"d"}
        indices = np.array(all_bias)

        if plot_style == 'FILLED_STDEV':
        #Shade for std values
            for metric in metric_list :
                if metric == 'FalsePosRateDiff': jitter = -0.005
                elif metric == 'FalseNegRateDiff': jitter = 0.005
                else : jitter = 0
                metric_mean = metrics_for_plot[metric]['mean']
                metric_stdev = metrics_for_plot[metric]['stdev']
                ax.fill_between(indices+jitter,metric_mean - metric_stdev, metric_mean + metric_stdev, edgecolor = None, facecolor=metric_colors[metric], alpha=0.4)
        #Mean values
        for metric in metric_list :
            if metric == 'FalsePosRateDiff': jitter = -0.005
            elif metric == 'FalseNegRateDiff': jitter = 0.005
            else : jitter = 0
            ax.plot(indices+jitter,metrics_for_plot[metric]['mean'], label=legend_metric_short[metric], linestyle="--",marker=metric_markers[metric], color=metric_colors[metric])
        
    ax.tick_params(labelsize = 'large',which='major')
    ax.set_ylim([ymin,ymax])
    ax.set_xlim([0,1])
    #minor_ticks = np.arange(-1,1,0.05)
    #ax.set_yticks(minor_ticks) #, minor=True)
    
    if medium == 'article' :
        plt.title(legend_ds[dataset], fontsize=20, pad=10)
        if bias_type == 'label':
            ax.set_xlabel("Bias intensity, $\\beta_l$", size=18)
        else :
            ax.set_xlabel("Bias intensity, $p_u$", size=18)
        if dataset == 'OULADsocial':
            if legend_in_plot :
                handles, labels = plt.gca().get_legend_handles_labels()
                if bias_type == 'selectPrivNoUnpriv': # 'sens_attr_usage' not included
                    order = [0,2,3,4,5,1,6]
                    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],prop={'size':13}, loc='lower left', ncols = 2)
                else :
                    #order in handles : ['accBal','GenEntropyIndex','BCC', 'FPR', 'FalseNegRateDiff', 'FalsePosRateDiff', 'StatParity','sens_attr_usage']
                    #order_for_legend : ['accBal','BCC' ,'sens_attr_usage','FPR', 'FalseNegRateDiff', 'FalsePosRateDiff', 'GenEntropyIndex', 'StatParity']
                    order = [0,2,7,3,4,5,1,6]
                    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],prop={'size':13}, loc='lower left', ncols = 2)
        elif bias_type == 'selectRandomWhole' :
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0,2,3,7,5,1,4,6]
            ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],prop={'size':13}, loc='lower left', ncols = 2)
        postfix = "_article.pdf"
        extension = "pdf"
    elif medium == 'articleOld' :
        #Add appropriate labels and legends
        handles, labels = plt.gca().get_legend_handles_labels()
        if bias_type == 'selectPrivNoUnpriv':
            plt.title(legend_ds[dataset], fontsize=20, pad=10)
            ax.set_xlabel("Bias intensity, $p_u$", size=18)
            if dataset == 'OULADsocialHarder' :
                order = [2,0,3,1,5,6,4]
                ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],prop={'size':14}, loc='lower left', ncols = 2)    
        else :
            if dataset in ['student','OULADstem','OULADsocial']:
                if bias_type == 'label':
                    plt.title(legend_ds[dataset], fontsize=20, pad=10)
                """
                    ax.set_ylim(bottom=-1)
                elif bias_type == 'selectRandom':
                    ax.set_ylim(bottom=-0.25)
                elif bias_type == 'selectLow':
                    ax.set_ylim(bottom=-0.25)
                elif bias_type == 'selectDoubleProp':
                    ax.set_ylim(bottom=-0.75)
                """
                if dataset == 'OULADsocial':
                    ax.set_ylabel(legend_bias_long[bias_type], size = 20)
                if bias_type == 'selectDoubleProp':
                    ax.set_xlabel("Bias intensity ($\\beta_l$ or $p_u$)", size=18)
            else :                
                #metric_list = ['accBal','GenEntropyIndex','BCC', 'FPR', 'FalseNegRateDiff', 'FalsePosRateDiff', 'StatParity','sens_attr_usage']
                if bias_type == 'selectRandom':
                    ax.set_ylabel(legend_ds[dataset], size = 20)
                if dataset == 'studentBalanced' :
                    plt.title(legend_bias_long[bias_type], fontsize=20, pad=10)
                    ax.set_xlabel("Bias intensity, $p_u$", size=18)
                    if bias_type == 'selectRandom':
                        #order = [1,5,0,2,7,3,6,4]
                        order = [2,0,3,5,7,1,6,4]
                        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],prop={'size':14}, loc='lower left', ncols = 2)
                elif dataset == 'OULADsocialHarder' :
                    plt.title(legend_bias_long[bias_type], fontsize=20, pad=10)
                elif dataset == 'OULADstemHarder' :
                    ax.set_xlabel("Bias intensity, $p_u$", size=18)
                    if bias_type == 'selectRandom':
                        #order = [2,4,0,3,5,6,7,1]
                        order = [2,7,0,3,1,4,5,6]
                        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],prop={'size':14}, loc='lower left', ncols = 2)
                else :
                    print("WARNING Not a valid dataset name to plot_metrics for article")
        postfix = '_article.pdf'
        extension = "pdf"
    elif medium == 'legend' :
        plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.1), fancybox=True, ncols = 4) #prop={'size':16},  mode='expand'
        postfix = "_legend.png"
        extension = "png"
    else :
        if title != '':
            plt.title(title, fontsize=14)
        if bias_type[0:5] == 'label': 
            ax.set_xlabel('Bias intensity $\\beta_l$', size=14)
        elif bias_type[0:6] == 'select' : # selection bias
            ax.set_xlabel('Bias intensity $p_u$', size=14)
        else :
            print("WARNING Wrong bias type for plotting legend (plot_metrics())")
        #ax.legend(loc='best')
        ax.legend(prop={'size':7}, loc='best') #lower right
        postfix = '.pdf'
        extension = "pdf"
    
    #ax.grid(which='both',linestyle='-',linewidth=0.5,color='lightgray')
    ax.grid(visible=True)
    ax.grid(which='minor',linestyle=':',linewidth=0.3,color='lightgray')
    ax.minorticks_on()

    if path_start is not None :
            plt.savefig(path_start+postfix, format=extension, bbox_inches="tight", dpi=1000) #dpi changes image quality
    if(display) :
        plt.show()
    plt.close()

def biasing_unmitig_models(retrieval_path:str, dataset_list:list[str], metric_list:list[str],bias_list:list[str], model_list:list[str], bias_levels:list[float],
                           biased_test:bool=False, blind:bool=False, plot_style: str = 'FILLED_STDEV', medium='paper',title='', path_save=None, to1=False) :
    """ Apply plot-by_bias for all combinations of datasets, bias, preproc, model and blinding given as argument
        Each graph represent one mitigation situation, plotting evolution of divers metrics with the increasing intensity of the bias
        Dataset splits metrics for plots must already be saved and be store at 'path_start'
        Returns None
    """
    if to1 : pref = "0to1_"
    else : pref = ''
    if blind : visibility = "Blinded"
    else : visibility = "Aware"
    if biased_test : biased = "_Biased"
    else : biased = ''
    num_x = len(bias_levels)
    for ds in dataset_list :
        for model in model_list :
            metrics_all = {}
            for bias in bias_list :
                if to1 and bias[0:6]=='select' : pref = "0to1_"
                else : pref = ''
                path = retrieval_path+pref+ds+'_'+bias+'__'+model+visibility+"_"+biased+"_metricsForPlot.pkl"
                with open(path,"rb") as file:
                    metrics_for_plot = pickle.load(file)
                metrics_all[bias] = metrics_for_plot

            for metric in metric_list :

                fig, ax = plt.subplots() # (figsize=(length, heigth)) #Scale size of image
                ax.hlines(0,0,1,colors='black')

                colors_bias = {'label':colorblindIBM['orange'], 'selectRandom': colorblindIBM['magenta'], 'selectLow': colorblindIBM['indigo'], 'selectDoubleProp':colorblindIBM['ultramarine']}
                makers_bias = {'label':"o", 'selectRandom': "v", 'selectLow': "s", 'selectDoubleProp': "P"}
                if medium == 'slide' : markersize = 12
                else : markersize = None
                for bias in bias_list :
                    mean_vector = metrics_all[bias][metric]['mean'][0:num_x]
                    stdev_vector = metrics_all[bias][metric]['stdev'][0:num_x]
                    ax.plot(bias_levels,mean_vector, label = legend_bias_abrev[bias], linestyle="--", marker=makers_bias[bias], markersize=markersize,color=colors_bias[bias])
                    if plot_style == 'FILLED_STDEV':
                    #Shade for std values
                        ax.fill_between(bias_levels, mean_vector-stdev_vector, mean_vector+stdev_vector, edgecolor = None, facecolor=colors_bias[bias], alpha=0.4)

                if medium == 'slide' :
                    labelsize = '16'
                    legendsize = 16
                    if metric == 'acc':
                        ax.legend(prop={'size':legendsize}, loc='best')
                else :
                    labelsize = 'large'
                    legendsize = 12
                    ax.set_xlabel("Bias intensity ($\\beta_l$ or $p_u$)", size=14)
                    ax.legend(prop={'size':legendsize}, loc='best')
                ax.tick_params(labelsize = labelsize,which='major')
                ax.grid(visible=True)
                ax.grid(which='minor',linestyle=':',linewidth=0.3,color='lightgray')
                ax.minorticks_on()

                if metric == 'acc':
                    ax.set_ylim([0.5,1])
                ax.set_xlim([0,0.925])

                if title == None :
                    if biased_test: test_set = "biased test set"
                    else : test_set = "unbiased test set"
                    displayed_title ="Evolution of "+metric+ " with different types of bias \n("+ds+", "+model+", "+visibility+", "+test_set+")"
                else :
                    displayed_title = title
                plt.title(displayed_title, fontsize=14)

                if path_save is not None :
                    path = path_save+pref+ds+'_biases_'+metric+'_'+model+visibility
                    if medium == 'slide' :
                        plt.savefig(path+".png", format="png", bbox_inches="tight", dpi=1000) #dpi changes image quality
                    else :
                        plt.savefig(path+".pdf", format="pdf", bbox_inches="tight")
                else :
                    plt.show()

                plt.close()

def plot_tradeoff(retrieval_path:str, metric:str, dataset_list:list[str], model_list:list[str], bias:str,  all_bias:list[float], biased_test:bool,
                  preproc_list=['','reweighting','massaging'], postproc_list = ['eqOddsProc', 'calEqOddsProc','ROC-SPD','ROC-EOD', 'ROC-AOD'],
                  plot_style:str = 'FILLED_STDEV', title: str = '', plot_path:str = None, display=True) :
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
    #ax.hlines(0,0,0.91,colors="black")
    ax.axhline(0, linewidth=1, color=colors[''])#Black

    indices = np.array(all_bias)
    #fuzzy_start = -0.0035
    fuzzy = 0.0025
    for ds in dataset_list :
        for model in model_list :
            if plot_style == 'FILLED_STDEV':
                ax.fill_between(indices,metrics_all[ds][model][''][metric]['mean'] - metrics_all[ds][model][''][metric]['stdev'], metrics_all[ds][model][''][metric]['mean'] + metrics_all[ds][model][''][metric]['stdev'], edgecolor = None, facecolor='black', alpha=0.3)
                ax.fill_between(indices-4*fuzzy,metrics_all[ds][model]['reweighting'][metric]['mean'] - metrics_all[ds][model]['reweighting'][metric]['stdev'], metrics_all[ds][model]['reweighting'][metric]['mean'] + metrics_all[ds][model]['reweighting'][metric]['stdev'], edgecolor = None, facecolor=colors['reweighting'], alpha=0.3)
                ax.fill_between(indices-3*fuzzy,metrics_all[ds][model]['massaging'][metric]['mean'] - metrics_all[ds][model]['massaging'][metric]['stdev'], metrics_all[ds][model]['massaging'][metric]['mean'] + metrics_all[ds][model]['massaging'][metric]['stdev'], edgecolor = None, facecolor=colors['massaging'], alpha=0.3)
                ax.fill_between(indices-2*fuzzy,metrics_all[ds][model]['FTU'][metric]['mean'] - metrics_all[ds][model]['FTU'][metric]['stdev'], metrics_all[ds][model]['FTU'][metric]['mean'] + metrics_all[ds][model]['FTU'][metric]['stdev'], edgecolor = None, facecolor=colors['FTU'], alpha=0.3)
                ax.fill_between(indices-1*fuzzy,metrics_all[ds][model]['eqOddsProc'][metric]['mean'] - metrics_all[ds][model]['eqOddsProc'][metric]['stdev'], metrics_all[ds][model]['eqOddsProc'][metric]['mean'] + metrics_all[ds][model]['eqOddsProc'][metric]['stdev'], edgecolor = None, facecolor=colors['eqOddsProc'], alpha=0.3)
                ax.fill_between(indices+1*fuzzy,metrics_all[ds][model]['calEqOddsProc'][metric]['mean'] - metrics_all[ds][model]['calEqOddsProc'][metric]['stdev'], metrics_all[ds][model]['calEqOddsProc'][metric]['mean'] + metrics_all[ds][model]['calEqOddsProc'][metric]['stdev'], edgecolor = None, facecolor=colors['calEqOddsProc'], alpha=0.3)
                ax.fill_between(indices+2*fuzzy,metrics_all[ds][model]['ROC-SPD'][metric]['mean'] - metrics_all[ds][model]['ROC-SPD'][metric]['stdev'], metrics_all[ds][model]['ROC-SPD'][metric]['mean'] + metrics_all[ds][model]['ROC-SPD'][metric]['stdev'], edgecolor = None, facecolor=colors['ROC-SPD'], alpha=0.3)
                ax.fill_between(indices+3*fuzzy,metrics_all[ds][model]['ROC-EOD'][metric]['mean'] - metrics_all[ds][model]['ROC-EOD'][metric]['stdev'], metrics_all[ds][model]['ROC-EOD'][metric]['mean'] + metrics_all[ds][model]['ROC-EOD'][metric]['stdev'], edgecolor = None, facecolor=colors['ROC-EOD'], alpha=0.3)
                ax.fill_between(indices+4*fuzzy,metrics_all[ds][model]['ROC-AOD'][metric]['mean'] - metrics_all[ds][model]['ROC-AOD'][metric]['stdev'], metrics_all[ds][model]['ROC-AOD'][metric]['mean'] + metrics_all[ds][model]['ROC-AOD'][metric]['stdev'], edgecolor = None, facecolor=colors['ROC-AOD'], alpha=0.3)
                
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
                 
    #ax.tick_params(labelsize = 'large',which='major')
    major_ticks = np.arange(-1,1.2,0.2)
    ax.set_yticks(major_ticks) #, minor=True)
    
    if metric == 'acc':
        if bias == 'selectLow':
            ax.set_ylim([0.4,1])
        elif bias == 'selectDoubleProp':
            ax.set_ylim([0.55,1])
        elif bias == 'label':
            ax.set_ylim([0.45,1])
    elif metric == 'StatParity':
        if bias == 'selectLow':
            ax.set_ylim([-0.1,1])
        elif bias == 'selectDoubleProp':
            ax.set_ylim([-0.9,0.45])
        elif bias == 'label':
            ax.set_ylim([-0.5,1.1])
    else : 
        if bias == 'selectLow':
            ax.set_ylim([-0.05,1])
        elif bias == 'selectDoubleProp':
            ax.set_ylim([-0.05,1])
        elif bias == 'label':
            ax.set_ylim([-0.05,1])
    ax.set_xlim([0,0.91])
    
    if metric != 'acc':
        if bias == 'label': 
            ax.set_xlabel('Bias intensity $\\beta_l$', size=14)
        else : # selection bias
            ax.set_xlabel('Bias intensity $p_u$', size=14)
    
    # Name of metric on leftmost graph in paper
    if not biased_test :
        try :  y_label = legend_metric[metric]
        except KeyError : y_label = metric
        ax.set_ylabel(y_label, size = 20)
    else :
        ax.set_ylabel(" ", size = 20)
    
    #ax.grid(which='both',linestyle='-',linewidth=0.5,color='lightgray')
    #ax.grid(visible=True)
    ax.grid(which='both',linestyle=':',linewidth=0.3,color='lightgray')
    ax.minorticks_on()

    #ax.legend(loc='best')
    if not biased_test and metric == 'acc':
        ax.legend(prop={'size':12}, loc='lower left') #, 'lower left' bbox_to_anchor=(1, 0.87))

    if title != '':
        plt.title(title, fontsize=14)

    if plot_path is not None :
            if biased_test : test = "BiasedTest"
            else : test = "FairTest"
            path=plot_path+"tradeoff_"+bias+'_'+metric+'_'+ds+'_'+model+'_'+test+".pdf"
            plt.savefig(path, format="pdf", bbox_inches="tight", dpi=1000) #dpi changes image quality #
    if(display) :
        plt.show()
    plt.close()

def plot_all_tradeoff(retrieval_path:str, bias_list:list[str], metric_list:list[str], dataset_list:list[str], model_list:list[str], plot_style:str = 'FILLED_STDEV',  plot_path:str = None) :
    bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for bias_type in bias_list :
        for ds in dataset_list :
            for model in model_list :
                for metric in metric_list :
                    plot_tradeoff(retrieval_path,metric,[ds],[model],bias_type,bias_levels,biased_test=False,plot_style=plot_style,plot_path=plot_path,display = False)
                    plot_tradeoff(retrieval_path,metric,[ds],[model],bias_type,bias_levels,biased_test=True,plot_style=plot_style,plot_path=plot_path,display = False)


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
                        plt.title(ds, fontsize=20, pad=10)

                if path_start is not None :
                        plt.savefig(path_start+ds+"_"+bias+"_"+metric+"_study.pdf", format="pdf", bbox_inches="tight") # dpi=1000) #dpi changes image quality
                if(display) :
                    plt.show()
                plt.close()

def bargraph_all_methods(retrieval_path:str, dataset_list=['studentBalanced','OULADstem', 'OULADsocial'], metric_list=['acc','StatParity','EqqOddsDiff','GenEntropyIndex'],
                  bias_list=['label','selectDoubleProp','selectLow'], preproc_list=['', 'reweighting','massaging'], postproc_list = ['eqOddsProc', 'calEqOddsProc','ROC-SPD','ROC-EOD', 'ROC-AOD'],
                  all_bias:list[float]=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  medium = None, plot_style:str = 'FILLED_STDEV', title: str = '', path_start:str = None, display=False) :
    """ Plot comparison of all mitigation methods in a bar graph.
    plot_style : string, optional
        'FILLED_DEV' for display of standard deviation
        'SIMPLE_PLOT' for choice of metrics displaid without standard deviation
    medium : 'article', 'articleAppendix' or None
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
                figsize=(6.4, 4.8)
                if (ds == 'OULADstem' or ds == 'studentBalanced') and metric != 'BCC' :
                    figsize=(9, 4.8)
                fig, ax = plt.subplots(figsize=figsize) # (figsize=(length, heigth)) #Scale size of image
                #ax.hlines(y=0,xmin=-0.05,xmax=1,colors='black')

                bar_width = 0.009
                indices = np.array(all_bias)
                #colors = {'':'black', 'reweighting': "#C85200", 'massaging': "#FF800E", 'FTU':"#FFBC79",
                #          'eqOddsProc':"#006BA4", 'calEqOddsProc':"#5F9ED1", 'ROC-SPD':"#A2C8EC", 'ROC-EOD':"#595959"}

                ax.axhline(0, linewidth=0.5, color=colors[''])#Black
                #Reference - mitigation
                ax.bar(indices - 4*bar_width,metrics_all[ds][bias][''][metric]['mean'], label = 'biased model', width=bar_width,color=colors[''])#Black
                #plt.axhline(metrics_all[''][fair_metric]['mean'][0], color='black',linestyle=':',linewidth=1,alpha = 0.5)
                ax.axhline(metrics_all[ds][bias][''][metric]['mean'][0], linestyle = 'dashed', linewidth=0.5, color=colors[''], alpha=0.6)#Black
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
                

                if medium == 'article':
                    if ds == 'OULADstem' or ds == 'studentBalanced':
                        if metric == 'acc':
                            if bias == 'selectLow':
                                ax.set_ylim([0,1])
                            else :
                                ax.set_ylim([0.3,1])
                        elif metric == 'EqqOddsDiff':
                            ax.set_ylim(0,1.05)
                        elif metric == 'StatParity':
                            if bias == 'label':
                                ax.set_ylim([-0.8,0.4])
                            elif bias == 'selectDoubleProp':
                                ax.set_ylim([-0.6,0.45])
                            elif bias == 'selectLow':
                                ax.set_ylim([-0.21,1])
                        elif metric == 'BlindCons':
                            ax.set_ylim([0.4,1.2])
                        elif metric == 'BCC': 
                            ax.set_ylim([0,1]) #ax.set_ylim([0.2,1.05])
                        elif metric == 'GenEntropyIndex':
                            if bias == 'label':
                                ax.set_ylim([0.0,1])
                            else :
                                ax.set_ylim([0.0,0.5])
                    elif ds == 'OULADsocial' :
                        if metric == 'acc':
                            ax.set_ylim([0.3,1])
                        elif metric == 'EqqOddsDiff':
                            ax.set_ylim(0,1.05)
                        elif metric == 'StatParity':
                                ax.set_ylim([-0.55,0.95])
                        elif metric == 'BlindCons':
                            ax.set_ylim([0.4,1.2])
                        elif metric == 'BCC': 
                            ax.set_ylim([0,1]) #ax.set_ylim([0.2,1.05])
                        elif metric == 'GenEntropyIndex':
                            ax.set_ylim([0.0,0.55])

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
                if medium == 'article' and ds in ['OULADstem','studentBalanced'] and metric in ['acc','StatParity','EqqOddsDiff']:
                    pass
                else :
                    xlabels = {'label':"Bias intensity $\\beta_l$", 'selec':"Bias intensity $p_u$"}
                    ax.set_xlabel(xlabels[bias[0:5]], size=18)
                
                if medium == 'article' :
                    if ds == 'OULADstem' or ds == 'studentBalanced' : # Name of metric on lefmost graph
                        ax.set_ylabel(legend_metric[metric], size = 20)
                        if metric == 'acc':
                            ax.legend(loc='lower left')
                            plt.title(legend_ds[ds], fontsize=25, pad=20)
                    elif ds == 'studentBalanced' :
                        plt.title(legend_ds[ds], fontsize=25, pad=10)
                    elif ds == 'OULADsocial' :
                        if bias == 'label' :
                            ax.set_ylabel(legend_metric[metric], size = 20)
                            if metric == 'acc' :
                                ax.legend(loc='lower left')
                        if metric == 'acc' :
                            plt.title(legend_bias_long[bias], fontsize=25, pad=20)
                        ax.set_xlabel(xlabels[bias[0:5]], size=18)
                    if metric == 'BCC' :
                        ax.set_ylabel(legend_ds[ds], size = 20)
                        if ds == 'OULADstem' :
                            plt.title(legend_bias_long[bias], fontsize=25, pad=20)
                            if bias == 'label' :
                                ax.legend(loc='upper left')
                else :
                    ax.set_ylabel(legend_metric[metric], size = 20)
                    ax.legend(loc='best')
                    if title is None:
                        plt.title(metric+" values wrt train set "+bias+" bias level :\n ("+ds+", unbiased test set)" , fontsize=14)
                    elif title != '' :
                        plt.title(title, fontsize=14)                    
                
                #minor_ticks = np.arange(-1,1,0.05)
                #ax.set_yticks(minor_ticks) #, minor=True)
                #ax.set_xticks(all_bias)  # Set x-ticks to all_bias values
                #ax.set_xticklabels(['0','0.2','0.4','0.6','0.8'])
                ax.tick_params(labelsize = 'large',which='major')
                ax.grid(which='both',linestyle=':',linewidth=0.5,color='lightgray')
                ax.minorticks_on()

                if path_start is not None :
                        plt.savefig(path_start+ds+"_"+bias+"_"+metric+".pdf", format="pdf", bbox_inches="tight", dpi=1000) #dpi changes image quality #,bbox_inches=bbox_inches
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
                try :
                    y_label = legend_metric[metric]
                except KeyError :
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
                    try :
                        y_label = legend_metric[metric]
                    except KeyError :
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

    ax.plot(metrics_all['']['acc']['mean'],metrics_all[''][fair_metric]['mean'], c=colors[''], label=legend_method[''], alpha=0.8)
    ax.scatter(metrics_all['reweighting']['acc']['mean'],metrics_all['reweighting'][fair_metric]['mean'], c=colors['reweighting'], label=legend_method['reweighting'], alpha=0.8)
    ax.scatter(metrics_all['massaging']['acc']['mean'],metrics_all['massaging'][fair_metric]['mean'], c=colors['massaging'], label=legend_method['massaging'], alpha=0.8)
    ax.scatter(metrics_all['FTU']['acc']['mean'],metrics_all['FTU'][fair_metric]['mean'], c=colors['FTU'], label=legend_method['FTU'], alpha=0.8)
    ax.scatter(metrics_all['eqOddsProc']['acc']['mean'],metrics_all['eqOddsProc'][fair_metric]['mean'], c=colors['eqOddsProc'], label=legend_method['eqOddsProc'], alpha=0.8)
    ax.scatter(metrics_all['calEqOddsProc']['acc']['mean'],metrics_all['calEqOddsProc'][fair_metric]['mean'], c=colors['calEqOddsProc'], label=legend_method['calEqOddsProc'], alpha=0.8)
    ax.scatter(metrics_all['ROC-SPD']['acc']['mean'],metrics_all['ROC-SPD'][fair_metric]['mean'], c=colors['ROC-SPD'], label=legend_method['ROC-SPD'], alpha=0.8)
    ax.scatter(metrics_all['ROC-EOD']['acc']['mean'],metrics_all['ROC-EOD'][fair_metric]['mean'], c=colors['ROC-EOD'], label=legend_method['ROC-EOD'], alpha=0.8)
    
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
                  postproc_list = ['eqOddsProc', 'calEqOddsProc','ROC-SPD','ROC-EOD','ROC-AOD'], medium='article',title: str = '', path_start:str = None, display=False) :
    """ Plot comparison of several mitigation methods in a bar graph
    medium : 'slide', 'article' or 'articleAppendix'
    title : string, optional
        None for automatic title
        '' (empty string) for no title
    """
    metrics_all = {}
    for ds in dataset_list :
        metrics_all[ds] = {}
        if ds[0:7] == 'student':
            model_list = ['RF','tree']
        else : model_list = ['RF']
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

    fig, ax = plt.subplots(figsize=(5, 5)) #Scale size of image
    ax.grid(which='major',linestyle=':',linewidth=0.5,color='lightgray')
    #ax.minorticks_on()
    #ax.grid(visible=True)
    if medium == 'article' :
        min, max = -1, 1
        ax.plot([min, max], [min, max], linestyle=':', color='black', zorder=1)
        ax.set_ylim([min, max])
        ax.set_xlim([min, max])
    else :
        if bias_type == 'label':
            min, max = -1, 2.8
            ax.plot([min, max], [min, max], linestyle=':', color='black', zorder=1)
            ax.set_ylim([min, max])
            ax.set_xlim([min, max])
        elif bias_type == 'selectDoubleProp':
            min, max = -1, 3
            ax.plot([min, max], [min, max], linestyle=':', color='black', zorder=1)
            ax.set_ylim([min, max])
            ax.set_xlim([min, max])
        elif bias_type == 'selectLow':
            min, max = -0.25,4
            ax.plot([min, max], [min, max], linestyle=':', color='black', zorder=1)
            ax.set_ylim([min, max])
            ax.set_xlim([min, max])
        elif bias_type == 'selectRandom' :
            min, max = -0.25, 3.2
            ax.plot([min, max], [min, max], linestyle=':', color='black', zorder=1)
            ax.set_ylim([min, max])
            ax.set_xlim([min, max])
    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)   
    plt.axis('scaled')    

    #min_val = min(min(metrics_all[proc]['acc']['mean']), min(metrics_all[proc][fair_metric]['mean']))
    #max_val = max(max(metrics_all[proc]['acc']['mean']), max(metrics_all[proc][fair_metric]['mean']))

    colorblindIBM = {'ultramarine': "#648FFF", 'indigo':"785EF0", 'magenta':"#DC267F", 'orange':"#FE6100", 'gold':"#FFB000",
                 'black':"#000000", 'white':"#FFFFFF"}
    m_colors = {'acc':"#785EF0", 'BlindCons':"#DC267F", 'BCC':"#DC267F", 'StatParity': "#FE6100", 'EqqOddsDiff': "#FFB000", 'GenEntropyIndex':"#648FFF"}
    m_legend = {'StatParity': "SPD", 'EqqOddsDiff': "EqOd", 'acc':"Accuracy", 'BlindCons':"Consistency", 'BCC':"BCC", 'GenEntropyIndex':"GEI"} 
    m_marker = {'acc':"o", 'BlindCons':"p", 'BCC':"p", 'StatParity': "d", 'EqqOddsDiff': "P", 'GenEntropyIndex':"p"}
    alpha = 0.4
    label = True
    if medium == 'slide' : markersize = 80
    else : markersize = None
    for ds in dataset_list:
        for model in metrics_all[ds].keys() :
            for proc in metrics_all[ds][model].keys():
                if label :
                    ax.scatter(metrics_all[ds][model][proc]['fair']['acc']['mean'][1:],metrics_all[ds][model][proc]['biased']['acc']['mean'][1:], facecolors=m_colors['acc'], edgecolors = m_colors['acc'], label=m_legend['acc'], marker=m_marker['acc'], s=markersize, alpha=alpha)
                    #ax.scatter(metrics_all[ds][model][proc]['fair']['BlindCons']['mean'][1:],metrics_all[ds][model][proc]['biased']['BlindCons']['mean'][1:], facecolors=m_colors['BlindCons'], edgecolors = m_colors['BlindCons'], label=m_legend['BlindCons'], marker=m_marker['BlindCons'], s=markersize, alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['BCC']['mean'][1:],metrics_all[ds][model][proc]['biased']['BCC']['mean'][1:], facecolors=m_colors['BCC'], edgecolors = m_colors['BCC'], label=m_legend['BCC'], marker=m_marker['BCC'], s=markersize, alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['StatParity']['mean'][1:],metrics_all[ds][model][proc]['biased']['StatParity']['mean'][1:], facecolors=m_colors['StatParity'], edgecolors = m_colors['StatParity'], label=m_legend['StatParity'], marker=m_marker['StatParity'], s=markersize, alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['GenEntropyIndex']['mean'][1:],metrics_all[ds][model][proc]['biased']['GenEntropyIndex']['mean'][1:], facecolors=m_colors['GenEntropyIndex'], edgecolors = m_colors['GenEntropyIndex'], label=m_legend['GenEntropyIndex'], marker=m_marker['GenEntropyIndex'], s=markersize, alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['EqqOddsDiff']['mean'][1:],metrics_all[ds][model][proc]['biased']['EqqOddsDiff']['mean'][1:], facecolors=m_colors['EqqOddsDiff'], edgecolors = m_colors['EqqOddsDiff'], label=m_legend['EqqOddsDiff'], marker=m_marker['EqqOddsDiff'], s=markersize, alpha=alpha)
                else :
                    ax.scatter(metrics_all[ds][model][proc]['fair']['acc']['mean'][1:],metrics_all[ds][model][proc]['biased']['acc']['mean'][1:], facecolors=m_colors['acc'], edgecolors = m_colors['acc'], label=None, marker=m_marker['acc'], s=markersize, alpha=alpha)
                    #ax.scatter(metrics_all[ds][model][proc]['fair']['BlindCons']['mean'][1:],metrics_all[ds][model][proc]['biased']['BlindCons']['mean'][1:], facecolors=m_colors['BlindCons'], edgecolors = m_colors['BlindCons'], label=None, marker=m_marker['BlindCons'], s=markersize, alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['BCC']['mean'][1:],metrics_all[ds][model][proc]['biased']['BCC']['mean'][1:], facecolors=m_colors['BCC'], edgecolors = m_colors['BCC'], label=None, marker=m_marker['BCC'], s=markersize, alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['StatParity']['mean'][1:],metrics_all[ds][model][proc]['biased']['StatParity']['mean'][1:], facecolors=m_colors['StatParity'], edgecolors = m_colors['StatParity'], label=None, marker=m_marker['StatParity'], s=markersize, alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['GenEntropyIndex']['mean'][1:],metrics_all[ds][model][proc]['biased']['GenEntropyIndex']['mean'][1:], facecolors=m_colors['GenEntropyIndex'], edgecolors = m_colors['GenEntropyIndex'], label=None, marker=m_marker['GenEntropyIndex'], s=markersize, alpha=alpha)
                    ax.scatter(metrics_all[ds][model][proc]['fair']['EqqOddsDiff']['mean'][1:],metrics_all[ds][model][proc]['biased']['EqqOddsDiff']['mean'][1:], facecolors=m_colors['EqqOddsDiff'], edgecolors = m_colors['EqqOddsDiff'], label=None, marker=m_marker['EqqOddsDiff'], s=markersize, alpha=alpha)
                label = False

    if medium == 'slide' :
        labelsize = '16'
        legendsize = 14
        if bias_type == 'selectLow':
            ax.legend(prop={'size':legendsize}, loc='best')
        postfix = '_slide'
    if medium[0:7] == 'article' :
        labelsize = 'large'
        legendsize = 14
        xy_label = 18
        if bias_type == 'label' or bias_type == 'selectLow':
            ax.set_ylabel("Biased evaluation", size = xy_label)
        if bias_type == 'label' :
            ax.legend(prop={'size':legendsize}, loc='best')
        ax.set_xlabel("Fair evaluation", size=xy_label)
        if medium == 'article' : postfix = '_article'
        else : postfix = '_appendix'
    else :
        labelsize = 'large'
        legendsize = 14
        xy_label = 20
        ax.set_ylabel("Biased evaluation", size = xy_label)
        ax.set_xlabel("Fair evaluation", size=xy_label)
        ax.legend(prop={'size':legendsize}, loc='best')
        postfix = ''
    ax.tick_params(labelsize = labelsize,which='major')

    if medium[0:7] == 'article':
        plt.title(legend_bias_long[bias_type], fontsize=20, pad=10)
    elif title is None:
        plt.title("Fair versus biased evaluation of metrics for "+bias_type , fontsize=14)
    elif title != '' :
        plt.title(title, fontsize=14)
    #No title for title == ''

    if path_start is not None :
        plt.savefig(path_start+"biasVSunbiased_"+bias_type+postfix+".pdf", format="pdf", dpi=1000) #dpi changes image quality # bbox_inches="tight"
    if(display) :
        plt.show()
    plt.close()