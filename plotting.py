import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import numpy as np
import pickle
import math
import gc
import sys
sys.path.append('..')

colors = {'':'black', 'reweighting': "#C85200", 'massaging': "#FF800E", 'FTU':"#FFBC79"}
legend = {'':'Unmitigated', 'reweighting': "Reweighting", 'massaging': "Massaging", 'FTU':"FTU"}
colorblindIBM = {'ultramarine': "#648FFF", 'indigo':"785EF0", 'magenta':"#DC267F", 'orange':"#FE6100", 'gold':"#FFB000",
                 'black':"#000000", 'white':"#FFFFFF"}

def bargraph_EWAF2025(retrieval_path:str, dataset_list=['student','OULADstem', 'OULADsocial'], metric_list=['acc','StatParity','EqqOddsDiff','GenEntropyIndex'],
                  bias_list=['label','selectDoubleProp'],preproc_list=['', 'reweighting','massaging'],
                  ylim:list[float]=None, all_bias:list[float]=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  title: str = '', path_start:str = None, display=False) :
    """ Plot results as presented in EWAF2025 paper "Influence of Label and Selection Bias on Fairness Interventions"
    Plot a graph for each dataset with comparison of several mitigation methods in a bar graph, the x axis shows the values for bias intensity
    retrieval_path : path at which the different results for the model considered can be found
    dataset_list : list of datasets for which a plot should be computed
    metric_list : list of metrics that should be displayed in the graphs
    bias_list : list of bias type(s) that should be displayed in the graphs
    preproc_list : list of preprocessing function(s) that should be displayed in the graphs
    ylim : optional, limit of the y axis
    all_bias : list of bias level values that should be displayed on the x axis, must correspond to the values stored for the metrics
    title : string, optional
        None for automatic title
        '' (empty string) for no title
    path_start: String
        if not None, save plots on disk at address 'path_start' + function postfix
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
                  bias_list=['label','selectDoubleProp'],preproc_list=['', 'reweighting','massaging'],
                  style = 'slide', all_bias:list[float]=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  title: str = '', path_start:str = None, display=False) :
    """ Plot comparison of several mitigation methods in a bar graph with distinct graphs for selection and label bias
    retrieval_path : path at which the different results for the model considered can be found
    dataset_list : list of datasets for which a plot should be computed
    metric_list : list of metrics that should be displayed in the graphs
    bias_list : list of bias type(s) that should be displayed in the graphs
    preproc_list : list of preprocessing function(s) that should be displayed in the graphs
    ylim : optional, limit of the y axis
    style : 'slide' ou 'poster'
    all_bias : list of bias level values that should be displayed on the x axis, must correspond to the values stored for the metrics
    title : string, optional
        None for automatic title
        '' (empty string) for no title
    path_start: String
        if not None, save plots on disk at address 'path_start' + function postfix
    
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

def plot_all(data_list, bias_list, preproc_list, model_list, blinding_list, all_bias, retrieval_path) :
    """ Apply plot-by_bias for all combinations of datasets, bias, preproc, model and blinding given as argument
        Each graph represent one mitigation situation, plotting evolution of divers metrics with the increasing intensity of the bias
        Results will be saved in folder "plots_extra"
        retrieval_path : path at which the different results are stored
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
                        path = retrieval_path+ds+'_'+bias+'_noValid_'+preproc+'_'+model+visibility
                        with open(path+"__metricsForPlot.pkl","rb") as file:
                            metrics_for_plot = pickle.load(file)
                        with open(path+"__Biased"+"_metricsForPlot.pkl","rb") as file:
                            metricsBiased_for_plot = pickle.load(file)
                        """ Version with title
                        plot_by_bias(metricsBiased_for_plot, all_bias, bias, biased_test=True, plot_style='FILLED_STDEV', title='Metric values wrt train set bias level\n ('+bias+' bias, '+preproc+', '+model+visibility+', biased test set)',path_start='Code/ControlledBiasPrivate/plots/'+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_byBias_BiasedTest', display=False)
                        plot_by_bias(metrics_for_plot, all_bias, bias, biased_test=False, plot_style='FILLED_STDEV',title='Metric values wrt train set bias level\n ('+bias+' bias, '+preproc+', '+model+visibility+', unbiased test set)', path_start='Code/ControlledBiasPrivate/plots/'+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_byBias_unbiasedTest', display=False)
                        """
                        plot_by_bias(metricsBiased_for_plot, all_bias, bias, biased_test=True, plot_style='FILLED_STDEV', title='',path_start='plots_extra/'+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_byBias_BiasedTest', display=False)
                        plot_by_bias(metrics_for_plot, all_bias, bias, biased_test=False, plot_style='FILLED_STDEV',title='', path_start='plots_extra/'+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_byBias_unbiasedTest', display=False)
                        #Manage memory
                        del metrics_for_plot, metricsBiased_for_plot
                        gc.collect()

def plot_by_bias(metrics_by_bias, all_bias: list[float], bias_type:str, biased_test:bool, plot_style: str = 'SIMPLE_PLOT', title: str = '', path_start:str = None, display=True) :

    """ Plot results with each graph presenting several metrics for one specific combination of dataset, bias type, preprocessing metrics and blinding
        The x axis shows the values for bias intensity
    nk_results_dic : Dictionary {float: {int: {str: float}}}
        Nested dictionaries where nk_results_dic[b][f][metric_name] = Value of 'metric_name' obtained for fold nbr 'k' of model trained with bias level 'b'
    plot_style : string, optional
        'ALL' for basic display of all metrics
        'SIMPLE_PLOT' for choice of metrics displaid without standard deviation
        'FILLED_STDEV' for choice of metrics displaid with standard deviation as colored area arround the curve
    path_start: String
        if not None, save results dict on disk at address 'path_start' + function postfix
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
        ax.plot(all_bias,metrics_by_bias['acc']['mean'], label = 'Accuracy', linestyle="--",marker="o", color='#595959')##Dark gray
        ax.plot(all_bias,metrics_by_bias['BlindCons']['mean'], label = 'Consistency', linestyle="--",marker="p", color="#5F9ED1")#Picton blue
        ax.plot(all_bias,metrics_by_bias['FPR']['mean'], label = 'FPR global', linestyle="--",marker="+", color='#ABABAB')##Light gray
        ax.plot(all_bias,metrics_by_bias['FNR']['mean'], label = 'FNR global', linestyle="--",marker="x", color='#ABABAB')##Light gray
        #ax.plot(all_bias,metrics_by_bias['F1']['mean'], label = 'F1 score', linestyle="--",marker="o", color='#595959')##Dark gray
        #ax.plot(all_bias,metrics_by_bias['EqqOddsDiff']['mean'], label = 'Eq. Odds', linestyle="--",marker="X", c="#A2C8EC")#Seil/Light blue
        #ax.plot(all_bias,metrics_by_bias['EqqOppDiff']['mean'], label = 'EqOpp=TPRdiff', linestyle="--",marker="d", c="#FF800E")#Pumpkin/Bright orange
        ax.plot(all_bias,metrics_by_bias['FalsePosRateDiff']['mean'], label = 'FPR diff.', linestyle="--",marker="P", color="#FFBC79")#Mac and cheese orange
        ax.plot(all_bias,metrics_by_bias['FalseNegRateDiff']['mean'], label = 'FNR diff.', linestyle="--",marker="X", c='#FF800E')#Pumpkin/Bright orange
        ax.plot(all_bias,metrics_by_bias['GenEntropyIndex']['mean'], label = 'GEI', linestyle="--",marker=".", c="#A2C8EC")#Seil/Light blue
        ax.plot(all_bias,metrics_by_bias['StatParity']['mean'], label = 'SPD', linestyle="--",marker="d", c='#C85200')#Tenne/Dark orange

        if plot_style == 'FILLED_STDEV':
        #Shade for std values
            ax.fill_between(all_bias,metrics_by_bias['acc']['mean'] - metrics_by_bias['acc']['stdev'], metrics_by_bias['acc']['mean'] + metrics_by_bias['acc']['stdev'], edgecolor = None, facecolor='#595959', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['BlindCons']['mean'] - metrics_by_bias['BlindCons']['stdev'], metrics_by_bias['BlindCons']['mean'] + metrics_by_bias['BlindCons']['stdev'], edgecolor = None, facecolor='#5F9ED1', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['FPR']['mean'] - metrics_by_bias['FPR']['stdev'], metrics_by_bias['FPR']['mean'] + metrics_by_bias['FPR']['stdev'], edgecolor = None, facecolor='#ABABAB', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['FNR']['mean'] - metrics_by_bias['FNR']['stdev'], metrics_by_bias['FNR']['mean'] + metrics_by_bias['FNR']['stdev'], edgecolor = None, facecolor='#ABABAB', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['FalsePosRateDiff']['mean'] - metrics_by_bias['FalsePosRateDiff']['stdev'], metrics_by_bias['FalsePosRateDiff']['mean'] + metrics_by_bias['FalsePosRateDiff']['stdev'], edgecolor = None, facecolor='#FFBC79', alpha=0.4)
            #ax.fill_between(all_bias,metrics_by_bias['EqqOppDiff']['mean'] - metrics_by_bias['EqqOppDiff']['stdev'], metrics_by_bias['EqqOppDiff']['mean'] + metrics_by_bias['EqqOppDiff']['stdev'], edgecolor = None, facecolor='#FF800E', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['FalseNegRateDiff']['mean'] - metrics_by_bias['FalseNegRateDiff']['stdev'], metrics_by_bias['FalseNegRateDiff']['mean'] + metrics_by_bias['FalseNegRateDiff']['stdev'], edgecolor = None, facecolor='#FF800E', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['GenEntropyIndex']['mean'] - metrics_by_bias['GenEntropyIndex']['stdev'], metrics_by_bias['GenEntropyIndex']['mean'] + metrics_by_bias['GenEntropyIndex']['stdev'], edgecolor = None, facecolor='#A2C8EC', alpha=0.4)
            ax.fill_between(all_bias,metrics_by_bias['StatParity']['mean'] - metrics_by_bias['StatParity']['stdev'], metrics_by_bias['StatParity']['mean'] + metrics_by_bias['StatParity']['stdev'], edgecolor = None, facecolor='#C85200', alpha=0.4)

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
    if biased_test :
        ax.legend(prop={'size':9}, loc='lower left') #,  bbox_to_anchor=(1, 0.87))

    if title != '':
        plt.title(title, fontsize=14)

    if path_start is not None :
            plt.savefig(path_start+".pdf", format="pdf", bbox_inches="tight") # dpi=1000) #dpi changes image quality
    if(display) :
        plt.show()
    plt.close()


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
                path = retrieval_path+ds+'_'+bias+'_noValid_'+preproc+"_RFAware_metricsForPlot.pkl"
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

