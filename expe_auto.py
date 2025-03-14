from pandas import DataFrame
import pickle
import time
from datetime import timedelta
import sys 
sys.path.append('Code/')
sys.path.append('Code/parent_aif360')

from ControlledBias.dataset.studentMale_dataset import StudentMaleDataset
from ControlledBias.dataset.oulad_dataset import OULADDataset
from ControlledBias import dataset_biasing as db
from ControlledBias import dataset_creation as dc
from ControlledBias import model_training as mt
from ControlledBias import analyzing as a
from ControlledBias import fairness_intervention as fair

start = time.perf_counter()
stop = start
bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #, 1]
k = -1 #negative value will create error instead of silent mistake
path_start = "Code/ControlledBias/data/"
blind = False

datasets = ['student']#,'OULADstem', 'OULADsocial']  #, 'student' OULAD
biases = ['label'] #['selectDoubleProp','label', 'labelDouble', 'selectDouble','selectLow'] #['label','selectDouble','selectLow'] #,'selectDouble','selectLow' label
preproc_methods = ['massaging', 'reweighting'] #, 'reweighting', 'LFR','massaging'] #
#ftu corresponds to no preproc + blind model
classifiers = ['tree','RF','neural'] #,'tree','neural']
blind_model = [True,False]

save = True
display_plot = False


def run_expe(datasets,biases,preproc_methods,classifiers,blind_model,path_start):
    if not save :
        path_start = None
        path = None
        path_biased = None

    computed = True #Set to False to force recomputation of everything

    for ds in datasets :
        if computed :
            try :
                with open(path_start+ds+"_dataset",'rb') as file:
                    dataset_orig = pickle.load(file)
            except IOError as err:
                computed = False
        if not computed :
            if ds == 'student' :
                dataset_orig = StudentMaleDataset(balanced=False)
            elif ds == 'OULADsocial' :
                dataset_orig = OULADDataset(domain='social')
            elif ds == 'OULADstem' :
                dataset_orig = OULADDataset(domain='stem')
            else :
                print("WARNING Not a valid dataset name")
            if save :
                with open(path_start+ds+"_dataset",'wb') as file:
                    pickle.dump(dataset_orig,file)

        #computed = False 
        for bias in biases :
            if save : path = path_start+ds+'_'+bias
            if computed :
                try : #retrieve biased dataset split into k partitions
                    with open(path+"_nbias_splits_datasets.pkl","rb") as file:
                        nk_folds_dict = pickle.load(file)
                except IOError as err: #create n biased datasets and split them into k partitions
                    computed = False
            if not computed :   
                if bias == 'label' or bias == 'labelDouble' :
                    if ds == 'student' : noise = 0.1
                    else : noise = 0.2 #ds == 'OULADsocial' or ds == 'OULADstem'
                    if bias == 'label' : double_disc = False
                    else : double_disc = True #bias == 'labelDouble'
                    nbias_data_dict = db.mislabeling_nbias(dataset_orig, bias_levels, noise=noise, double_disc=double_disc,path_start=path)
                    k = 10
                    nk_folds_dict, n_folds_lists = mt.common_splits(nbias_data_dict,k,path)
                    del n_folds_lists
                    print("Label bias introduced")
                elif bias == 'selectDouble' or bias == 'selectLow' or bias == 'selectDoubleProp' :
                    if bias == 'selectDoubleProp' :
                        removal_distr = 'double_disc'
                    elif bias == 'selectDouble' :
                        removal_distr = 'double_radnom_disc'
                    elif bias == 'selectLow' :
                        removal_distr = 'lower_weight'
                    nbias_data_dict = db.undersampling_nbias(dataset_orig, bias_levels, removal_distr, path)
                    k = 5
                    nk_folds_dict = mt.random_splits(nbias_data_dict,k,path)
                    print("Selection bias introduced")
                else :
                    print("WARNING Not a valid bias type for making folds")
                del nbias_data_dict
            #create train and test sets for each fold and bias, nk_train_splits[bias][fold]: {'train': train set, 'test': test set}
            nk_train_splits = mt.nk_merge_train(nk_folds_dict)
            
            #apply preprocessing if needed
            for preproc in preproc_methods :
                if save : path = path_start+ds+'_'+bias+'_'+preproc
                if preproc == '' :
                    preproc_dict = nk_train_splits
                    print("No preproc was applied")
                else :
                    if computed :
                        try :
                            with open(path+'_nkdatasets.pkl',"rb") as file:
                                preproc_dict = pickle.load(file)
                        except IOError as err:
                            computed = False
                    if not computed :
                        preproc_dict = fair.apply_preproc(nk_train_splits, preproc, path_start=path)
                stop = time.perf_counter()
                print("All preproc done for "+str(ds)+' '+str(bias)+' '+str(preproc)+' at hh:mm:ss',timedelta(seconds=stop-start))

                for model in classifiers :
                    for blind in blind_model :
                        stop = time.perf_counter()
                        #Train models with preprocessed data
                        if blind :
                            visibility = 'Blinded'
                        else :
                            visibility = 'Aware'
                        if save : path = path_start+ds+'_'+bias+'_'+preproc+'_'+model+visibility
                        if computed :
                            try :
                                with open(path+"_all.pkl","rb") as file:
                                    nk_models = pickle.load(file)
                            except IOError as err:
                                computed = False
                        if not computed :
                            nk_models = mt.classifier_nbias(model, preproc_dict, blinding=blind, path_start=path)
                        #Create predictions from data with no preprocessing (both original and biased test set)
                        if save : path_biased = path +"_biasedTest"
                        if computed :
                            try :
                                with open(path+"_pred_all.pkl","rb") as file:
                                    n_pred = pickle.load(file)
                                with open(path_biased+"_pred_all.pkl","rb") as file:
                                    n_pred_bias = pickle.load(file)
                            except IOError as err:
                                computed = False
                        if not computed :
                            n_pred = mt.prediction_nbias(nk_models, nk_folds_dict, blinding=blind, path_start=path)
                            n_pred_bias = mt.prediction_nbias(nk_models, nk_folds_dict, blinding=blind, biased_test=True, path_start=path_biased)
                        #Manage memory
                        del nk_models, n_pred_bias, n_pred
                        #Create plots
                        """
                        all_metrics = a.get_all_metrics(nk_folds_dict, n_pred, path_start=path)
                        if save : path_biased = path+'_Biased'
                        all_metrics_biasedTest = a.get_all_metrics(nk_folds_dict, n_pred_bias, biased_test=True, path_start=path)

                        metrics_for_plot, all_bias = a.metrics_for_plot(all_metrics,path_start=path)
                        if save : path_biased = path+'_Biased'
                        metricsBiased_for_plot, all_bias = a.metrics_for_plot(all_metrics_biasedTest,path_start=path_biased)

                        a.plot_by_bias(metrics_for_plot, all_bias, plot_style='FILLED_STDEV',title='Metric values wrt train set bias level\n ('+bias+' bias, '+preproc+', '+model+visibility+', unbiased test set)', path_start='Code/ControlledBias/plots/'+ds+'_'+bias+'_'+preproc+'_'+model+'_byBias_unbiasedTest', display=display_plot)
                        a.plot_by_bias(metricsBiased_for_plot, all_bias, plot_style='FILLED_STDEV', title='Metric values wrt train set bias level\n ('+bias+' bias, '+preproc+', '+model+visibility+', biased test set)',path_start='Code/ControlledBias/plots/'+ds+'_'+bias+'_'+preproc+'_'+model+'_byBias_BiasedTest', display=display_plot)
                        """
                        end = time.perf_counter()
                        print("Experiment completed for "+str(ds)+' '+str(bias)+' '+str(preproc)+' '+str(model)+' in hh:mm:ss',timedelta(seconds=end-stop))
                #Manage memory
                del preproc_dict
            del nk_folds_dict, nk_train_splits

    end = time.perf_counter()
    print("Total experiment time : hh:mm:ss",timedelta(seconds=end-start))


#run_expe(datasets,biases,preproc_methods,classifiers,blind_model,path_start=path_start)
a.compute_all(datasets,biases,preproc_methods,classifiers,blind_model,path_start=path_start)
print("Start producing plots")
a.plot_all(datasets,biases,preproc_methods,classifiers,blind_model,bias_levels,path_start=path_start)
