from pandas import DataFrame
from datetime import timedelta
import pickle
import time
import gc

import sys 
sys.path.append('..')

from ControlledBias.dataset.studentMale_dataset import StudentMaleDataset
from ControlledBias.dataset.oulad_dataset import OULADDataset
from ControlledBias import dataset_biasing as db
from ControlledBias import model_training as mt
from ControlledBias import analyzing as a
from ControlledBias import fairness_intervention as fair

#Add path to the directory in which you placed the 'ControlledBias' folder.
import sys 
sys.path.append('..')

start = time.perf_counter()
stop = start
k = -1 #negative value will create error instead of silent mistake
path_start = "data/" #Location of saved (intermediate) results TODO was Code/ControlledBias/data/

#You can change here the datasets, biases, preprocessing methods and bias intensity to be used in experiment
datasets = ['student','OULADstem', 'OULADsocial']
biases = ['label','selectDoubleProp'] #Other possible options (not used in EWAF2025 publication) are 'labelDouble' and 'selectLow'
preproc_methods = ['', 'massaging', 'reweighting'] #Fairness through Unawareness corresponds to no preproc + blind_model = True
classifiers = ['RF'] #RF was used for the results in EWAF2025 publication, other options are 'tree' and 'neural'
blind_model = [True,False] #True to exclude sensitive attribute from features used in training, False to include it
bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

save = True #Whether results, included intermediate ones, will be saved. Necassary to obtain plots
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
                    del n_folds_lists
                    print("Label bias introduced")
                elif bias == 'selectLow' or bias == 'selectDoubleProp' :
                    if bias == 'selectDoubleProp' :
                        removal_distr = 'double_disc'
                    elif bias == 'selectLow' :
                        removal_distr = 'lower_weight'
                    nbias_data_dict = db.undersampling_incremental_nbias(dataset_orig, removal_distr, path_start=path) #bias_levels #CHANGE
                    print("Selection bias introduced")
                else :
                    print("WARNING Not a valid bias type")
                del nbias_data_dict

            #create train and test sets for each fold and bias, nk_train_splits[bias][fold]: {'train': train set, 'test': test set}
            if computed :
                try : #retrieve biased dataset split into k partitions
                    with open(path+"_nbias_splits_datasets.pkl","rb") as file:
                        nk_folds_dict = pickle.load(file)
                except IOError as err: #create n biased datasets and split them into k partitions
                    computed = False
            if not computed :
                if bias == 'label' or bias == 'labelDouble' : #same split for all folds
                    k = 10
                elif bias == 'selectDouble' or bias == 'selectLow' or bias == 'selectDoubleProp' : #different split for each fold
                    k = 5
                nk_folds_dict, _ = mt.common_splits(nbias_data_dict,k,path)
                #nk_folds_dict[bias]: list of test folds for each bias level
            if save : path = path_start+ds+'_'+bias+"_noValid_" #(_noValid_ because no validation set is used)
            if computed :
                try :#retrieve dict of train-test splits for all bias levels
                    with open(path+"_train-test-nk.pkl","rb") as file:
                        nk_train_splits = pickle.load(file)
                except IOError as err: #create n biased datasets and split them into k partitions
                    computed = False
            if not computed :
                nk_train_splits = mt.nk_merge_train(nk_folds_dict,valid=False,path_start=path)
                #nk_train_splits[bias][fold]: {'train': train set, 'test': test set}
            
            #apply preprocessing if needed
            for preproc in preproc_methods :
                if save : path = path_start+ds+'_'+bias+"_noValid_"+preproc
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
                        if save : path = path_start+ds+'_'+bias+"_noValid_"+preproc+'_'+model+visibility
                        if computed :
                            try :
                                with open(path+"_all.pkl","rb") as file:
                                    nk_models = pickle.load(file)
                            except IOError as err:
                                computed = False
                        if not computed :
                            nk_models = mt.classifier_nbias(model, preproc_dict, blinding=blind, path_start=path)
                        #Create predictions from data with no preprocessing (both original and biased test set)
                        if save :
                            path_biased = path +"_biasedTest"
                        if computed :
                            try :
                                with open(path+"_pred_all.pkl","rb") as file:
                                    n_pred = pickle.load(file)
                                with open(path_biased+"_pred_all.pkl","rb") as file:
                                    n_pred_bias = pickle.load(file)
                            except IOError as err:
                                computed = False
                        if not computed :
                            n_pred = mt.prediction_nbias(nk_models, nk_train_splits, set='test', pred_type='labels', blinding=blind, path_start=path)
                            n_pred_bias = mt.prediction_nbias(nk_models, nk_train_splits, set='test', pred_type='labels', blinding=blind, biased_test=True, path_start=path_biased)
                        #Manage memory
                        del nk_models, n_pred_bias, n_pred
                        end = time.perf_counter()
                        print("Experiment completed for "+str(ds)+' '+str(bias)+' '+str(preproc)+' '+str(model)+' in hh:mm:ss',timedelta(seconds=end-stop))
                        computed = True
                    computed = True
                #Manage memory
                del preproc_dict
                gc.collect()
                computed = True
            del nk_folds_dict, nk_train_splits
            gc.collect()
            computed = True
        computed = True

    end = time.perf_counter()
    print("Total experiment time : hh:mm:ss",timedelta(seconds=end-start))


run_expe(datasets,biases,preproc_methods,classifiers,blind_model,path_start=path_start)
# Compute metrics for all model produced (Uses a lot of RAM, if it exceeds you memory, running the script for one preprocessing method at a time should help)
#TODO check if this statement is still necessary
for ds in datasets :
    a.compute_all([ds],biases,preproc_methods,classifiers,blind_model,path_start=path_start)
# Produce the same bar graph as in EWAF2025 publication
metrics_list = ['acc','StatParity','EqqOddsDiff','GenEntropyIndex']
results_path = path_start
plot_path = "plotsEWAF/"
a.plot_bargraph(retrieval_path=results_path, dataset_list=datasets, path_start=plot_path)
#To plot extra graphs not included in the EWAF2025 publication, uncomment the following line :
a.plot_all(datasets,biases,preproc_methods,classifiers,blind_model,bias_levels,path_start=path_start)