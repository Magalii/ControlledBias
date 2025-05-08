from datetime import timedelta
import multiprocessing as mp
from pandas import DataFrame
import pickle
import time
import gc
import sys 
sys.path.append('..')

from dataset.studentMale_dataset import StudentMaleDataset
from dataset.oulad_dataset import OULADDataset
import dataset_biasing as db
import dataset_creation as dc
import model_training as mt
import analyzing as a
import fairness_intervention as fair

def run_expe(datasets, biases, bias_levels, preproc_methods, postproc_methods, classifiers, blind_model, path_start, save=True):
    start = time.perf_counter()

    save = True

    if not save :
        path_start = None
        path = None
        path_biased = None

    computed = True #Set to False to force recomputation of everything

    for ds in datasets :
        print("\b##### "+ds+" #####")
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
            print("#### "+bias+" ####")

            if save : path = path_start+ds+'_'+bias
            if computed :
                try : #retrieve biased datasets
                    with open(path+"_nBiasDatasets.pkl","rb") as file:
                        nbias_data_dict = pickle.load(file)
                except IOError as err:
                    computed = False
            if not computed :
                if bias == 'label' or bias == 'labelDouble' :
                    if ds == 'student' : noise = 0.1
                    else : noise = 0.2 #ds == 'OULADsocial' or ds == 'OULADstem'
                    if bias == 'label' : double_disc = False
                    else : double_disc = True #bias == 'labelDouble'
                    nbias_data_dict = db.mislabeling_nbias(dataset_orig, bias_levels, noise=noise, double_disc=double_disc,path_start=path)
                    print("Label bias introduced")
                elif bias == 'selectDouble' or bias == 'selectLow' or bias == 'selectDoubleProp' :
                    if bias == 'selectDoubleProp' :
                        removal_distr = 'double_disc'
                    elif bias == 'selectDouble' :
                        removal_distr = 'double_radnom_disc'
                    elif bias == 'selectLow' :
                        removal_distr = 'lower_weight'
                    nbias_data_dict = db.undersampling_incremental_nbias(dataset_orig, removal_distr, path_start=path) #bias_levels
                    print("Selection bias introduced : "+str(bias))
                else :
                    print("WARNING Not a valid bias type")

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
            if computed :
                try :#retrieve dict of train-test splits for all bias levels
                    with open(path+"_train-test-nk.pkl","rb") as file:
                        nk_train_splits = pickle.load(file)
                except IOError as err: #create n biased datasets and split them into k partitions
                    computed = False
            if not computed :
                nk_train_splits = mt.nk_merge_train(nk_folds_dict,valid=True,path_start=path)
                #nk_train_splits[bias][fold]: {'train': train set, 'valid': validation set, 'test': test set}

            #apply preprocessing if needed
            for preproc in preproc_methods :
                print("### "+preproc+" ###")
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
                
                # Train models
                for model in classifiers :
                    print("### "+model+" ###")
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
                        
                        #computed = False

                        if len(postproc_methods) == 0 : #Compute predicted labels directly 
                            #Create predictions from data (both original and biased test set, no preprocessing on test set)
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
                                n_pred = mt.prediction_nbias(nk_models, nk_train_splits, set='test', pred_type='labels', blinding=blind, path_start=path)
                                n_pred_bias = mt.prediction_nbias(nk_models, nk_train_splits, set='test', pred_type='labels', blinding=blind, biased_test=True, path_start=path_biased)
                                #n_pred['pred'] = Dictionary with prediction where pred[b][f] = StandardDataset
                                #n_pred['orig] = nk_train_splits
                                gc.collect()

                        else : # Compute predicted labels through post-processing
                            
                            #Predict classification probabilites (scores) for validation and test set
                            if computed :
                                try :
                                    with open(path+"_valid-fair-scores_all.pkl","rb") as file:
                                        n_valid_fair_scores = pickle.load(file)
                                except IOError as err:
                                    computed = False
                            if not computed :
                                n_valid_fair_scores = mt.prediction_nbias(nk_models, nk_train_splits, set='valid', pred_type='scores', blinding=blind, biased_test=False, path_start=path)
                            if computed :
                                try :
                                    with open(path+"_valid-biased-scores_all.pkl","rb") as file:
                                        n_valid_biased_scores = pickle.load(file)
                                except IOError as err:
                                    computed = False
                            if not computed :
                                n_valid_biased_scores = mt.prediction_nbias(nk_models, nk_train_splits, set='valid', pred_type='scores', blinding=blind, biased_test=True, path_start=path)
                            if computed :
                                try :
                                    with open(path+"_test-fair-scores_all.pkl","rb") as file:
                                        n_test_fair_scores = pickle.load(file)
                                except IOError as err:
                                    computed = False
                            if not computed :
                                n_test_fair_scores = mt.prediction_nbias(nk_models, nk_train_splits, set='test', pred_type='scores', blinding=blind, biased_test=False, path_start=path)
                            if computed :
                                try :
                                    with open(path+"_test-biased-scores_all.pkl","rb") as file:
                                        n_test_biased_scores = pickle.load(file)
                                except IOError as err:
                                    computed = False
                            if not computed :
                                n_test_biased_scores = mt.prediction_nbias(nk_models, nk_train_splits, set='test', pred_type='scores', blinding=blind, biased_test=True, path_start=path)
                            #n_pred['pred'] = Dictionary with prediction where pred[b][f] = StandardDataset
                            #n_pred['orig] = nk_train_splits

                            #Apply post-processing if needed
                            for postproc in postproc_methods :
                                print("### "+postproc+" ###")
                                if save :
                                    path = path_start+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_'+postproc
                                    path_biasedValidFairTest = path+"-BiasedValidFairTest"
                                    path_biasedValidBiasedTest = path+"-BiasedValidBiasedTest"
                                    path_fairValidFairTest = path+"-FairValidFairTest"
                                    #path_biasedPred = path +"-FairValidBiasedTest"
                                #n_valid_fair_scores, n_valid_biased_scores, n_test_fair_scores, n_test_biased_scores
                                                                
                                #print("Biased validation + fair test set")
                                if computed :
                                    try :
                                        with open(path_biasedValidFairTest+"_n.pkl","rb") as file:
                                            n_pred_transf_biasedValidFairTest = pickle.load(file)
                                    except IOError as err:
                                        computed = False
                                if not computed :
                                    #postproc:str, nk_train_split, n_valid_pred, n_test_pred, biased_valid:bool, path_start:str = None):
                                    #fair.n_postproc(postproc, nk_train_splits, n_valid_biased_scores, n_test_fair_scores, True, path_biasedValidFairTest)
                                    proc1 = mp.Process(target = fair.n_postproc, args = (postproc, n_valid_biased_scores, n_test_fair_scores, True, path_biasedValidFairTest))
                                    proc1.start()
                                    proc1.join()
                                print(computed)
                                #print("Biased Validation + biased test set")
                                if computed :
                                    try :
                                        with open(path_biasedValidBiasedTest+"_n.pkl","rb") as file:
                                            n_pred_transf_biasedValidBiasedTest = pickle.load(file)
                                    except IOError as err:
                                        computed = False
                                if not computed :
                                    #n_predBias_transf = fair.n_postproc(postproc, nk_train_splits, n_pred_bias, biased_truth = True, path_start=path_biasedPred) #Traditional configuration (biased evaluation of realistic (biased) postprocessing)
                                    proc3 = mp.Process(target = fair.n_postproc, args = (postproc, n_valid_biased_scores,n_test_biased_scores,True,path_biasedValidBiasedTest))
                                    proc3.start()
                                    proc3.join()
                                
                                #print("Fair validation set + fair test set")
                                if computed :
                                    try :
                                        with open(path_fairValidFairTest+"_n.pkl","rb") as file:
                                            n_pred_transf_FairValidFairTest = pickle.load(file)
                                    except IOError as err:
                                        computed = False
                                if not computed :
                                    #postproc:str, nk_train_split, n_valid_pred, n_test_pred, biased_valid:bool, path_start:str = None):
                                    proc2 = mp.Process(target = fair.n_postproc, args = (postproc, n_valid_fair_scores, n_test_fair_scores,False,path_fairValidFairTest))
                                    proc2.start()
                                    proc2.join()
                                """
                                #print("Fair validation set + biased test set")
                                #TODO adapt this to new format if used
                                if computed :
                                    try :
                                        with open(path_biasedPred_unbiasedTruth+"_n.pkl","rb") as file:
                                            n_predBias_transf_unbiasedTruth = pickle.load(file)
                                    except IOError as err:
                                        computed = False
                                if not computed :
                                    #n_predBias_transf_unbiasedTruth = fair.n_postproc(postproc, n_pred_bias, nbias_data_dict, biased_truth = False, path_start=path_biasedPred_unbiasedTruth) # Biased evaluation of "ideal" post-processing
                                    proc4 = mp.Process(target = fair.n_postproc, args = (postproc, n_pred_bias, nbias_data_dict, False, path_biasedPred_unbiasedTruth))
                                    proc4.start()
                                    proc4.join()
                                """
                                #Manage memory
                                #if computed :
                                #    del n_pred_transf, n_pred_transf_unbiasedTruth, n_predBias_transf #, n_predBias_transf_unbiasedTruth
                                gc.collect()
                                computed = True #end postproc
                            end = time.perf_counter()
                            print("Experiment completed for "+str(ds)+' '+str(bias)+' '+str(preproc)+' '+str(model)+ ' '+str(postproc)+' in hh:mm:ss',timedelta(seconds=end-stop))

                        computed = True #end blind
                    computed = True #end model
                #Manage memory
                del preproc_dict
                gc.collect()
                computed = True #end preproc
            del nbias_data_dict, nk_folds_dict, nk_train_splits
            gc.collect()
            computed = True #end bias
        computed = True #end ds

    end = time.perf_counter()
    print("Total experiment time : hh:mm:ss",timedelta(seconds=end-start))
