"""
    Method that controls the general experiment pipeline.
    Is called by the methods in run_expe.py
"""

from datetime import timedelta
import multiprocessing as mp
import pickle
import time
import gc
import sys 
sys.path.append('../')

from dataset.studentMale_dataset import StudentMaleDataset
from dataset.oulad_dataset import OULADDataset
import dataset_biasing as db
import model_training as mt
import fairness_intervention as fair

def pipeline(datasets:list[str], biases:list[str], bias_levels:list[float], preproc_methods:list[str], postproc_methods:list[str], classifiers:list[str], blind_model:list[bool], path_start:str, save_intermediate:str, verbose:bool):
    """
        Experiment pipeline with dataset biasing, model training with pre-processing, post-processing or no mitigation, and predictions
        Results, potentially including intermediate results, are saved on disk.
        Only (intermediate) results that are not already presents on disk will be recompute, unless you change the variable 'computed' in the code.
        datasets : list[String]
            Names of datasets experiment should be performed with, strings must be in ['OULADsocial','OULADstem','studentBalanced', 'Student, 'OULADstemHarder','OULADsocialHarder']
        biases : list[String]
            Names of biases experiment should be performed with, strings must be in ['label','labelDouble','selectLow','selectDoubleProp','selectRandom','selectPrivNoUnpriv','selectRandomWhole']
        bias_levels : list[float]
            List of values for parameter controlling bias intensity (beta_l or p_u) experiment should be performed with, values must be in [0,1]
        preproc_methods : list[String]
            Names of preprocessing methods experiment should be performed with, strings must be in ['','massaging', 'reweighting', 'LFR']
            [''] for no preprocessing
            Fairness Through Unawareness corresponds to no preprocessing (preproc_methods = ['']) and blinding (blind_model == [True])
        postproc_methods : list[String]
            Names of postprocessing methods experiment should be performed with, strings must be in ['eqOddsProc','calEqOddsProc','ROC-SPD','ROC-EOD','ROC-AOD']
            [] for no post-processing
            If postprocessing is applied, there must be no preprocessing (preproc_methods == [''])
        classifiers : list[String]
            Names of training algorithms experiment should be performed with, strings must be in ['tree','RF','neural']
        blind_model : list[bool]
            Blinding modes that experiment should be performed with, True to exclude sensitive attribute in training, False to include it
            Bias mitigation other than Fairness Through Unawareness are not compatible with blinding
        path_start : String
            Path at wich (intermediate) results are saved
        save_intermediate : String
            Controle how much intermediate results are saved on disk
            'no' to not save any intermediate result (NOT RECOMMENDED, as it will lead to the same results recomputed many times)
            'dataset_only' to only save the original datasets (without their bias mitigated versions)
            'minimal' to save the original and biased version of the datasets
            'intermediate' to save all datasets verisons and their splits. Will guarantee that the same splits are used accross all experiments
            'all' to save all intermadiate results (WARNING may require a lot of disk space)
            Results that are present on disk but are not included in the level selected will be recomputed

    No return value
    """
    start = time.perf_counter()

    if save_intermediate not in ['all','intermediate','minimal','no'] :
        print("WARNING save must a value in ['all','intermediate','minimal','no']")

    computed = True #Set this value to False here AND later in the code to force recomputation, even if data has already been saved on disk.

    for ds in datasets :
        step = ds
        if verbose : print("\n#### "+step)
        if save_intermediate in ['all','intermediate','minimal'] :
            path_dataset = path_start+ds+"_dataset"
        else : path_dataset = None
        if computed and path_dataset is not None :
            try :
                with open(path_start+ds+"_dataset",'rb') as file:
                    dataset_orig = pickle.load(file)
            except (Exception, pickle.UnpicklingError) as err:
                computed = False
        if not computed or path_dataset is None:
            if verbose : print("(re)compute "+step)
            if ds == 'student' :
                dataset_orig = StudentMaleDataset(balanced=False)
            elif ds == 'studentBalanced' :
                dataset_orig = StudentMaleDataset(balanced=True)
            elif ds == 'OULADsocial' :
                dataset_orig = OULADDataset(domain='social')
            elif ds == 'OULADstem' :
                dataset_orig = OULADDataset(domain='stem')
            elif ds == 'OULADsocialHarder' :
                dataset_orig = OULADDataset(domain='social', hard_problem=True)
            elif ds == 'OULADstemHarder' :
                dataset_orig = OULADDataset(domain='stem', hard_problem=True)
            elif ds == 'studentHarder' :
                dataset_orig = StudentMaleDataset(balanced=False,features_to_drop=['G1','G2'])
            else :
                print("WARNING '"+str(ds)+"' is not a valid dataset name")
            if path_dataset is not None :
                with open(path_start+ds+"_dataset",'wb') as file:
                    pickle.dump(dataset_orig,file)
       
        for bias in biases :
            step = step + " " + bias
            if verbose : print("- "+step)
            label_bias = ['label','labelDouble'] #List of bias falling under the "label bias" umbrella
            select_bias = ['selectLow','selectDoubleProp','selectRandom','selectPrivNoUnpriv','selectRandomWhole'] #List of bias falling under the "selection bias" umbrella   
            if save_intermediate in ['all','intermediate'] :
                path_bias = path_start+ds+'_'+bias
            else :
                path_bias = None
            if computed and path_bias is not None :
                try : #retrieve biased datasets
                    with open(path_bias+"_nBiasDatasets.pkl","rb") as file:
                        nbias_data_dict = pickle.load(file)
                except (Exception, pickle.UnpicklingError) as err:
                    computed = False
            if not computed or path_bias is None :
                if verbose : print("(re)compute "+step)
                if bias in label_bias :
                    if ds[0:7] == 'student' : noise = 0.1
                    elif ds[0:5] == 'OULAD' : noise = 0.2 #ds == 'OULADsocial' or ds == 'OULADstem'
                    else : print("WARNING Not a valid dataset name")
                    if bias == 'label' : double_disc = False
                    elif bias == 'labelDouble' : double_disc = True
                    else : print("WARNING Error with type of label bias")
                    nbias_data_dict = db.mislabeling_nbias(dataset_orig, bias_levels, noise=noise, double_disc=double_disc,path_start=path_bias)
                    if verbose : print("Label bias introduced")
                elif bias in select_bias:
                    if bias == 'selectDoubleProp' :
                        removal_distr = 'double_disc'
                    elif bias == 'selectLow' :
                        removal_distr = 'lower_weight'
                    elif bias == 'selectRandom' :
                        removal_distr = 'random'
                    elif bias == 'selectPrivNoUnpriv':
                        removal_distr = 'priv_neg_no_unpriv'
                    elif bias == 'selectRandomWhole' :
                        removal_distr = 'double_random'
                    else : print("WARNING Error with type of selection bias")
                    nbias_data_dict = db.undersampling_incremental_nbias(dataset_orig, removal_distr, bias_levels=bias_levels,path_start=path_bias) #bias_levels
                    #undersampling_incremental_nbias(dataset_orig: StandardDataset, removal_distr: str, incr_bias:float=0.1, bias_levels:list=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], path_start: str = None) :
    
                    if verbose : print("Selection bias introduced : "+str(bias))
                else :
                    print("WARNING Not a valid bias type")
            
            #create train and test sets for each fold and bias, nk_train_splits[bias][fold]: {'train': train set, 'test': test set}
            if save_intermediate not in ['all','intermediate'] :
                path_bias = None
            if computed and path_bias is not None :
                try : #retrieve biased dataset split into k partitions
                    with open(path_bias+"_nbias_splits_datasets.pkl","rb") as file:
                        nk_folds_dict = pickle.load(file)
                except (Exception, pickle.UnpicklingError) as err: #create n biased datasets and split them into k partitions
                    computed = False
            if not computed or path_bias is None :
                if verbose : print("(re)compute "+step +" splitted in folds")
                if bias in label_bias : #same split for all folds
                    k = 10
                elif bias in select_bias :
                    k = 5
                nk_folds_dict, _ = mt.common_splits(nbias_data_dict,k,path_bias)
                #nk_folds_dict[bias]: list of test folds for each bias level
            if computed and path_bias is not None :
                try :#retrieve dict of train-test splits for all bias levels
                    with open(path_bias+"_train-test-nk.pkl","rb") as file:
                        nk_train_splits = pickle.load(file)
                except (Exception, pickle.UnpicklingError) as err: #create n biased datasets and split them into k partitions
                    computed = False
            if not computed or path_bias is None :
                if verbose : print("(re)compute "+step+" form train, valid and test split")
                #use path_split because these results are needed for analysis and thus must be saved
                nk_train_splits = mt.nk_merge_train(nk_folds_dict,valid=True,path_start=path_bias)
                #nk_train_splits[bias][fold]: {'train': train set, 'valid': validation set, 'test': test set}

            #apply preprocessing if needed
            for preproc in preproc_methods :
                step = step + " " + preproc
                if verbose : print("- "+step)
                if save_intermediate in ['all'] :
                    path_preproc = path = path_start+ds+'_'+bias+'_'+preproc
                else :
                    path_preproc = None
                if preproc == '' :
                    preproc_dict = nk_train_splits
                    if verbose : print("No preproc was applied")
                else :
                    if computed and path_preproc is not None :
                        try :
                            with open(path_preproc+'_nkdatasets.pkl',"rb") as file:
                                preproc_dict = pickle.load(file)
                        except (Exception, pickle.UnpicklingError) as err:
                            computed = False
                    if not computed or path_preproc is None :
                        if verbose : print("(re)compute "+step)
                        preproc_dict = fair.apply_preproc(nk_train_splits, preproc, path_start=path_preproc)
                stop = time.perf_counter()
                print("All preproc done for "+str(ds)+' '+str(bias)+' '+str(preproc)+' at ',timedelta(seconds=stop-start))
                
                # Train models
                for model in classifiers :
                    step = step + " " + model
                    if verbose : print("- "+step)
                    for blind in blind_model :
                        #Train models with (preprocessed) data
                        if blind : visibility = 'Blinded'
                        else : visibility = 'Aware'
                        step = step + " " + visibility
                        path_model = path_start+ds+'_'+bias+'_'+preproc+'_'+model+visibility
                        
                        if save_intermediate in ['all','intermediate','minimal'] :
                            path_trained = path_model
                        else :
                            path_trained = None
                        if computed and path_trained is not None :
                            try :
                                with open(path_trained+"_all.pkl","rb") as file:
                                    nk_models = pickle.load(file)
                            except (Exception, pickle.UnpicklingError) as err:
                                computed = False
                        if not computed or path_trained is None :
                            if verbose : print("(re)compute "+step)
                            nk_models = mt.classifier_nbias(model, preproc_dict, blinding=blind, path_start=path_trained)

                        #Make predictions
                        if len(postproc_methods) == 0 : #Compute predicted labels directly 
                            #Create predictions from data (both original and biased test set, no preprocessing on test set)
                            #End result, must be saved
                            path_pred = path_model
                            path_pred_biased = path_model +"_biasedTest"
                            if computed and path_pred is not None :
                                try :
                                    with open(path_pred+"_pred_all.pkl","rb") as file:
                                        n_pred = pickle.load(file)
                                    with open(path_pred_biased+"_pred_all.pkl","rb") as file:
                                        n_pred_bias = pickle.load(file)
                                except (Exception, pickle.UnpicklingError) as err:
                                    computed = False
                            if not computed or path_pred is None :
                                if verbose : print("(re)compute "+step+" fair and biased predictions")
                                n_pred = mt.prediction_nbias(nk_models, nk_train_splits, set='test', blinding=blind, path_start=path_pred) #pred_type='labels'
                                n_pred_bias = mt.prediction_nbias(nk_models, nk_train_splits, set='test', blinding=blind, biased_test=True, path_start=path_pred_biased)
                                #n_pred['pred'] = Dictionary with prediction where pred[b][f] = StandardDataset
                                #n_pred['orig] = nk_train_splits
                                gc.collect()

                        else : # Post-process models before making predictions
                            
                            #Predict classification probabilites (scores) for validation and test set
                            if save_intermediate in ['all'] :
                                #path_vfs = path_model + "_validFairScores" # Not used in expe
                                path_vbs = path_model + "_validBiasedScores"
                                path_tfs = path_model + "_testFairScores"
                                path_tbs = path_model + "_testBiasedScores"
                            else :
                                path_vbs, path_tfs, path_tbs = None, None, None
                                #path_vfs = path_model + "_validFairScores"
                            """ Not used in expe
                            if computed and path_vfs is not None :
                                try :
                                    with open(path_vfs+"_pred_all.pkl","rb") as file:
                                        n_valid_fair_scores = pickle.load(file)
                                except (Exception, pickle.UnpicklingError) as err:
                                    computed = False
                            if not computed or path_vfs is None :
                                if verbose : print("(re)compute "+step+ " validFairScores")
                                n_valid_fair_scores = mt.prediction_nbias(nk_models, nk_train_splits, set='valid', blinding=blind, biased_test=False, path_start=path_vfs)
                            """
                            #path_vbs = path_model + "_validBiasedScores"
                            if computed and path_vbs is not None :
                                try :
                                    with open(path_vbs+"_pred_all.pkl","rb") as file:
                                        n_valid_biased_scores = pickle.load(file)
                                except (Exception, pickle.UnpicklingError) as err:
                                    computed = False
                            if not computed or path_vbs is None :
                                if verbose : print("(re)compute "+step+ " validBiasedScores")
                                n_valid_biased_scores = mt.prediction_nbias(nk_models, nk_train_splits, set='valid', blinding=blind, biased_test=True, path_start=path_vbs)
                            #path_tfs = path_model + "_testFairScores"
                            if computed and path_tfs is not None :
                                try :
                                    with open(path_tfs+"_pred_all.pkl","rb") as file:
                                        n_test_fair_scores = pickle.load(file)
                                except (Exception, pickle.UnpicklingError) as err:
                                    computed = False
                            if not computed or path_tfs is None :
                                if verbose : print("(re)compute "+step+ " testFairScores")
                                n_test_fair_scores = mt.prediction_nbias(nk_models, nk_train_splits, set='test', blinding=blind, biased_test=False, path_start=path_tfs)
                            #path_tbs = path_model + "_testBiasedScores"
                            if computed and path_tbs is not None :
                                try :
                                    with open(path_tbs+"_pred_all.pkl","rb") as file:
                                        n_test_biased_scores = pickle.load(file)
                                except (Exception, pickle.UnpicklingError) as err:
                                    computed = False
                            if not computed or path_tbs is None :
                                if verbose : print("(re)compute "+step+ " testBiasedScores")
                                n_test_biased_scores = mt.prediction_nbias(nk_models, nk_train_splits, set='test', blinding=blind, biased_test=True, path_start=path_tbs)
                            #n_pred['pred'] = Dictionary with prediction where pred[b][f] = StandardDataset
                            #n_pred['orig] = nk_train_splits

                            #Apply post-processing if needed
                            for postproc in postproc_methods :
                                step = step + " " + model + postproc
                                if verbose : print("### "+step+" ###")
                                #End result, must be saved
                                path_postproc = path_start+ds+'_'+bias+'_'+preproc+'_'+model+visibility+'_'+postproc
                                path_biasedValidFairTest = path_postproc+"-BiasedValidFairTest"
                                path_biasedValidBiasedTest = path_postproc+"-BiasedValidBiasedTest"
                                #path_fairValidFairTest = path_postproc+"-FairValidFairTest" #Not used in expe
                                #path_biasedPred = path_postproc +"-FairValidBiasedTest"
                                #n_valid_fair_scores, n_valid_biased_scores, n_test_fair_scores, n_test_biased_scores
                                                                
                                if computed and path_biasedValidFairTest is not None :
                                    try :
                                        with open(path_biasedValidFairTest+"_n.pkl","rb") as file:
                                            n_pred_transf_biasedValidFairTest = pickle.load(file)
                                    except (Exception, pickle.UnpicklingError) as err:
                                        computed = False
                                if not computed or path_biasedValidFairTest is None :
                                    if verbose : print("(re)compute "+step+ " Fair evaluation")
                                    #postproc:str, nk_train_split, n_valid_pred, n_test_pred, biased_valid:bool, path_start:str = None):
                                    #fair.n_postproc(postproc, nk_train_splits, n_valid_biased_scores, n_test_fair_scores, True, path_biasedValidFairTest)
                                    proc1 = mp.Process(target = fair.n_postproc, args = (postproc, n_valid_biased_scores, n_test_fair_scores, True, path_biasedValidFairTest))
                                    proc1.start()
                                    proc1.join()
                                #print("Biased Validation + biased test set")
                                
                                if computed and path_biasedValidBiasedTest is not None :
                                    try :
                                        with open(path_biasedValidBiasedTest+"_n.pkl","rb") as file:
                                            n_pred_transf_biasedValidBiasedTest = pickle.load(file)
                                    except (Exception, pickle.UnpicklingError) as err:
                                        computed = False
                                if not computed or path_biasedValidBiasedTest is None :
                                    if verbose : print("(re)compute "+step+ " Biased evaluation")
                                    #n_predBias_transf = fair.n_postproc(postproc, nk_train_splits, n_pred_bias, biased_truth = True, path_start=path_biasedPred) #Traditional configuration (biased evaluation of realistic (biased) postprocessing)
                                    proc3 = mp.Process(target = fair.n_postproc, args = (postproc, n_valid_biased_scores,n_test_biased_scores,True,path_biasedValidBiasedTest))
                                    proc3.start()
                                    proc3.join()
                                """ #Not used in experiment
                                #if verbose : print("Fair validation set + fair test set")
                                if computed and path_fairValidFairTest is not None :
                                    try :
                                        with open(path_fairValidFairTest+"_n.pkl","rb") as file:
                                            n_pred_transf_FairValidFairTest = pickle.load(file)
                                    except (Exception, pickle.UnpicklingError) as err:
                                        computed = False
                                if not computed or path_fairValidFairTest is None :
                                    if verbose : print("(re)computed after "+path_fairValidFairTest+"_n.pkl")
                                    #postproc:str, nk_train_split, n_valid_pred, n_test_pred, biased_valid:bool, path_start:str = None):
                                    proc2 = mp.Process(target = fair.n_postproc, args = (postproc, n_valid_fair_scores, n_test_fair_scores,False,path_fairValidFairTest))
                                    proc2.start()
                                    proc2.join()
                                """
                                #Manage memory
                                #if computed :
                                #    del n_pred_transf, n_pred_transf_unbiasedTruth, n_predBias_transf #, n_predBias_transf_unbiasedTruth
                                gc.collect()
                                computed = True #end postproc
                                step =''
                            end = time.perf_counter()
                            print("Experiment completed for "+str(ds)+' '+str(bias)+' '+str(preproc)+' '+str(model)+ ' '+str(postproc)+' in ',timedelta(seconds=end-stop))

                        computed = True #end blind
                        step = ''
                    computed = True #end model
                    step = ''
                #Manage memory
                del preproc_dict
                gc.collect()
                computed = True #end preproc
                step = ''
            del nbias_data_dict, nk_folds_dict, nk_train_splits
            gc.collect()
            computed = True #end bias
            step = ''
        computed = True #end ds
        step = ''

    end = time.perf_counter()
    print("Experiment time : ",timedelta(seconds=end-start))