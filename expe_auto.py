from datetime import timedelta
import multiprocessing as mp
from pandas import DataFrame
import pickle
import time
import gc
import sys 
sys.path.append('..')

import expe_pipeline as expe
import analyzing as a
import plotting as plot

start = time.perf_counter()
stop = start
bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #, 1]
k = -1 #negative value will create error instead of silent mistake
path_start = "Code/ControlledBiasPrivate/data/"
blind = False


datasets = ['student','OULADsocial','OULADstem']
biases = ['label','selectDoubleProp','selectLow'] #,'selectDoubleProp', #'label', selectDoubleProp 'labelDouble', 'selectDouble','selectLow'] #['label','selectDouble','selectLow'] #,'selectDouble','selectLow' label
preproc_methods = ['','massaging', 'reweighting'] #, 'reweighting', 'LFR',
#ftu corresponds to no preproc + blind model
postproc_methods = [] #'eqOddsProc','calEqOddsProc','ROC-SPD','ROC-EOD'] #['eqOddsProc','calEqOddsProc','ROC-SPD','ROC-EOD','ROC-AOD'] #] #eqOddsProc #Empty list for no post-proc 
classifiers = ['neural'] #['tree','RF','neural']
blind_model = [False] #,True]

save = True
display_plot = False


print("\n ### Pre-proc : Start experiment ###\n")
expe.run_expe(datasets,biases,bias_levels,preproc_methods,postproc_methods,classifiers,blind_model,path_start=path_start)
gc.collect()
print("\n ### Pre-proc : Start computing results ###\n")
a.compute_all(datasets,biases,preproc_methods,postproc_methods,classifiers,blind_model,path_start=path_start)
gc.collect()
#print("\n ### Pre-proc : Start producing plots ###\n")
#plot.plot_all(datasets,biases,preproc_methods,postproc_methods,classifiers,blind_model,bias_levels,path_start=path_start)

print("\n ### FTU : Start experiment ###\n")
preproc_methods = ['']
blind_model = [True]
expe.run_expe(datasets,biases,bias_levels,preproc_methods,postproc_methods,classifiers,blind_model,path_start=path_start)
gc.collect()
exit()
print("\n ### FTU : Start computing results ###\n")
a.compute_all(datasets,biases,preproc_methods,postproc_methods,classifiers,blind_model,path_start=path_start)
gc.collect()

print("\n ### Post-proc : Start experiment ###\n")
preproc_methods = ['']
postproc_methods = ['eqOddsProc','calEqOddsProc','ROC-SPD','ROC-EOD'] #['eqOddsProc','calEqOddsProc','ROC-SPD','ROC-EOD','ROC-AOD']  #Empty list for no post-proc 
blind_model = [False] #,True] #True
expe.run_expe(datasets,biases,bias_levels,preproc_methods,postproc_methods,classifiers,blind_model,path_start=path_start)
gc.collect()
print("\n ### Post-proc : Start computing results ###\n")
a.compute_all(datasets,biases,preproc_methods,postproc_methods,classifiers,blind_model,path_start=path_start)
gc.collect()
#print("\n ### Post-proc : Start producing plots ###\n")
#plot.plot_all(datasets,biases,preproc_methods,postproc_methods,classifiers,blind_model,bias_levels,path_start=path_start)
