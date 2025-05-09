# ControlledBias

> The code in this directory is openly licensed via [CC-BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This branch contains the code used for the experiments presented in the article "*Influence of Label and Selection Bias on Fairness Interventions*" by Magali Legast, Toon Calder and François Fouss that will be presented at [EWAF2025](https://2025.ewaf.org/home).
The submitted version of the paper is available in this repository.

## Run experiment code

To reproduce the experiments, run the file 'experiment_ewaf2025.py' with Python from the 'ControlledBias' directory.

The OULAD datasets needs to be downloaded beforehand and the different CVS files placed in the directory 'dataset/OULAD'.

Running this code requires several Python module, including :
- aif360
- sklearn
- ucimlrepo (to download the dataset Student)
- pandas
- numpy


### Troubleshooting

- Windows environement :\
    This code was written for a Linux environment. All the paths to files needs to be changed for it to be run in a Windows environment.
- RAM needed exceeds computer capacities :\
   This is due to Python memory management. To counter this, you can run the experiment script for one dataset at a time by changing the datasets presents in the 'datasets' list. If it isn't enough, you can also try running it for one dataset and preprocessing method at a time.\
   Know that the script method that uses the most RAM is 'a.compute_all()'.
- ControlledBias module not found :\
    To run the script as is, you need to place all the files and directories of this repository in a directory named 'ControlledBias'. If you do not want to do that, you need to adjust all the 'import' statements involving the 'ControlledBias' module.