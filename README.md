# ControlledBias

This branch contains the code for submitted article "No evaluation without fair representation".

> The code in this directory is openly licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa]
> 
> ![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]
> 
> [![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]


[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg




## Run experiment code

To reproduce the experiments, run the file 'run_expe.py' with Python.

The OULAD datasets needs to be downloaded beforehand and the different CVS files placed in the directory 'dataset/OULAD'.

Requirements :
- Several Python module, including :
    - aif360
    - sklearn
    - ucimlrepo (to download the dataset Student)
    - pandas
    - numpy
- A minimum of 16GB RAM
- Around 74Gb of disk space to store the results

## Aditionnal results

The graphs for additional results are found in directory 'plots'.

The different bias types are indicated in all file names as follow :
- Label bias : 'label'
- Random selection : 'selectRandom'
- Self-selection : 'selectLow'
- Malicious selection : 'selectDouble'
- Malicious selection with unprivileged group excluded from the train set : ''


