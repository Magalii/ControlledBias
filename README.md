# ControlledBias

This branch contains the code used for the experiments presented in the article "*Influence of Label and Selection Bias on Fairness Interventions*" by Magali Legast, Toon Calder and François Fouss that was accepted at [EWAF2025](https://2025.ewaf.org/home).
The article can be accessed in [EWAF25 proceedings](https://proceedings.mlr.press/v294/) or directly downloaded by clicking [here](https://raw.githubusercontent.com/mlresearch/v294/main/assets/legast25a/legast25a.pdf).


> The code in this directory is openly licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa]
> 
> ![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]
> 
> [![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]


[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg




## Run experiment code

To reproduce the experiments, run the file 'experiment_ewaf2025.py' with Python.

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

## Textual results

The results presented in the paper graphs can also be found in textual form in the file 'textual_results.md'.