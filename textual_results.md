# Textual presentation of experiment results

Evolution of models accuracy and fairness metrics with the increase of bias intensity in the training sets, as presented in Figure 1 of the paper "*Influence of Label and Selection Bias on Fairness Interventions*".

## Label bias

Results are presented as a list of values corresponding to the bias intensities $\beta_l$ = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

### Accuracy results
#### Models trained with dataset Student
Unmitigated models : [0.92, 0.87, 0.8, 0.78, 0.72, 0.67, 0.62, 0.6, 0.6, 0.6]  
Models mitigated with reweighting : [0.92, 0.86, 0.82, 0.76, 0.72, 0.69, 0.66, 0.63, 0.58, 0.54]  
Models mitigated with massaging : [0.93, 0.84, 0.8, 0.78, 0.73, 0.69, 0.66, 0.65, 0.64, 0.63]  
Models mitigated with Fairness Through Unawareness (FTU) : [0.92, 0.85, 0.81, 0.77, 0.74, 0.71, 0.69, 0.68, 0.66, 0.66]  

#### Models trained with dataset OULADstem
Unmitigated models : [0.94, 0.94, 0.93, 0.9, 0.73, 0.62, 0.59, 0.59, 0.59, 0.59]  
Models mitigated with reweighting : [0.94, 0.94, 0.93, 0.91, 0.76, 0.65, 0.6, 0.59, 0.57, 0.55]  
Models mitigated with massaging : [0.94, 0.93, 0.92, 0.85, 0.73, 0.68, 0.66, 0.64, 0.63, 0.61]  
Models mitigated with FTU : [0.94, 0.93, 0.93, 0.9, 0.74, 0.63, 0.58, 0.55, 0.53, 0.52]  

#### Models trained with dataset OULADsocial
Unmitigated models : [0.92, 0.92, 0.92, 0.92, 0.91, 0.91, 0.9, 0.89, 0.89, 0.89]  
Models mitigated with reweighting : [0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.91, 0.91]  
Models mitigated with massaging : [0.92, 0.92, 0.91, 0.91, 0.9, 0.89, 0.89, 0.89, 0.88, 0.88]  
Models mitigated with FTU :  [0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.91, 0.92, 0.91]  

### Statistical Parity Difference results
#### Models trained with dataset Student
Unmitigated models :  [-0.02, -0.14, -0.27, -0.32, -0.46, -0.57, -0.71, -0.76, -0.76, -0.76]  
Models mitigated with reweighting : [-0.01, -0.1, -0.14, -0.17, -0.2, -0.24, -0.28, -0.33, -0.43, -0.52]  
Models mitigated with massaging : [-0.01, -0.05, -0.08, -0.09, -0.1, -0.1, -0.12, -0.11, -0.11, -0.11]  
Models mitigated with FTU : [-0.02, -0.08, -0.14, -0.18, -0.17, -0.2, -0.2, -0.21, -0.21, -0.23]  

#### Models trained with dataset OULADstem
Unmitigated models : [-0.02, -0.03, -0.04, -0.09, -0.33, -0.44, -0.4, -0.42, -0.41, -0.4]  
Models mitigated with reweighting : [-0.02, -0.03, -0.03, -0.05, -0.23, -0.2, -0.21, -0.2, -0.18, -0.14]  
Models mitigated with massaging : [-0.02, -0.02, -0.02, -0.04, -0.08, -0.08, -0.07, -0.07, -0.05, -0.06]  
Models mitigated with FTU : [-0.02, -0.03, -0.03, -0.05, -0.1, -0.07, -0.05, -0.03, -0.01, -0.01]  

#### Models trained with dataset OULADsocial
Unmitigated models : [-0.01, -0.02, -0.03, -0.04, -0.09, -0.19, -0.32, -0.35, -0.39, -0.37]  
Models mitigated with reweighting : [-0.01, -0.02, -0.02, -0.02, -0.02, -0.03, -0.04, -0.06, -0.08, -0.1]  
Models mitigated with massaging : [-0.02, -0.02, -0.02, -0.02, -0.02, -0.04, -0.04, -0.04, -0.03, -0.03]  
Models mitigated with FTU : [-0.01, -0.02, -0.02, -0.02, -0.03, -0.02, -0.03, -0.03, -0.03, -0.02]  

### Equalized Odds Difference results
#### Models trained with dataset Student
Unmitigated models : [0.26, 0.13, 0.27, 0.33, 0.52, 0.65, 0.8, 0.88, 0.88, 0.87]  
Models mitigated with reweighting : [0.26, 0.13, 0.12, 0.16, 0.2, 0.25, 0.3, 0.36, 0.49, 0.6]  
Models mitigated with massaging : [0.34, 0.15, 0.09, 0.09, 0.11, 0.11, 0.12, 0.12, 0.1, 0.12]  
Models mitigated with FTU : [0.2, 0.14, 0.13, 0.18, 0.19, 0.2, 0.2, 0.21, 0.22, 0.24]  

#### Models trained with dataset OULADstem
Unmitigated models : [0.02, 0.03, 0.03, 0.12, 0.59, 0.82, 0.76, 0.8, 0.78, 0.77]  
Models mitigated with reweighting : [0.02, 0.03, 0.04, 0.07, 0.4, 0.37, 0.39, 0.38, 0.34, 0.28]  
Models mitigated with massaging : [0.03, 0.03, 0.04, 0.08, 0.14, 0.14, 0.13, 0.12, 0.1, 0.1]  
Models mitigated with FTU : [0.02, 0.03, 0.04, 0.07, 0.17, 0.12, 0.1, 0.05, 0.03, 0.01]  

#### Models trained with dataset OULADsocial
Unmitigated models : [0.05, 0.06, 0.06, 0.06, 0.12, 0.31, 0.54, 0.6, 0.69, 0.65]  
Models mitigated with reweighting : [0.06, 0.06, 0.05, 0.06, 0.05, 0.06, 0.07, 0.08, 0.12, 0.15]  
Models mitigated with massaging : [0.05, 0.05, 0.05, 0.06, 0.05, 0.06, 0.06, 0.06, 0.07, 0.06]  
Models mitigated with FTU : [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.06, 0.06]  

### Generalized Entropy Index results
#### Models trained with dataset Student
Unmitigated models : [0.03, 0.08, 0.13, 0.15, 0.2, 0.25, 0.31, 0.33, 0.34, 0.34]  
Models mitigated with reweighting : [0.03, 0.08, 0.11, 0.16, 0.2, 0.23, 0.26, 0.31, 0.38, 0.43]  
Models mitigated with massaging : [0.03, 0.09, 0.13, 0.15, 0.19, 0.22, 0.26, 0.27, 0.28, 0.3]  
Models mitigated with FTU : [0.04, 0.09, 0.12, 0.16, 0.18, 0.21, 0.23, 0.24, 0.26, 0.27]  

#### Models trained with dataset OULADstem
Unmitigated models : [0.03, 0.03, 0.03, 0.05, 0.19, 0.31, 0.35, 0.35, 0.35, 0.36]  
Models mitigated with reweighting : [0.03, 0.03, 0.03, 0.05, 0.16, 0.27, 0.33, 0.35, 0.37, 0.41]  
Models mitigated with massaging : [0.03, 0.03, 0.04, 0.08, 0.18, 0.23, 0.26, 0.28, 0.3, 0.32]  
Models mitigated with FTU : [0.03, 0.03, 0.03, 0.05, 0.17, 0.29, 0.37, 0.41, 0.44, 0.46]  

#### Models trained with dataset OULADsocial
Unmitigated models : [0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.06, 0.06]  
Models mitigated with reweighting : [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]  
Models mitigated with massaging : [0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06]  
Models mitigated with FTU : [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]  


## Selection bias

Results are presented as a list of values corresponding to the bias intensities $p_u$ = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

### Accuracy results
#### Models trained with dataset Student
Unmitigated models : [0.92, 0.92, 0.92, 0.91, 0.91, 0.9, 0.9, 0.89, 0.87, 0.88]  
Models mitigated with reweighting : [0.92, 0.92, 0.92, 0.91, 0.91, 0.91, 0.9, 0.9, 0.87, 0.86]  
Models mitigated with massaging : [0.92, 0.92, 0.92, 0.91, 0.91, 0.91, 0.89, 0.88, 0.86, 0.86]  
Models mitigated with Fairness Through Unawareness (FTU) : [0.93, 0.92, 0.92, 0.93, 0.91, 0.91, 0.91, 0.9, 0.89, 0.88]  

#### Models trained with dataset OULADstem
Unmitigated models : [0.94, 0.94, 0.94, 0.93, 0.93, 0.93, 0.93, 0.92, 0.9, 0.84]  
Models mitigated with reweighting : [0.94, 0.94, 0.93, 0.94, 0.93, 0.93, 0.93, 0.92, 0.92, 0.9]  
Models mitigated with massaging : [0.94, 0.93, 0.93, 0.93, 0.93, 0.93, 0.92, 0.91, 0.9, 0.87]  
Models mitigated with FTU : [0.94, 0.94, 0.94, 0.93, 0.93, 0.93, 0.93, 0.92, 0.91, 0.89]  

#### Models trained with dataset OULADsocial
Unmitigated models : [0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.91, 0.91, 0.91, 0.89]  
Models mitigated with reweighting : [0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.91, 0.91, 0.91, 0.9]  
Models mitigated with massaging : [0.92, 0.92, 0.91, 0.91, 0.91, 0.9, 0.89, 0.88, 0.85, 0.83]  
Models mitigated with FTU : [0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.91, 0.91, 0.91, 0.9]  

### Statistical Parity Difference results
#### Models trained with dataset Student
Unmitigated models : [-0.0, -0.01, -0.01, -0.03, -0.03, -0.05, -0.07, -0.09, -0.12, -0.12]  
Models mitigated with reweighting : [0.01, -0.01, -0.0, 0.01, -0.02, -0.02, -0.01, -0.01, -0.03, -0.02]  
Models mitigated with massaging : [-0.0, -0.02, 0.01, 0.01, -0.0, -0.01, 0.0, -0.0, 0.0, 0.01]  
Models mitigated with FTU : [-0.01, -0.01, -0.0, -0.01, -0.0, -0.03, -0.01, -0.02, -0.03, -0.06]  

#### Models trained with dataset OULADstem
Unmitigated models : [-0.02, -0.02, -0.03, -0.03, -0.04, -0.04, -0.05, -0.07, -0.1, -0.2]  
Models mitigated with reweighting : [-0.02, -0.02, -0.02, -0.03, -0.03, -0.03, -0.04, -0.04, -0.04, -0.05]  
Models mitigated with massaging : [-0.02, -0.02, -0.01, -0.0, -0.0, 0.02, 0.06, 0.13, 0.23, 0.39]  
Models mitigated with FTU : [-0.02, -0.02, -0.03, -0.03, -0.03, -0.03, -0.04, -0.04, -0.05, -0.06]  

#### Models trained with dataset OULADsocial
Unmitigated models : [-0.01, -0.0, -0.0, -0.01, -0.01, -0.02, -0.04, -0.03, -0.05, -0.07]  
Models mitigated with reweighting : [-0.01, -0.0, -0.0, -0.01, -0.01, -0.0, -0.01, -0.01, -0.01, -0.01]  
Models mitigated with massaging : [-0.01, -0.01, -0.01, 0.01, 0.03, 0.06, 0.08, 0.11, 0.22, 0.23]  
Models mitigated with FTU : [-0.01, -0.01, -0.01, -0.01, -0.0, -0.01, -0.0, -0.0, -0.01, -0.01]  

### Equalized Odds Difference results
#### Models trained with dataset Student
Unmitigated models : [0.24, 0.21, 0.23, 0.14, 0.2, 0.23, 0.22, 0.3, 0.48, 0.52]  
Models mitigated with reweighting : [0.25, 0.22, 0.19, 0.29, 0.2, 0.1, 0.11, 0.17, 0.15, 0.12]  
Models mitigated with massaging : [0.26, 0.21, 0.22, 0.29, 0.19, 0.21, 0.21, 0.11, 0.15, 0.11]  
Models mitigated with FTU : [0.23, 0.21, 0.23, 0.21, 0.18, 0.22, 0.23, 0.21, 0.2, 0.18]  

#### Models trained with dataset OULADstem
Unmitigated models : [0.03, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.07, 0.12, 0.31]  
Models mitigated with reweighting : [0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.04, 0.04, 0.04, 0.07]  
Models mitigated with massaging : [0.03, 0.02, 0.03, 0.03, 0.04, 0.07, 0.11, 0.23, 0.41, 0.7]  
Models mitigated with FTU : [0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.05, 0.09]  

#### Models trained with dataset OULADsocial
Unmitigated models : [0.04, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.05, 0.07]  
Models mitigated with reweighting : [0.05, 0.04, 0.05, 0.05, 0.05, 0.04, 0.05, 0.04, 0.04, 0.04]  
Models mitigated with massaging : [0.05, 0.05, 0.05, 0.07, 0.09, 0.15, 0.18, 0.23, 0.42, 0.43]  
Models mitigated with FTU : [0.06, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]  

### Generalized Entropy Index results
#### Models trained with dataset Student
Unmitigated models : [0.03, 0.03, 0.03, 0.04, 0.03, 0.04, 0.04, 0.04, 0.05, 0.04]  
Models mitigated with reweighting : [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.04]  
Models mitigated with massaging : [0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05]  
Models mitigated with FTU : [0.03, 0.03, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04, 0.04, 0.04]  

#### Models trained with dataset OULADstem
Unmitigated models : [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.05, 0.09]  
Models mitigated with reweighting : [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.06]  
Models mitigated with massaging : [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.05, 0.07]  
Models mitigated with FTU : [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.06]  

#### Models trained with dataset OULADsocial
Unmitigated models : [0.04, 0.04, 0.04, 0.04, 0.04, 0.03, 0.04, 0.04, 0.04, 0.04]  
Models mitigated with reweighting : [0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.04, 0.04, 0.04, 0.04]  
Models mitigated with massaging : [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06]  
Models mitigated with FTU : [0.04, 0.04, 0.04, 0.04, 0.04, 0.03, 0.04, 0.04, 0.04, 0.04]  