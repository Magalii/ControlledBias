import pandas as pd
import numpy as np
import scipy.stats as st
import pickle

from aif360.datasets import StandardDataset

###############################
# Measurement related biasing #
###############################

def measurement_biasing(df: pd.DataFrame, attr: str, sens_attr: str, b_m: float, orig_values: pd.Series=None, noise=0.1, double_disc=False, fav_one=True, inplace=True) :
    """ Introduce measurement bias to chosen attribute by adding penalty to unprivileged group (or both unprivileged and favored groups if double_disc = True)

    Parameters
    ----------
    df : Pandas DataFrame
        Dataset to which the bias need to be added
    attr : string
        Name of the attribute to be biased, must be a column of df if no orig_values is provided
    sens_attr : string
        Name of the sensitive attribute
    b_m : float
        Coefficient of the measurement bias added, 0 means no bias, 1 means penalty on minority is half of max value for the biased attribute
    orig_values : Pandas Series, optional
        Series of the values the bias needs to be added to, must correspond to a DataFrame column with the same indices than df
    noise : float, optional
        Scale of the standart deviation, must be positive, should be between 0 (no noise) and 1 (standard deviation is half of the difference between lowest and highest value attr), will be scaled to the values attribute attr
    double_disc : Boolean, optional
        Whether the label bias is applied only on unprivileged group (False) or on both unprivileged as a penalty and privileged group as a bonus (True)
    fav_one : Boolean, optional
        Wether the favorable value for the sensitive attribute is 1 (True) or 0 (False)
    inplace : Boolean, optional
        Wether the biased values should replace the original ones (True) or a new column with the biased values should be created (False)

    Returns
    -------
    Pandas DataFrame
        New DataFrame with the biased dataset

    """
    #TODO raise error if needed, like id len(df) != len(orig_values.values)

    df_biased = df.copy()

    if orig_values is None :
        values = df_biased[attr].values #values is now a np array
    else :
        values = orig_values.values #values is now a np array
    #Noise
    if b_m == 0 :
        Nr = np.zeros(len(df_biased.index))
    else :
        scale = (np.max(values) - np.min(values))/2 #df_biased[name].max() - df_biased[name].min()
        sigma_R = noise*scale # Standart deviation value must be proportional to the values of the attributes
        rng = np.random.default_rng(seed = 4224)
        Nr = rng.normal(0, sigma_R, len(df_biased.index))
        b_m = b_m*scale
        
    if inplace :
        attr_biased = attr
    else :
        attr_biased = attr + "_biased"
    if double_disc : # -b_m*0 + b_m*1 when df[sens_attr] == fav_one and -b_m*1 + b_m*0 when df[sens_attr] != fav_one
        df_biased[attr_biased] = values - b_m*abs(fav_one - df[sens_attr]) + b_m*abs(1-fav_one-df[sens_attr]) + Nr
    else : #Only unprivileged will be affected, everybody will get noise
        df_biased[attr_biased] = values - b_m*abs(df[sens_attr]-fav_one) + Nr

    return df_biased

def flip_label_bias(df: pd.DataFrame, attr: str, sens_attr: str, b_m: float, double_disc : True, fav_one=True, inplace=True) :
    """ WARNING This function has not been tested !!

    Add measurement bias to chosen binary attribute by flipping the label

    Parameters
    ----------
    df : Pandas DataFrame
        Dataset to which the bias need to be added
    attr : string
        Name of the attribute to be biased, must be a binary attribute
    sens_attr : string
        Name of the sensitive attribute
    b_m : float
        Coefficient of the measurement bias added, 0 means no bias, 1 means 50% of dataset is affected by the bias
    double_disc : Boolean, optional
        Whether the label bias is applied only on unprivileged group (False) or on both unprivileged as a penalty and privileged group as a bonus (True)
    fav_one : Boolean, optional
        Wether the favorable value for the sensitive attribute is 1 (True) or 0 (False)
    inplace : Boolean, optional
        Wether the biased values should replace the original ones (True) or a new column with the biased values should be created (False)

    Returns
    -------
    Pandas DataFrame
        New DataFrame with the biased dataset
    """
    unpriv = 1-fav_one
    pos_val = 1
    neg_val = 0
    if inplace :
        attr_biased = attr
    else :
        attr_biased = attr + "_biased"
    
    df_biased = df.copy()

    if double_disc :
        coef = 0.5
        df_priv_neg = df_biased.loc[(df[sens_attr] == fav_one) & (df[attr] == neg_val)]
        num_priv_mislab = coef*b_m*df_priv_neg.sum()
        flip_priv_id = my_random_choice(df_priv_neg.index.to_list(),num_priv_mislab)
        df_biased.loc[flip_priv_id,attr_biased] = pos_val
    else :
        coef = 1
        df_priv_neg = None

    df_unpriv_pos = df_biased.loc[(df[sens_attr] == unpriv) & (df[attr] == pos_val)]
    num_unpriv_mislab = coef*b_m*df_unpriv_pos.sum()
    flip_unpriv_id = my_random_choice(df_unpriv_pos.index.to_list(),num_unpriv_mislab)
    df_biased.loc[flip_unpriv_id,attr_biased] = neg_val
        
    return df_biased

def mislabeling_nbias(dataset_orig: StandardDataset, b_m: list, noise:float = 0.1, double_disc=False, path_start: str = None) :
    """
        create Datasets objects with measurement bias
        One dataset is created for each biasing value in the list b_m
        Returns : Dictionary {float: StudentDataset}
        stud_biased_dict[b_m] = StudentDataset with bias level b_m
    """
    label_multi = 'label_multi'
    df_multiclass = dataset_orig.to_df_multiclass(label_multi)
    sens_attr = dataset_orig.protected_attribute_names[0]
    label = dataset_orig.label_names[0]
    data_biased_dict = {}
    for i in b_m :
        df_biased = measurement_biasing(df_multiclass, label_multi, sens_attr, b_m=i, double_disc=double_disc, noise=noise)
        df_biased.drop(label,axis=1,inplace=True)
        df_biased.rename(columns={label_multi:label}, inplace=True)
        dataset_biased = StandardDataset(df = df_biased,
                                        label_name=label,
                                        protected_attribute_names=[sens_attr],
                                        favorable_classes= (lambda n: n>=dataset_orig.metadata['multi_class_threshold']),
                                        privileged_classes=[[1.]],
                                        categorical_features=[],
                                        metadata=dataset_orig.metadata)
        data_biased_dict[i] = dataset_biased

    if path_start is not None :
        path = path_start+"_nBiasDatasets.pkl"
        with open(path,'wb') as file:
            pickle.dump(data_biased_dict,file)
        
    return data_biased_dict

#################################
# Undersampling related biasing #
#################################

def undersampling_biasing(df: pd.DataFrame, sens_attr: str, removal_distr:str,  p_u:float=None, nbr_removal:int=None, cond_attr:str=None, label: str=None, fav_one=True) :
    """ Add undersampling/selection bias to the given dataset by removing a proportion p_u of a specific group (unprivileged group for most removal_distr options)
        The removal_distr used for the experiment in Legast et al. "Influence of Label and Selection Bias on Fairness Intervention" EWAF 2025 is 'double_disc'.

    Parameters
    ----------
    df : Pandas DataFrame
        Dataset to which the bias needs to be added
    sens_attr : string
        Name of the sensitive attribute
    removal_distr : string, optional
        if 'random': the unprivileged instances are removed randomly
        if 'random_pos': unprivileged instances are randomly removed amongst those with positive label
        if 'normal': unprivileged instances removed with probabilities following the pdf of a normal distribution on unprivileged instances sorted by 'cond_attr' values (index with mean values of 'cond_attr' are more likely to be removed)
        if 'lower_weight': unprivileged instances removed with probabilities weighted according to 'cond_attr' values, lower values more likely to be removed. weight_i = (val_max-val_i)²/SumOfAll((val_max-val)²)
        if 'higher_weight': unprivileged instances removed with probabilities weighted according to 'cond_attr' values, higher values more likely to be removed. weight_i = val_i²/SumOfAll(val²)
        if 'double_disc': remove a proportion 'p_u' of unprivileged instances with positive label and the same proportion 'p_u' of privileged instances with negative label
        else the unprivileged individuals with lowest values are removed
    p_u : float
        If specified, proportion of the undersammpled group to remove, 0 means no instance is removed, 1 means all the instances are removed. If unspecified, 'nbr_removal' must be specified.
    nbr_removal:int
        If specified, number of individuals to remove from the undersampled group. This value is ignored if p_u is also specified.
    cond_attr : string, optional
        attribute on which the undersampling is conditioned, if None the unprivileged instances are removed randomly
    label : string, optional
        name of target label in the dataset 'df'
    fav_one : Boolean, optional
        Wether the favorable value for the sensitive attribute is 1 (True) or 0 (False)
        
    Returns
    -------
    Pandas DataFrame
        New DataFrame with the biased dataset
    """
    if p_u is None and nbr_removal is None :
        #TODO raise an actual error
        print("ERROR : At list one of p_u or nbr_removal needs to be specified")
        exit()
    unpriv = 1-fav_one
    df_unpriv = df.loc[df[sens_attr] == unpriv,:]
    num_unpriv = (df[sens_attr] == unpriv).sum()
    if p_u is not None :
        if nbr_removal is not None :
            print("WARNING : For undersampling, p_u will be used and nbr_removal will be ignored")
        nbr_unpriv_removal = int(num_unpriv*p_u) #nbr of instances to remove
    else :
        nbr_unpriv_removal = nbr_removal

    if nbr_unpriv_removal > 0 :
        if nbr_unpriv_removal >= num_unpriv : #all unpriv index. Needed to avoid error in sampling with probability
            drop_id = df_unpriv.index.to_list()
        elif cond_attr is None or removal_distr == 'random': #random selection of index to remove
            print("undersampling : random unpriv")
            id_unpriv = df_unpriv.index.to_list()
            drop_id = my_random_choice(id_unpriv,nbr_unpriv_removal)
        else : 
            if removal_distr == 'normal' : #probability to be removed follows the pdf of normal distribution, so index with mean values of cond_attr are more likely to be removed than outer values
                print("undersampling : normal")
                id_sorted = df_unpriv.sort_values(by=cond_attr, ascending=True).index.to_list()
                prob = normal_prob(len(id_sorted))
                drop_id = my_random_choice(id_sorted,nbr_unpriv_removal,p=prob)
            elif removal_distr == 'random_pos' :
                print("undersampling : random_pos")
                df_unpriv_pos = df_unpriv.loc[df[label] == 1,:]
                id_unpriv_pos = df_unpriv_pos.index.to_list()
                num_pos = len(id_unpriv_pos)
                if nbr_unpriv_removal <=  num_pos:
                    drop_id = my_random_choice(id_unpriv_pos,nbr_unpriv_removal)
                else :
                    df_unpriv_neg = df_unpriv.loc[df[label] == 0,:]
                    id_unpriv_neg = df_unpriv_neg.index.to_list()
                    drop_id_neg = my_random_choice(id_unpriv_neg,nbr_unpriv_removal-num_pos)
                    drop_id = [*id_unpriv_pos, *drop_id_neg]
            elif removal_distr == 'double_disc' : # Drop p_u*nbr_unpriv_pos and p_u*nbr_priv_neg) elements, with the undersampling proportional to the size of each of those two groups
                #print("undersampling : double_disc")
                df_unpriv_pos = df_unpriv.loc[df[label] == 1,:]
                id_unpriv_pos = df_unpriv_pos.index.to_list()
                nbr_unpriv_pos = len(id_unpriv_pos)
                df_priv = df.loc[df[sens_attr] == fav_one,:]
                df_priv_neg = df_priv.loc[df[label] == 0,:]
                id_priv_neg = df_priv_neg.index.to_list()
                nbr_priv_neg = len(id_priv_neg)
                #nbr_removal = p_u*(nbr_unpriv_pos+nbr_priv_neg)
                if p_u is not None :
                    nbr_priv_undersample = int(p_u*nbr_priv_neg)
                    nbr_unpriv_undersample = int(p_u*nbr_unpriv_pos)
                else :
                    coeff_priv_neg = nbr_priv_neg/(nbr_unpriv_pos+nbr_priv_neg)
                    nbr_priv_undersample = int(coeff_priv_neg*nbr_removal)
                    nbr_unpriv_undersample = int((1-coeff_priv_neg)*nbr_removal)
                drop_unpriv = my_random_choice(id_unpriv_pos,nbr_unpriv_undersample)
                drop_priv = my_random_choice(id_priv_neg,nbr_priv_undersample)
                drop_id = [*drop_unpriv, *drop_priv]
            elif removal_distr == 'lower_weight' : #probility is weighted according to cond_attr, lower values more likely to be removed
                print("undersampling : lower_weight")
                max_val = df_unpriv[cond_attr].max()
                mod_val = (max_val-df_unpriv[cond_attr]).pow(2)+0.25 # Addition of small value (< 1) needed so no individual has a probabability 0 to be removed
                sum_val = mod_val.sum()
                prob = mod_val/sum_val
                drop_id = my_random_choice(df_unpriv.index.to_list(),nbr_unpriv_removal,p=prob)
            elif removal_distr == 'higher_weight' : #probility is weighted according to cond_attr, higher values more likely to be removed
                print("undersampling : higher_weight")
                mod_val = df_unpriv[cond_attr].pow(2)+0.25 # Addition of small value (< 1) needed so no individual has a probabability 0 to be removed
                sum_val = mod_val.sum()
                prob = mod_val/sum_val
                drop_id = my_random_choice(df_unpriv.index.to_list(),nbr_unpriv_removal,p=prob)                
            else : #remove index with lowest values of cond_attr
                #This section is inspired by the 'Undersample' section of the dataset generation code from https://github.com/rcrupiISP/BiasOnDemand.
                print("undersampling : lowest vals")
                id_sorted = df.loc[df[sens_attr] == unpriv,:].sort_values(by=cond_attr, ascending=True).index.to_list()
                drop_id = id_sorted[:nbr_unpriv_removal]
        
        df_biased = df.drop(drop_id)
        return df_biased
    else :
        return df

def undersampling_independent_nbias(dataset_orig: StandardDataset, p_u: list, removal_distr: str, path_start: str = None) :
    """
        Undersampling of each bias value in list p_u is done independently
        create StandardDataset objects with sampling bias and store them in a dictionary
        One Dataset is created for each biasing value in the list p_u
        The dictionary is saved in "data/"+path+"_nBiasDatasets.pkl" if path is given
        Returns : Dictionary {float: StandardDataset}
        data_biased_dict[p_u] = StandardDataset with bias level p_u
    """
    label_multi = 'label_multi'
    df_multiclass = dataset_orig.to_df_multiclass(label_multi)
    sens_attr = dataset_orig.protected_attribute_names[0]
    label = dataset_orig.label_names[0]
    data_biased_dict = {}
    for i in p_u :
        df_biased = undersampling_biasing(df_multiclass, sens_attr, p_u=i,  removal_distr=removal_distr, cond_attr=label_multi, label=label) #, cond_attr=None, removal_distr='lower_weight')
        dataset_biased = StandardDataset(df = df_biased,
                                        label_name=label,
                                        protected_attribute_names=dataset_orig.protected_attribute_names,
                                        favorable_classes= (lambda n: n>=0.5), #No change of classes label in undersampling
                                        privileged_classes=[[1.]],
                                        categorical_features=[],
                                        features_to_drop=[label_multi],
                                        metadata=dataset_orig.metadata)
        data_biased_dict[i] = dataset_biased

    if path_start is not None :
        path = path_start+"_nBiasDatasets.pkl"
        with open(path,'wb') as file:
            pickle.dump(data_biased_dict,file)

    return data_biased_dict

def undersampling_incremental_nbias(dataset_orig: StandardDataset, removal_distr: str, incr_bias:float=0.1, bias_levels:list=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], path_start: str = None) :
    """ create StandardDataset objects with increamental sampling bias and store them in a dictionary
        The biased datasets are subsets of each others
        One Dataset is created for each biasing value in the bias_levels
    dataset_orig : StandardDataset
        Baseline dataset in which bias should be introduced
    incr_bias: float, optional
        increment added to bias level at each step
    bias_levels: list, optional
        list of the bias intensity for each resulting dataset. The increment between values must be 
    Returns : Dictionary {float: StandardDataset}
        data_biased_dict[p_u] = StandardDataset with bias level p_u
    """
    label_multi = 'label_multi'
    df_multiclass = dataset_orig.to_df_multiclass(label_multi)
    sens_attr = dataset_orig.protected_attribute_names[0]
    label = dataset_orig.label_names[0]
    # Get total number of people in group to be undersampled
    if removal_distr == 'double_disc':
        nbr_unpriv_pos = len(df_multiclass.loc[(df_multiclass[sens_attr] == 0) & (df_multiclass[label] == 1)].index.to_list())
        nbr_priv_neg = len(df_multiclass.loc[(df_multiclass[sens_attr] == 1) & (df_multiclass[label] == 0)].index.to_list())
        u_group_size = nbr_unpriv_pos+nbr_priv_neg
    else : # undersampled group is the underprivileged one
        u_group_size = len(df_multiclass.loc[df_multiclass[sens_attr] == 0,:].index.to_list())
    inc_removal = int(u_group_size*incr_bias) #nbr of individuals to remove at each biasing incrementation
    #inc_bias = (max_bias-min_bias)/step #can't use that because result is not rounded well
    df_biased = df_multiclass
    data_biased_dict = {}
    for b in bias_levels :
        #print("Working on bias level : "+str(b))
        if b > 0:
            df_biased = undersampling_biasing(df_biased, sens_attr, removal_distr=removal_distr, nbr_removal=inc_removal, cond_attr=label_multi, label=label) #, cond_attr=None, removal_distr='lower_weight')
        dataset_biased = StandardDataset(df = df_biased,
                                        label_name=label,
                                        protected_attribute_names=dataset_orig.protected_attribute_names,
                                        favorable_classes= (lambda n: n>=0.5), #No change of classes label in undersampling
                                        privileged_classes=[[1.]],
                                        categorical_features=[],
                                        features_to_drop=[label_multi],
                                        metadata=dataset_orig.metadata)
        data_biased_dict[b] = dataset_biased

    if path_start is not None :
        path = path_start+"_nBiasDatasets.pkl"
        with open(path,'wb') as file:
            pickle.dump(data_biased_dict,file)

    return data_biased_dict

def normal_prob(len: int):
    """Return an np.array of length len containing probabilities approximatly corresponding to the pdf of normal distribution for range(len)
    """
    r = np.arange(0,len)
    mean = np.mean(r)
    std = np.std(r)
    norm = st.norm(mean,std)
    prob = np.arange(0,len)
    prob = np.empty((len), float)
    for i in range(len) :
        prob[i] = norm.pdf(i)
    equ_term = (1-sum(prob))/len #equalizer term so all probabilities sum to 1
    for i in range(len) :
        prob[i] = prob[i] + equ_term
    return prob
#TODO could assign pdf according to label values instead of label ranking, maybe giving the threshold as the mean.

def my_random_choice(array, size, p=None) :
    """ Wrapper for numpy.random.Generator.choice """
    rng = np.random.default_rng(seed = 4224)
    choice = rng.choice(array,size, replace=False, p=p, shuffle=False)
    return choice


    