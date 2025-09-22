import pickle
import sys
sys.path.append('..')

from aif360.datasets import StandardDataset

import dataset_biasing as db

def create_dataset(dataset_name: str, bias_name: str, bias_levels: list[float], path:str=None) :
    if dataset_name[0:7] == 'student' :
        if dataset_name == 'studentM' :
            from dataset.studentMale_dataset import StudentMaleDataset
            student_orig = StudentMaleDataset(balanced=False)
        else :
            student_orig = None
        if bias_name == 'label' :
            nbias_data_dict = student_mislabeling(bias_levels, path=path, student_orig=student_orig)
        elif bias_name == 'selectDouble' :
            nbias_data_dict = student_undersampling(bias_levels, removal_distr='double_random_disc', path=path, student_orig=student_orig)
        elif bias_name == 'selectLow' :
            nbias_data_dict = student_undersampling(bias_levels, removal_distr='lower_weight', path=path, student_orig=student_orig)
        else :
            print("WARNING : Not a valid bias name")
    else :
        print("WARNING : Not a valid dataset name")
    return nbias_data_dict

def student_undersampling(p_u: list, removal_distr: str, path: str = None, student_orig=None) :
    """
        create Student Datasets objects with sampling bias
        One Student Dataset is created for each biasing value in the list p_u
        The datasets are each saved in "data/student_select_"+[p_u value]+".pkl" if save==True
        Returns : Dictionary {float: StudentDataset}
        stud_biased_dict[p_u] = StudentDataset with bias level p_u
    """
    if student_orig==None :
        from dataset.student_dataset import StudentDataset
        student_orig = StudentDataset(balanced=True) #Standard preproc -> 'age' is kept as numerical, sensitive attribute is 'sex'
    print(student_orig)
    label_multi = 'label_multi'
    df_multiclass = student_orig.to_df_multiclass(label_multi)
    sens_attr = 'sex'
    label = 'G3'
    stud_biased_dict = {}
    for i in p_u :
        df_biased = db.undersampling_biasing(df_multiclass, sens_attr, p_u=i,  removal_distr=removal_distr, cond_attr=label_multi, label=label) #, cond_attr=None, removal_distr='lower_weight')
        student_biased = StandardDataset(df = df_biased,
                                        label_name=label,
                                        protected_attribute_names=student_orig.protected_attribute_names,
                                        favorable_classes= (lambda n: n>=0.5),
                                        privileged_classes=[[1.]],
                                        categorical_features=[],
                                        features_to_drop=[label_multi],
                                        metadata=student_orig.metadata)
        stud_biased_dict[i] = student_biased

    if path is not None :
        path = path+"_nBiasDatasets.pkl"
        with open(path,'wb') as file:
            pickle.dump(stud_biased_dict,file)

    return stud_biased_dict

def student_mislabeling(b_m: list, path: str = None, student_orig=None) :
    """
        create Student Datasets objects with measurement bias
        One Student Dataset is created for each biasing value in the list b_m
        The datasets are each saved in "data/student_label_"+[p_u value]+".pkl" if save==True
        Returns : Dictionary {float: StudentDataset}
        stud_biased_dict[b_m] = StudentDataset with bias level b_m
    """
    if student_orig==None :
        from dataset.student_dataset import StudentDataset
        student_orig = StudentDataset(balanced=True) #Standard preproc -> 'age' is kept as numerical, sensitive attribute is 'sex'
    label_multi = 'label_multi'
    df_multiclass = student_orig.to_df_multiclass(label_multi)
    sens_attr = 'sex'
    label = student_orig.label_names[0]
    stud_biased_dict = {}
    for i in b_m :
        df_biased = db.measurement_biasing(df_multiclass, label_multi, sens_attr, b_m=i, noise=0.1)
        df_biased.drop(label,axis=1,inplace=True)
        df_biased.rename(columns={label_multi:label}, inplace=True)
        student_biased = StandardDataset(df = df_biased,
                                        label_name=label,
                                        protected_attribute_names=student_orig.protected_attribute_names[0],
                                        favorable_classes= (lambda n: n>=10),
                                        privileged_classes=[[1.]],
                                        categorical_features=[],
                                        metadata=student_orig.metadata)
        stud_biased_dict[i] = student_biased

    if path is not None :
        path = path+"_nBiasDatasets.pkl"
        with open(path,'wb') as file:
            pickle.dump(stud_biased_dict,file)
        
    return stud_biased_dict

