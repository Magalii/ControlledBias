""" 
    Class of Student datasets
    Allows to create Student and StudentBalanced
"""

import pandas as pd
import numpy as np

import sys
sys.path.append('../')
sys.path.append('parent_aif360')
from aif360.datasets import StandardDataset

path_to_data = 'dataset/'

default_mappings = {
    'label_maps': [{1.0: 'pass', 0.0: 'fail'}],
    'protected_attribute_maps': [{1.0: 'female', 0.0: 'male'}]
}
# default dataset has only 'sex' as protected attribute. To study 'age', you should load the dataset with "load_preproc_data_student"

class StudentMaleDataset(StandardDataset):
    """Student Performance UCI dataset (Portuguese subject), written by Magali Legast to be compatible with aif360 format
        uci repo dataset with id=320
        See https://archive.ics.uci.edu/dataset/320/student+performance
        or http://fairnessdata.dei.unipd.it
    """

    def __init__(self, df = None, balanced=False, label_name='G3',
                 favorable_classes= (lambda n: n>=10),
                 protected_attribute_names=['sex'],
                 privileged_classes=[['F']],
                 instance_weights_name=None,
                 categorical_features=['school','address','Pstatus', 'Mjob', 'Fjob', 'guardian', 'famsize', 'reason','schoolsup','famsup','activities','paid','internet','nursery','higher','romantic'],
                 features_to_keep=[], features_to_drop=[], #['G1','G2'] #G1 and G2 are previous grades. Removing those attributes increases the difficulty of the prediction task
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings):
        #See :obj:`StandardDataset` for a description of the arguments

        filepath = path_to_data + 'student_df'

        if df is None :
            try:
                df = pd.read_pickle(filepath)
                print("Student unpickled")
            except IOError as err:
                try :
                    from ucimlrepo import fetch_ucirepo 
                    student_performance = fetch_ucirepo(id=320)
                    df = student_performance.data.original
                    try :
                        pd.to_pickle(df,filepath)
                        print("Student downloaded because of error : "+str(err))
                    except Exception as err :
                        print("Student Dataset could not be downloaded. "+str(err))
                except ModuleNotFoundError :
                    print("ModuleNotFoundError: {}".format("Student Dataset could not be imported. Python module 'ucimlrepo' needs to be installed."))
                    sys.exit(1)
        
        if balanced :
            df_unpriv = df.loc[df['sex'] == 'F',:]
            df_unpriv_pos = df_unpriv.loc[df['G3'] >= 10,:]
            id_unpriv_pos = df_unpriv_pos.index.to_list()
            drop_id = self._my_random_choice(id_unpriv_pos,117)
            df = df.drop(drop_id)
            df.reset_index(inplace = True)

        label_multi = df['G3']
        metadata = {'default mapping' : default_mappings,
                  'label multiclass' : label_multi,
                  'multi_class_threshold' : 10
                  }

        super(StudentMaleDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing,
            metadata=metadata)

    def to_df_multiclass(self,label_multi: str) :

        df_orig = self.convert_to_dataframe()[0]
        multi_val = self.metadata['label multiclass']
        df_full = df_orig.copy()
        df_full[label_multi] = multi_val.values
        return df_full

    def _my_random_choice(self, array: list, size, p=None) :
        """ Wrapper for numpy.random.Generator.choice """
        rng = np.random.default_rng(seed = 4224)
        choice = rng.choice(array, size, replace=False, p=p, shuffle=False)
        return choice