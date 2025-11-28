""" 
    Class of OULAD datasets
    Allows to create OULADsocial and OULADstem
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

class OULADDataset(StandardDataset):
    """Open University Learning Analytics Dataset
        See https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
        or https://www.nature.com/articles/sdata2017171
    """

    def __init__(self, df = None, domain:str = 'all', hard_problem:bool = False,
                 label_name='final_result',
                 favorable_classes= ["Distinction","Pass"], #(lambda n: n=="Pass" or n=="Distinction"),
                 protected_attribute_names=['gender'],
                 privileged_classes=[['F']],
                 instance_weights_name=None,
                 categorical_features=['code_module','code_presentation','region', 'highest_education', 'imd_band', 'disability', 'age_band'],
                 features_to_keep=[], features_to_drop=['id_student','index'],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings):
        """
            domain : 'all' for complete dataset, 'stem' for module FFF, 'social' for module BBB (with no duplicate students in all cases)
            hard_problem : True to make to prediction problem harder by only using features in studentInfo.csv, False to add VLE related features (recommended for higher accuracy)
            See :obj:`StandardDataset` for a description of the other arguments
        """

        if df is None :
            try :
                df = pd.read_csv(path_to_data+ 'OULAD/studentInfo.csv', na_values=na_values)
                if not hard_problem :
                    df_assessments = pd.read_csv(path_to_data+ 'OULAD/assessments.csv', na_values=na_values)
                    df_studentAssessment = pd.read_csv(path_to_data+ 'OULAD/studentAssessment.csv', na_values=na_values)
                    df_studentVle = pd.read_csv(path_to_data+ 'OULAD/studentVle.csv', na_values=na_values)
                    df_vle = pd.read_csv(path_to_data+ 'OULAD/vle.csv', na_values=na_values)
                    df_courses = pd.read_csv(path_to_data+ 'OULAD/courses.csv', na_values=na_values)
            except IOError as err:
                print("IOError: {}".format(err))
                print("To use this class, please download the files for the OULAD dataset, for example at this link:")
                print("\n\thttps://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset")
                print("\nand place it, as-is, in a folder dataset/OULAD")
                import sys
                sys.exit(1)
        
        #Filter unkown values (noted as '?' in csv file)
        df = df.replace(to_replace='?', value=np.nan).dropna()

        #Retrieve wanted students and 
        if domain != 'all' :
            categorical_features=['code_presentation','region', 'highest_education', 'imd_band', 'disability', 'age_band']
            features_to_drop=['id_student', 'index', 'code_module']
            if domain == 'stem':
                module = 'FFF'
            elif domain == 'social':
                module = 'BBB'
            #Select only students in the desired module
            df = df.loc[df['code_module'] == module,:]

        #Remove duplicated students, keep last occurence (last presentation (and course if domain==all) they did)
        for id in df.index :
            df_rest = df.loc[id:,:]
            same_students = df_rest.loc[df_rest['id_student'] == df_rest.loc[id,'id_student'],:]
            if len(same_students) > 1 :
                df.drop(id,inplace=True)

        if not hard_problem : # preprocess to add features related to VLE
        #Features related to VLE have been reproduced from Riazy et al, 2020 "Fairness in Learning Analytics: Student At-risk Prediction in Virtual Learning Environments" doi: 10.5220/0009324100150025
            #Reduce extra df to wanted module
            df_assessments = df_assessments[df_assessments['code_module']==module]
            df_studentVle = df_studentVle[df_studentVle['code_module']==module]
            df_vle = df_vle[df_vle['code_module']==module]
            df_courses = df_courses[df_courses['code_module']==module]

            #Compute wanted features from different dataframes    
            df_studentAssessment = df_studentAssessment.merge(df_assessments, how='inner', on='id_assessment',validate='many_to_one')
            for stud in df['id_student'] :
                stud_presentation = df.loc[df['id_student']==stud,'code_presentation'].iloc()[0] #use iloc to access value by position
                #Features num_CMA and num_TMA
                df_stud_assess = df_studentAssessment.loc[(df_studentAssessment['id_student']==stud)
                                                & (df_studentAssessment['code_presentation']==stud_presentation),:]
                df.loc[df['id_student']==stud,'num_CMA'] = len(df_stud_assess.loc[(df_studentAssessment['assessment_type']=='CMA'),:])
                df.loc[df['id_student']==stud,'num_TMA'] = len(df_stud_assess.loc[(df_studentAssessment['assessment_type']=='TMA'),:])
                df_stud_assess = None
                #df_CMA = df_studentAssessment.loc[(df_studentAssessment['id_student']==stud)
                #                                  & (df_studentAssessment['code_presentation']==df.loc[df['code_student']==stud,'code_presentation'])
                #                                  & (df_studentAssessment['assessment_type']=='CMA'),:]
                #Features related to VLE
                df_stud_vle = df_studentVle.loc[(df_studentVle['id_student']==stud)
                                            & (df_studentVle['code_presentation']==stud_presentation),:]
                #num_logings : total number of login over the module presentation
                num_logins = len(df_stud_vle)
                df.loc[df['id_student']==stud,'num_logins'] = num_logins
                #login_day : average number of logins per day
                days = df_courses.loc[df_courses['code_presentation']==stud_presentation,'module_presentation_length'].iloc()[0]
                df.loc[df['id_student']==stud,'login_day'] = num_logins/days
                #Number of clicks on resource pages 'forumng', 'glossary', 'homepage', 'resource'
                #forumng
                forum_id = df_vle.loc[(df_vle['code_presentation']==stud_presentation) & (df_vle['activity_type']=="forumng")
                                    ,'id_site'].iloc()[0]
                df_forum = df_stud_vle.loc[(df_stud_vle['code_presentation']==stud_presentation) 
                                        & (df_stud_vle['id_site']==forum_id),:]
                df.loc[df['id_student']==stud,'forumng'] = sum(df_forum['sum_click'])
                #glossary
                gloss_id = df_vle.loc[(df_vle['code_presentation']==stud_presentation) & (df_vle['activity_type']=="glossary")
                                    ,'id_site'].iloc()[0]
                df_gloss = df_stud_vle.loc[(df_stud_vle['code_presentation']==stud_presentation)
                                        & (df_stud_vle['id_site']==gloss_id),:]
                df.loc[df['id_student']==stud,'glossary'] = sum(df_gloss['sum_click'])
                #homepage
                home_id = df_vle.loc[(df_vle['code_presentation']==stud_presentation) & (df_vle['activity_type']=="homepage")
                                    ,'id_site'].iloc()[0]
                df_home = df_stud_vle.loc[(df_stud_vle['code_presentation']==stud_presentation)
                                        & (df_stud_vle['id_site']==home_id),:]
                df.loc[df['id_student']==stud,'homepage'] = sum(df_home['sum_click'])
                #resource
                reso_id = df_vle.loc[(df_vle['code_presentation']==stud_presentation) & (df_vle['activity_type']=="resource")
                                    ,'id_site'].iloc()[0]
                df_reso = df_stud_vle.loc[(df_stud_vle['code_presentation']==stud_presentation)
                                        & (df_stud_vle['id_site']==reso_id),:]
                df.loc[df['id_student']==stud,'resource'] = sum(df_reso['sum_click'])

            #Allow Python to free memory
            df_assessments = None
            df_studentVle = None
            df_vle = None
            df_courses = None            

        df.reset_index(inplace=True)
        
        #Keep information about final_result to be used in biasing methods
        #Values given to each label and to multi_class_threshold need to make sens for label biasing
        label_multi = df['final_result'].copy()
        label_multi.replace(to_replace="Fail", value='1', inplace=True)
        label_multi.replace(to_replace="Withdrawn", value='1', inplace=True)
        label_multi.replace(to_replace="Pass", value='3', inplace=True)
        label_multi.replace(to_replace="Distinction", value='4', inplace=True)
        label_multi = pd.Series(map(int,label_multi))
        #Values for threshold for favorable class at 2.0
        metadata = {'default mapping' : default_mappings,
                  'label multiclass' : label_multi,
                  'multi_class_threshold' : 2.5
                  }

        super(OULADDataset, self).__init__(df=df, label_name=label_name,
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