from aif360.datasets import StandardDataset

from dataset.studentMale_dataset import StudentMaleDataset
import consistency_metrics as cm

data_orig = StudentMaleDataset()
label = 'G3'
sens_attr = data_orig.protected_attribute_names[0]
priv_val = 1
unpriv_val = 0

df_orig = data_orig.convert_to_dataframe()[0]
#print(df_orig)

df_small = df_orig[639:649]
df_small = df_small.loc[:,['sex','Medu','famrel','G3']]

X = df_small.loc[:,['sex','Medu','famrel']]
y = df_small[label]

print(df_small)


student_small = StandardDataset(df = df_small,
                                        label_name=label,
                                        protected_attribute_names=[data_orig.protected_attribute_names[0]],
                                        favorable_classes= [data_orig.favorable_label],
                                        privileged_classes=[[1.]],
                                        categorical_features=[],
                                        features_to_drop=[],
                                        metadata=data_orig.metadata)

print(student_small)

dataset = data_orig

const_BLD = cm.consistency(dataset)
print("const_BLD =" +str(const_BLD))

#const_score = cm.consistency_score(X, y, n_neighbors=5)
#print("const_score = "+str(const_score))

bcc = cm.bcc(dataset)
print("BCC = "+str(bcc))

bcc_penalty = cm.bcc(dataset, penalty = 1)
print("BCC penalty = "+str(bcc_penalty))
