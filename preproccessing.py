from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

class dataset:
    pass

df = pd.read_pickle('final_dataset.pkl')

print(df.head())

df = df[['SRC_ADD','DES_ADD','PKT_ID','FROM_NODE','TO_NODE','PKT_TYPE','PKT_SIZE','FLAGS','PKT_CLASS']]

df.dropna(inplace=True)
df.info(verbose=True)

dataset.train = df.groupby('PKT_CLASS').apply(pd.DataFrame.sample, frac=0.8).reset_index(level='PKT_CLASS', drop=True)
dataset.test = df.drop(dataset.train.index)
dataset.label = dataset.train.PKT_CLASS.copy()
dataset.train
print(dataset.label.unique())

d1 = dataset.train.replace('Normal', 0)
d2 = d1.replace('UDP-Flood', 1)
d3 = d2.replace('Smurf', 1)
d4 = d3.replace('HTTP-FLOOD', 1)
d5 = d4.replace('SIDDOS', 1)

d6_label = d5.PKT_CLASS.copy()
d6_label.unique()
d6_label.value_counts()

dataset.test_label = dataset.test.PKT_CLASS.copy()
dataset.test_label.unique()

a1_label = dataset.test.PKT_CLASS.copy()
a1_label.unique()

a1 = dataset.test.replace('Normal', 0)
a2 = a1.replace('UDP-Flood', 1)
a3 = a2.replace('Smurf', 1)
a4 = a3.replace('SIDDOS', 1)
a5 = a4.replace('HTTP-FLOOD', 1)

a5_label = a5.PKT_CLASS.copy()
a5_label.unique()
a5_label.value_counts()


class preprocessing:
    train_labels = pd.get_dummies(d5) #columns = dummy_variables_2labels, prefix=dummy_variables_2labels)
    test_labels = pd.get_dummies(a5) #columns = dummy_variables_2labels, prefix=dummy_variables_2labels)

preprocessing.test_labels.info(verbose=True)
d5.head()
preprocessing.test_labels.head()
preprocessing.train_labels.to_csv("preprocessed_train.csv")
preprocessing.test_labels.to_csv("preprocessed_test.csv")
preprocessing.train_labels.head()
