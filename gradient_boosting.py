from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score



traindata = pd.read_csv("preprocessed_train.csv")
testdata = pd.read_csv("preprocessed_test.csv")

traindata.pop("Unnamed: 0")
testdata.pop("Unnamed: 0")

traindata.pop("FLAGS_-------")
testdata.pop("FLAGS_-------")

traindata.pop("FLAGS_---A---")
testdata.pop("FLAGS_---A---")

traindata.head()

Y = traindata.pop('PKT_CLASS')
X = traindata.iloc[:,0:13]
C = testdata.pop('PKT_CLASS')
T = testdata.iloc[:,0:13]

X.info(verbose=True)

_traindata = np.array(X)
_trainlabel = np.array(Y)

_testdata = np.array(T)
_testlabel = np.array(C)

_testlabel

# 5. Declare data preprocessing steps
pipeline = make_pipeline(GradientBoostingClassifier())

# Add a dict of estimator and estimator related parameters in this list
hyperparameters = {
                'gradientboostingclassifier__n_estimators': [25,50,75,100],
                'gradientboostingclassifier__max_features' : [None, "log2", "auto"]
                }

# 7. Tune model using cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=5,verbose=1,n_jobs=-1)
clf.fit(_traindata, _trainlabel)

print(clf.best_params_)
print(clf.best_estimator_)
# print(clf.cv_results_ )
print(clf.best_score_ )
print(clf.refit)
# 9. Evaluate model pipeline on test data
pred = clf.predict(_testdata)
from sklearn.metrics import accuracy_score
print(accuracy_score(_testlabel, pred))
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(_testlabel, pred)
print(classification_report(_testlabel, pred))
print(cm)
