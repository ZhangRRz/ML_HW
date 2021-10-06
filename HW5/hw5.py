import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVC

# here is read data, fillna and one-hot coding
hw5_data = pd.read_csv("hr-analytics.csv", sep=",")
count = 0
X = pd.read_csv("hr-analytics.csv", sep=",")

for i in hw5_data.columns:

    if (is_numeric_dtype(hw5_data[i]) == False):
        X = X.drop([i], axis=1)
        df = pd.get_dummies(hw5_data[i])
        X = pd.concat([X, df], axis=1)
    else:
        hw5_data[i] = hw5_data[i].fillna(hw5_data[i].mean(skipna=True))
        if(i != "left"):
            X[i] = X[i].fillna(
                X[i].mean(skipna=True))

Y = pd.DataFrame(hw5_data['left'])

X_train, X_test, y_train, y_test = train_test_split(
    X.drop(['left'], axis=1), Y, test_size=0.3, random_state=1)

# ------------------------------------------------------------
log_reg = linear_model.LogisticRegression()
log_reg.fit(X_train, np.array(y_train))
y_pred = log_reg.predict(X_test)
c = 0
for i, j in zip(y_pred, np.array(y_test['left'])):
    if i == j:
        c += 1
print('Accuracy of logistic regression is:', c/len(y_test))

# ------------------------------------------------------------
svm_module = SVC()
svm_module.fit(X_train, np.array(y_train))
svm_y_pred = svm_module.predict(X_test)


c = 0
for i, j in zip(svm_y_pred, np.array(y_test['left'])):
    if i == j:
        c += 1
print('Accuracy of SVM is:', c/len(y_test))

# ------------------------------------------------------------

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
stand = svm_module.fit(scaler.transform(X_train), np.asarray(y_train))
s_pred = stand.predict(scaler.transform(X_test))

c = 0
for i, j in zip(s_pred, np.array(y_test['left'])):
    if i == j:
        c += 1
print('Accuracy of SVM after Standardlization is:', c / len(y_test))
print('it seems that Standardlizing X gave great effection to the Accuracy.')
