import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from xgboost.sklearn import XGBClassifier


def f1score_cul(CMmatrix):
    sensitivity = CMmatrix[0][0] / (CMmatrix[0][0] + CMmatrix[1][0])
    precision = CMmatrix[0][0] / (CMmatrix[0][0] + CMmatrix[0][1])

    return 2/(1/sensitivity+1/precision)


# # here is read data, fillna and one-hot coding
# hw6_data = pd.read_csv("hr-analytics.csv", sep=",")
# count = 0
# X = pd.read_csv("hr-analytics.csv", sep=",")

# for i in hw6_data.columns:

#     if (is_numeric_dtype(hw6_data[i]) == False):
#         X = X.drop([i], axis=1)
#         df = pd.get_dummies(hw6_data[i])
#         X = pd.concat([X, df], axis=1)
#     else:
#         hw6_data[i] = hw6_data[i].fillna(hw6_data[i].mean(skipna=True))
#         if(i != "left"):
#             X[i] = X[i].fillna(
#                 X[i].mean(skipna=True))

# Y = pd.DataFrame(hw6_data['left'])

# X_train, X_test, y_train, y_test = train_test_split(
#     X.drop(['left'], axis=1), Y, test_size=0.3, random_state=1)

# # ---------------------------Logistic-------------------------
# log_reg = linear_model.LogisticRegression()
# log_reg.fit(X_train, np.array(y_train))
# y_pred = log_reg.predict(X_test)
# c = 0
# for i, j in zip(y_pred, np.array(y_test['left'])):
#     if i == j:
#         c += 1
# print('Accuracy of logistic regression is:', c/len(y_test))

# # ----------------------------SVC-----------------------------
# svm_module = SVC()
# svm_module.fit(X_train, np.array(y_train))
# svm_y_pred = svm_module.predict(X_test)

# c = 0
# for i, j in zip(svm_y_pred, np.array(y_test['left'])):
#     if i == j:
#         c += 1
# print('Accuracy of SVM is:', c/len(y_test))

# # --------------------------decision tree---------------------
# hw6_tree = DecisionTreeClassifier(max_depth=10)
# hw6_tree.fit(X_train, y_train)
# tree_predict = hw6_tree.predict(X_test)

# c = 0
# for i, j in zip(tree_predict, np.array(y_test['left'])):
#     if i == j:
#         c += 1
# print('Accuracy of DecisionTree is:', c/len(y_test))
# --------------------------part1 con---------------------


# ---------------------------------------------------------
# 3. 用以預測存活與否的欄位，其餘可以自行決定要如何採用。
# 4. 對非數字類型的資料如何編碼、缺失值如何填補也是自行決定。
# 5. 特徵是否進行轉換，或是增加額外特徵，也請自行決定。
# 6. 每個模型針對873 位乘客的存活預測結果與實際結果繪製

# ----------------------------train set----------------------------------------
hw6_data_part2 = pd.read_csv("titanic_train.csv", sep=",")
hw6_data_part2_test = pd.read_csv("titanic_test.csv", sep=",")
count = 0


merge_df = pd.concat([hw6_data_part2, hw6_data_part2_test], axis=0)
# print(merge_df)

Y = pd.DataFrame(hw6_data_part2['survived'])

# drop :body,home.dest,ticket(because pclass),name

merge_df = merge_df.drop(['body', 'home.dest', 'ticket', 'name'], axis=1)
X = merge_df.drop(['survived'], axis=1)

# print(X)

for i in X.columns:

    if (is_numeric_dtype(X[i]) == False):
        df = pd.get_dummies(merge_df[i])
        X = X.drop([i], axis=1)
        X = pd.concat([X, df], axis=1)
    else:
        X[i] = X[i].fillna(X[i].mean(skipna=True))

X_test_473 = X.tail(len(df) - 873)
X = X.head(873)

# print(X_test)
# print(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1)


# -----------------------------SVM---------------------------------
svm_module = SVC()
svm_module.fit(X_train, np.array(y_train))
svm_y_pred = svm_module.predict(X_test)

svm_CM = confusion_matrix(y_test, svm_y_pred)
print("svm CM is:\n")
sns.heatmap(svm_CM, annot=True)
print("svm f1 score is:", f1score_cul(svm_CM))

# -----------------------------random forest---------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, np.array(y_train))
rf_y_pred = rf.predict(X_test)

RandomForest_CM = confusion_matrix(y_test, rf_y_pred)
print("RandomForest CM is:\n")
sns.heatmap(RandomForest_CM, annot=True)
print("RandomForest f1 score is:", f1score_cul(RandomForest_CM))

# -----------------------------XGBoost---------------------------------
