# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 16:26:54 2019

@author: Mahmoud Nada
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Proccessing Training Data
training_data = pd.read_csv("training.csv", sep = ";")
y = training_data["classLabel"]
y = y.map({'yes.': 1, 'no.': 0})
X = training_data.drop("classLabel", axis = 1)
validation_data = pd.read_csv("validation.csv", sep = ";")

#Handling missing Data
print(X.isnull().values.sum())
X.fillna(method='ffill', inplace=True)
print(X.isnull().values.sum())

# Encoding Categorical variables
X1 = pd.get_dummies(X[['variable1', 'variable4', 'variable5', 'variable6', 'variable7',
                       'variable9','variable10','variable12','variable13','variable18']])
encoded = list(X1.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))
print(encoded)

X = X.drop(columns=['variable1', 'variable4', 'variable5', 'variable6', 'variable7',
        'variable9','variable10','variable12','variable13','variable18'])
X = X.join(X1)

def arrangeDF(df):
    var = df.values.tolist()
    var2 = [str(var[i]).split(',') for i in range(len(var))]
    df = pd.DataFrame(var2)
    df = df.astype(float)
    if df.isnull().values.sum() > 0:
        df.fillna(method='ffill', inplace=True)
    return df,df.isnull().values.sum()

df1, check1 = arrangeDF(X['variable2'])
df2, check = arrangeDF(X['variable3'])
df3, check2 = arrangeDF(X['variable8'])

df1.columns = ['variable2_1', 'variable2_2']
df2.columns = ['variable3_1', 'variable3_2']
df3.columns = ['variable8_1', 'variable8_2']

X = X.drop(columns=['variable2', 'variable3', 'variable8'])
X = X.join(df1)
X = X.join(df2)
X = X.join(df3)
"""
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
X = scaler.fit_transform(X)"""
###########################################################################################
# Processing Validation Data
y_valid = validation_data["classLabel"]
y_valid = y_valid.map({'yes.': 1, 'no.': 0})
X_valid = validation_data.drop("classLabel", axis = 1)

# Encoding Categorical variables
X1_valid = pd.get_dummies(X_valid[['variable1', 'variable4', 'variable5', 'variable6', 'variable7',
                       'variable9','variable10','variable12','variable13','variable18']])
encoded_valid = list(X1_valid.columns)
print("{} total features after one-hot encoding.".format(len(encoded_valid)))
print(encoded_valid)

missing_columns = [i for i in encoded if i not in encoded_valid]
missind_df = pd.DataFrame([], columns=missing_columns)

X_valid = X_valid.drop(columns=['variable1', 'variable4', 'variable5', 'variable6', 'variable7',
        'variable9','variable10','variable12','variable13','variable18'])
X_valid = X_valid.join(X1_valid)

X1_valid = X1_valid.join(missind_df) 
X_valid = X_valid.join(missind_df)
X_valid[missing_columns] =X_valid[missing_columns].fillna(0)
X1_valid[missing_columns] = X1_valid[missing_columns].fillna(0)

df1_V, check1_V = arrangeDF(X_valid['variable2'])
df2_V, check_V = arrangeDF(X_valid['variable3'])
df3_V, check2_V = arrangeDF(X_valid['variable8'])

df1_V.columns = ['variable2_1', 'variable2_2']
df2_V.columns = ['variable3_1', 'variable3_2']
df3_V.columns = ['variable8_1', 'variable8_2']

X_valid = X_valid.drop(columns=['variable2', 'variable3', 'variable8'])
X_valid = X_valid.join(df1)
X_valid = X_valid.join(df2)
X_valid = X_valid.join(df3)

#Handling missing Data
print(X_valid.isnull().values.sum())
X_valid.fillna(method='ffill', inplace=True)
print(X_valid.isnull().values.sum())
"""
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler_valid = MinMaxScaler() # default=(0, 1)
X_valid = scaler.fit_transform(X_valid)"""
####################################Imports#########################################
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
################################Classifiers#################################################
# SVC Classifiear
clf = SVC(random_state=0)
clf.fit(X, y)
y_predSVC = clf.predict(X_valid)
cmSVC = confusion_matrix(y_valid, y_predSVC)
aScoreSVC = accuracy_score(y_valid, y_predSVC)
fScoreSVC = f1_score(y_valid, y_predSVC)
print("SVC Classifier Accuracy : ", aScoreSVC)
print("SVC Classifier f1 score : ", fScoreSVC)
print("confusion matrix of SVC Classifier : ")
print(cmSVC)
########################################################################################
# AdaBoost Classifiear using DecisionTreeClassifier
clf1 = AdaBoostClassifier(base_estimator = DecisionTreeClassifier())
clf1.fit(X, y)
y_pred = clf1.predict(X_valid)
cm = confusion_matrix(y_valid, y_pred)
aScore = accuracy_score(y_valid, y_pred)
fScore = f1_score(y_valid, y_pred)
print("AdaBoost Classifier Accuracy : ", aScore)
print("AdaBoost Classifier f1 score : ", fScore)
print("confusion matrix of AdaBoost Classifier : ")
print(cm)
########################################################################################
# Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(X, y)
GaussianNB(priors=None, var_smoothing=1e-09)
y_pred_gnb = clf_gnb.predict(X_valid)
cm_gnb = confusion_matrix(y_valid, y_pred_gnb)

aScore_gnb = accuracy_score(y_valid, y_pred_gnb)
fScore_gnb = f1_score(y_valid, y_pred_gnb)

print("Gaussian Naive Bayes Classifier Accuracy : ", aScore_gnb)
print("Gaussian Naive Bayes Classifier f1 score : ", fScore_gnb)
print("confusion matrix of Gaussian Naive Bayes Classifier : ")
print(cm_gnb)
########################################################################################
# K Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
clf_k = KNeighborsClassifier()
clf_k.fit(X, y)
y_pred_k = clf_k.predict(X_valid)
cm_k = confusion_matrix(y_valid, y_pred_k)

aScore_k = accuracy_score(y_valid, y_pred_k)
fScore_k = f1_score(y_valid, y_pred_k)

print("K Nearest Neighbors Classifier Accuracy : ", aScore_k)
print("K Nearest Neighbors Classifier f1 score : ", fScore_k)
print("confusion matrix of K Nearest Neighbors Classifier : ")
print(cm_k)
########################################################################################
# K Nearest Neighbors Hyper-Parameter Tuning using grid search
clf_kgc = KNeighborsClassifier()
parameter_kgc = {
        'n_neighbors':[3,5,10,15,20],
        'weights':['uniform', 'distance'],
        'metric':['euclidean', 'manhattan', 'minkowski']
        }
scorer_kgc = make_scorer(accuracy_score)
grid_obj_kgc = GridSearchCV(clf_kgc, parameter_kgc, scoring = scorer_kgc, verbose=1, cv=3, n_jobs=-1)
grid_fit_kgc = grid_obj_kgc.fit(X, y)
best_clf_kgc = grid_fit_kgc.best_estimator_
best_clf_kgc.fit(X, y)
best_train_prediction_kgc = best_clf_kgc.predict(X)
best_valid_prediction_kgc = best_clf_kgc.predict(X_valid)
print('the training accuracy of Optimized K Nearest Neighbors Classifier is :', accuracy_score(best_train_prediction_kgc, y))
print('the validation accuracy of Optimized K Nearest Neighbors Classifier is :', accuracy_score(best_valid_prediction_kgc, y_valid))
print('the validation F1 score of Optimized K Nearest Neighbors Classifier is :', f1_score(best_valid_prediction_kgc, y_valid))

########################################################################################
# Stochastic Gradient Descent Classifier 
from sklearn import linear_model
clf_sgdc = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
clf_sgdc.fit(X, y)
y_pred_sgdc = clf_sgdc.predict(X_valid)
cm_sgdc = confusion_matrix(y_valid, y_pred_sgdc)

aScore_sgdc = accuracy_score(y_valid, y_pred_sgdc)
fScore_sgdc = f1_score(y_valid, y_pred_sgdc)

print("Stochastic Gradient Descent Classifier Accuracy : ", aScore_sgdc)
print("Stochastic Gradient Descent Classifier f1 score : ", fScore_sgdc)
print("confusion matrix of Stochastic Gradient Descent Classifier : ")
print(cm_sgdc)
########################################################################################
# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
clf_lr =LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X, y)
y_pred_lr = clf_lr.predict(X_valid)
cm_lr = confusion_matrix(y_valid, y_pred_lr)

aScore_lr = accuracy_score(y_valid, y_pred_lr)
fScore_lr = f1_score(y_valid, y_pred_lr)

print("LogisticRegression Classifier Accuracy : ", aScore_lr)
print("LogisticRegression Classifier f1 score : ", fScore_lr)
print("confusion matrix of LogisticRegression Classifier : ")
print(cm_lr)
########################################################################################
# Tuning the hyper-parameters of DecisionTreeClassifier using Gridsearch
clf_gcd = DecisionTreeClassifier(random_state=42)
parameters_gcd = {'max_depth':[50, 100, 200],
                  'min_samples_split':[2, 4, 6, 8], 'min_samples_leaf':[2,4,6,8]}
scorer_gcd = make_scorer(accuracy_score)
grid_obj_gcd = GridSearchCV(clf_gcd, parameters_gcd, scoring = scorer_gcd)
grid_fit_gcd = grid_obj_gcd.fit(X, y)
best_clf_gcd = grid_fit_gcd.best_estimator_
best_clf_gcd.fit(X, y)
best_train_prediction_gcd = best_clf_gcd.predict(X)
best_valid_prediction_gcd = best_clf_gcd.predict(X_valid)
print('the training accuracy of Optimized DecisionTree Classifier is :', accuracy_score(best_train_prediction_gcd, y))
print('the validation accuracy of Optimized DecisionTree Classifier is :', accuracy_score(best_valid_prediction_gcd, y_valid))
print('the validation F1 score of Optimized DecisionTree Classifier is :', f1_score(best_valid_prediction_gcd, y_valid))
########################################################################################
# Tuning the hyper-parameters of Support Vector Classsifier using Gridsearch
"""clf_gcsvc = SVC(random_state=42)
parameters_gcsvc = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
scorer_gcsvc = make_scorer(accuracy_score)
grid_obj_gcsvc = GridSearchCV(clf_gcsvc, parameters_gcsvc, scoring = scorer_gcsvc, n_jobs=-1, verbose=1)
grid_fit_gcsvc = grid_obj_gcsvc.fit(X, y)
best_clf_gcsvc = grid_fit_gcsvc.best_estimator_
best_clf_gcsvc.fit(X, y)
best_train_prediction_gcsvc = best_clf_gcsvc.predict(X)
best_valid_prediction_gcsvc = best_clf_gcsvc.predict(X_valid)
print('the training accuracy is :', accuracy_score(best_train_prediction_gcsvc, y))
print('the validation accuracy is :', accuracy_score(best_valid_prediction_gcsvc, y_valid))
print('the validation F1 score is :', f1_score(best_valid_prediction_gcsvc, y_valid))
best_clf_gcsvc"""
########################################################################################
# Neural network Classifier
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

clf_nn = Sequential()
clf_nn.add(Dense(512, activation='relu', input_shape=(53,)))
clf_nn.add(Dropout(.3))
clf_nn.add(Dense(256, activation='relu'))
clf_nn.add(Dropout(.3))
clf_nn.add(Dense(128, activation='relu'))
clf_nn.add(Dropout(.3))
clf_nn.add(Dense(64, activation='relu'))
clf_nn.add(Dropout(.2))
clf_nn.add(Dense(1, activation='sigmoid'))

clf_nn.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
clf_nn.summary()

clf_nn.fit(X, y, epochs=50,batch_size=100)
score_nn=clf_nn.evaluate(X_valid, y_valid)
print("Neural network Classifier Accuracy : ", score_nn[1])


