# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:56:54 2018

@author: pariya pourmohammadi

pep-8 style

Classification code for lad change predictions
This code includes four machine learning models of
RF, SVM, voting ensemble, and logistic regression
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, cohen_kappa_score, log_loss
from scipy import interpolate as interp
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import time
import joblib
import DataPrep

code_path = '.'
data_path = '../Data'

os.chdir(code_path)
DataPrep.load(code_path, data_path)

'Run Classifier with cross-validation and plot ROC for each classifier'

## Number of splits for k-fold validation
n = 10

## Extra trees Classifier with 10 fold validation
## Classifier with cross-validation and plot ROC for each classifier
all_lines = open("Lines.txt", "r")
num_cols = 0
num_rows = 0

lines = all_lines.readlines()

if num_cols == 0 and num_rows == 0:
    num_cols = int(lines[0][14:])
    num_rows = int(lines[1][14:])
total_size = num_cols * num_rows

###################################################################
## Extra trees Classifier with 10 fold validation
###################################################################
xTrainValid = np.load('xTrainValid.npy')
yTrainValid = np.load('yTrainValid.npy')
data_test = np.load('data_test.npy')
label_test = np.load('label_test.npy')
data_train = np.load('data_train.npy')
label_train = np.load('label_train.npy')
data_valid = np.load('data_val.npy')
label_valid = np.load('label_val.npy')

estimators = np.arange(200, 2001, 200)
depths = np.arange(10, 101, 10)


def rf(est, dpt, X, Y, x_test, y_test):
    """

    Parameters
    ----------
    est : int
        number of estimators
    dpt : int
        max depth
    X : np array
        training data
    Y : np array
        training labels
    x_test : np array
        test data
    y_test : np array
        test labels

    Returns
    -------
    kappa : float
        kappa value
    """
    random_forest = ExtraTreesClassifier(n_estimators=est,
                                         max_depth=dpt,
                                         class_weight='balanced')

    random_forest.fit(X, Y.ravel())
    y_predict = random_forest.predict(x_test)
    kappa = cohen_kappa_score(y_test, y_predict)
    return kappa


resultsRF = [(depth,
              estimator,
              rf(depth,
                 estimator,
                 data_train,
                 label_train,
                 data_test,
                 label_test))
             for depth in depths for estimator in estimators]

kappa_values = list(list(zip(*resultsRF))[2])
index = kappa_values.index(max(kappa_values))
best_depth, best_est = resultsRF[index][0], resultsRF[index][1]

clfRF = ExtraTreesClassifier(n_estimators=best_est,
                             max_depth=best_depth,
                             class_weight='balanced')

cv = StratifiedKFold(n_splits=n,
                     shuffle=False)

true_positive_rate_rf = []
area_under_curve_rf = []
mean_fpr_RF = np.linspace(0, 5, 100)

i = 0

for train, test in cv.split(xTrainValid, yTrainValid):
    # fit RF model and get OOB decision function
    clfRF.fit(xTrainValid[train], yTrainValid[train].ravel())
    clfRF_probs = clfRF.predict_proba(xTrainValid[test])

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(yTrainValid[test], clfRF_probs[:, 1])
    true_positive_rate_rf.append(interp(mean_fpr_RF, fpr, tpr))
    true_positive_rate_rf[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucsRF.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    print(i)
    i += 1

mean_tpr_RF = np.mean(true_positive_rate_rf, axis=0)
mean_tpr_RF[-1] = 1.0
mean_auc_RF = auc(mean_tpr_RF, mean_tpr_RF)
std_auc_RF = np.std(aucsRF)
plt.plot(mean_fpr_RF, mean_tpr_RF, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (
             mean_auc_RF, std_auc_RF),
         lw=2, alpha=.8)

std_tpr = np.std(true_positive_rate_rf, axis=0)
tprs_upper_RF = np.minimum(mean_tpr_RF + std_tpr, 1)
tprs_lower_RF = np.maximum(mean_tpr_RF - std_tpr, 0)
plt.fill_between(mean_fpr_RF, tprs_lower_RF, tprs_upper_RF, color='grey',
                 alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate of Ensemble Classifier')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic of RF Classifier')
plt.legend(loc="lower right")
plt.show()
plt.savefig('RF_10fold')

yPredict_Total_RF = clfRF.predict(data_test)
RF_Kappa = cohen_kappa_score(label_test, yPredict_Total_RF)

np.save('clf_RF', clfRF)
np.savetxt(data_path + '/featureImportance.txt', clfRF.feature_importances_,
           fmt='%1.6f', delimiter=',')
np.save("yPredict_Total_RF", yPredict_Total_RF)


###################################################################
## MLP Classifier with 10 fold validation
###################################################################
clfMLP = MLPClassifier(hidden_layer_sizes=(22, 2), warm_start=True,
                       solver='adam',
                       learning_rate='adaptive', verbose=True,
                       early_stopping=True)

true_positive_rate_mlp = []
area_under_curve_mlp = []
mean_fpr_MLP = np.linspace(0, 1, 100)

i = 0


for train, test in cv.split(xTrainValid, yTrainValid):
    # fit MLP model on each fold
    clfMLP.fit(xTrainValid[train], yTrainValid[train].ravel())
    clfMLP_probs = clfMLP.predict_proba(xTrainValid[test])

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(yTrainValid[test], clfMLP_probs[:, 1])
    true_positive_rate_mlp.append(interp(mean_fpr_MLP, fpr, tpr))
    true_positive_rate_mlp[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    area_under_curve_mlp.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    print(i)
    i += 1

mean_tpr_MLP = np.mean(true_positive_rate_mlp, axis=0)
mean_tpr_MLP[-1] = 1.0
mean_auc_MLP = auc(mean_tpr_MLP, mean_tpr_MLP)
std_auc_MLP = np.std(area_under_curve_mlp)
plt.plot(mean_fpr_MLP, mean_tpr_MLP, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (
             mean_auc_MLP, std_auc_MLP),
         lw=2, alpha=.8)

std_tpr = np.std(true_positive_rate_mlp, axis=0)
tprs_upper_MLP = np.minimum(mean_tpr_MLP + std_tpr, 1)
tprs_lower_MLP = np.maximum(mean_tpr_MLP - std_tpr, 0)
plt.fill_between(mean_fpr_MLP, tprs_lower_MLP, tprs_upper_MLP, color='grey',
                 alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate of Neural Networks Classifier')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic of Neural Networks Classifier')
plt.legend(loc="lower right")
plt.show()
plt.savefig('MLP_10fold')

xTrainValid, yTrainValid = None, None

yPredict_Total_MLP = clfMLP.predict(data_test)

np.save('clf_MLP', clfMLP)
np.save("yPredict_Total_MLP", yPredict_Total_MLP)

###################################################################
# voting classifiers Classifier with 10 fold validation
# The classifiers of this ensemble include LR, GaussianNB and DT
###################################################################
cv = StratifiedKFold(n_splits=10)
clf1 = LogisticRegression()
clf2 = GaussianNB()
clf3 = DecisionTreeClassifier()
clf_voting = VotingClassifier(
    estimators=[('lr', clf1), ('GNB', clf2), ('DT', clf3)])

true_positive_rate_voting = []
area_under_curve_voting = []
mean_fpr_voting = np.linspace(0, 1, 100)

i = 0

for train, test in cv.split(xTrainValid, yTrainValid):
    # fit voting ensamble model on each fold
    clf_voting.fit(xTrainValid[train], yTrainValid[train].ravel())
    clf_voting_probs = clf_voting.predict_proba(xTrainValid[test])

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(yTrainValid[test], clf_voting_probs[:, 1])
    true_positive_rate_voting.append(interp(mean_fpr_voting, fpr, tpr))
    true_positive_rate_voting[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    area_under_curve_voting.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    print(i)
    i += 1

mean_tpr_voting = np.mean(true_positive_rate_voting, axis=0)
mean_tpr_voting[-1] = 1.0
mean_tpr_voting = auc(mean_tpr_voting, mean_tpr_voting)
std_auc_voting = np.std(area_under_curve_voting)
plt.plot(mean_fpr_voting, mean_tpr_voting, color='pink',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)',
         lw=2, alpha=.8)

std_tpr = np.std(true_positive_rate_voting, axis=0)
tprs_upper_voting = np.minimum(mean_tpr_voting + std_tpr, 1)
tprs_lower_voting = np.maximum(mean_tpr_voting - std_tpr, 0)
plt.fill_between(mean_fpr_voting, tprs_lower_voting, tprs_upper_voting,
                 color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate of Neural Networks Classifier')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic of Neural Networks Classifier')
plt.legend(loc="lower right")
plt.show()
plt.savefig('Voting_10fold')

xTrainValid, yTrainValid = None, None

yPredict_Total_voting = clf_voting.predict(data_test)

np.save('clf_voting', clf_voting)
np.save("yPredict_Total_voting", yPredict_Total_voting)

###################################################################
# Convert the result back to txt file for Visualization
###################################################################
LUBaseRaw = np.genfromtxt('landuse_base.txt', delimiter=' ')
np.array(LUBaseRaw)
LUBase = LUBaseRaw.reshape((total_size, 1))

moist = np.genfromtxt('moisture.txt', delimiter=' ')
np.array(moist)
moist = moist.reshape((total_size, 1))

partition = np.load('partition.npy')
data_train = np.load('data_train.npy')
data_val = np.load('data_val.npy')

y = moist
y[(moist != -9999.0) & (partition == 1)] = data_train[:, 0]
y[(moist != -9999.0) & (partition == 2)] = yPredict_Total
y[(moist != -9999.0) & (partition == 3)] = data_val[:, 0]
y = y.reshape((num_rows, num_cols))
np.savetxt(data_path + '/y_predict.txt', y, fmt='%1.1f', delimiter=' ')
