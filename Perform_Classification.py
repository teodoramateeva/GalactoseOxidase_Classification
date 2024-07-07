#Import libraries
import warnings
warnings.filterwarnings('ignore') 
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import RocCurveDisplay
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn import metrics
from itertools import product
from collections import defaultdict, Counter
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_curve, auc, accuracy_score
import statistics
from random import shuffle
from sklearn.feature_selection import RFE
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score     
import math
from matplotlib.colors import Normalize

# Define the path to your file storing the features
df = pd.read_csv('features_fromMD.csv') 
# Update your X1 based on which features were chosen in the run of Feature_Select.py
X1 = df.iloc[:, [11, 36, 21, 46, 6]]
Y = df.iloc[:,49]
print(Y)

# Perorm the Classification
def PerformClassification(X1, Y):
    all_fpr = []
    all_tpr = []
    all_auc = []
    all_acc = []
    all_classification_reports = []
    all_precisions = []
    all_recalls = []

    all_classification_reports = []
    all_predictions = pd.DataFrame(columns=['Fold', 'Actual_Labels', 'Predicted_Labels'])

    random_state = 12883823
    rkf = RepeatedKFold(n_splits=2, n_repeats=100, random_state=random_state)

    for i, (train_index, test_index) in enumerate(rkf.split(X1)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        
        X_train, X_test = X1.iloc[train_index], X1.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        
        rrt = RandomForestClassifier(n_estimators=10, n_jobs=-1) 
        rrt.fit(X_train, Y_train)
        pred_values = rrt.predict(X_test)
        report = classification_report(Y_test, pred_values, output_dict=True)
        all_classification_reports.append(report)
        
        print(classification_report(Y_test, pred_values, output_dict=True))
        print(Y_test)
        print(pred_values)

        # Get class probabilities for a ROC curve in case you want to plot that
        probas_ = rrt.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])
        
        acc = accuracy_score(Y_test,pred_values)
        print('Accuracy: {}'.format(acc))
        all_acc.append(acc)
        roc_auc = auc(fpr, tpr)
        print('AUC: {}'.format(roc_auc))
        precision, recall, _, _ = precision_recall_fscore_support(Y_test, pred_values, average=None)
        all_precisions.append(precision)
        all_recalls.append(recall)
        
        all_tpr.append(tpr)
        all_fpr.append(fpr)
        all_auc.append(roc_auc)

        print(f"The average accuracy for this run is {statistics.mean(all_acc)}")
        mean_precision = np.mean(all_precisions, axis=0)
        std_precision = np.std(all_precisions, axis=0)
        mean_recall = np.mean(all_recalls, axis=0)
        std_recall = np.std(all_recalls, axis=0)

        print(f"The mean precision is {mean_precision} with std of {std_precision}")
        print(f"The mean recall is {mean_recall} with std of {std_recall}")

    return all_acc, all_tpr, all_fpr, all_auc
