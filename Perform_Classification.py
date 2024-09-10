#Import libraries
import warnings
import argparse
warnings.filterwarnings('ignore') 
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from scipy import stats
from sklearn import metrics
import math
import statistics
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure   
from itertools import product
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from random import shuffle
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_recall_curve, confusion_matrix, classification_report, precision_recall_fscore_support
from scipy.stats.stats import pearsonr
from matplotlib.colors import Normalize



# set up arg parser
parser = argparse.ArgumentParser(description='myscript')
parser.add_argument('-i', "--input", required=True, help="Path to the input file")
args = parser.parse_args()
df = pd.read_csv(args.input)

#Read the top features
top_features_df = pd.read_csv('top_features.csv')
feature_indices = top_features_df['Feature'].tolist()

X1 = df.iloc[:, feature_indices]
Y = df.iloc[:, 49]  # Update as needed based on your target column

# Perorm the Classification

def PerformClassification(X1, Y, csv_filename='Predictions.csv'):
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

        # Get class probabilities for ROC curve
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
    return all_acc, all_tpr, all_fpr, all_auc, all_precisions, all_recalls

PerformClassification(X1, Y)    
