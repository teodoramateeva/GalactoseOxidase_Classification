#Import libraries
import warnings
warnings.filterwarnings('ignore') 
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict, Counter
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
from matplotlib.pyplot import figure
from sklearn.metrics import accuracy_score     
from sklearn.feature_selection import SelectFromModel
import math
from scipy.stats.stats import pearsonr
from matplotlib.colors import Normalize

#Adjust here with the columns you want to be read as features and the column which is the Target variable
def ExtractColumns(df):
    X = df.iloc[:,1:49]
    Y = df.iloc[:,[49]]

    print("Function 1 executed: Extracted columns 'col1' as X and 'col2' as Y")
    return all_acc, MIF

def SelectFeatures(X, Y):
    # Accuracy scores to be saved here
    all_acc = []
    # Most important features for every iteration to be saved here
    MIF = []

    # Define a random state to split the data into 100 unique testing and training folds
    random_state = 12883823
    rkf = RepeatedKFold(n_splits=2, n_repeats=100, random_state=random_state)
    rkf.get_n_splits(X, Y)

    for train, test in rkf.split(X):
        X_train, X_test = X.iloc[train, :], X.iloc[test, :]
        Y_train, Y_test = Y.iloc[train], Y.iloc[test]

        rf = RandomForestRegressor(random_state=12883823, n_jobs=-1)
        rf.fit(X_train, Y_train)
        f_i = list(zip(X.columns, rf.feature_importances_))
        f_i.sort(key=lambda x: -x[1])
        MIF.append(f_i)

        pred_values = rf.predict(X_test)
        acc = accuracy_score(pred_values.round(), Y_test)  # Assuming Y_test is binary
        print('Accuracy : {}'.format(acc))
        all_acc.append(acc)

    return all_acc, MIF
    

def GetTopFeatures(MIF, top_n=3, common_n=5):
    """
    Extract the top N features from each iteration in MIF and return the most common features.
    Returns:
    list: Most common features across all iterations.
    """
    top_n_items = []
    strings_only = []

    for data_list in MIF:
        sorted_list = sorted(data_list, key=lambda x: x[1], reverse=True)
        top_n_features = sorted_list[:top_n]
        top_n_items.append(top_n_features)

    for item_list in top_n_items:
        strings = [item[0] for item in item_list]
        strings_only.extend(strings)

    counter = Counter(strings_only)
    most_common = counter.most_common(common_n)


    figure(figsize=(9, 6.5), dpi=80)

    # Unpack the elements and frequencies
    elements, frequencies = zip(*most_common)
    pastel_palette = sns.color_palette("Pastel1", len(elements))
    
    # Plot the bar chart with pastel colors
    plt.bar(elements, frequencies, color=pastel_palette, edgecolor=".01",)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Feature', fontsize=16)
    plt.ylabel('Importance', fontsize=16)
    #plt.title('Most important features',fontsize=16)
    plt.show()
    
    return most_common    

