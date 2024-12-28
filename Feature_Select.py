#Import libraries
import warnings
warnings.filterwarnings('ignore') 
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import RocCurveDisplay
from scipy import stats
import math
from sklearn import metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from itertools import product
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statistics
from random import shuffle
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_recall_curve, confusion_matrix, classification_report
from matplotlib.pyplot import figure  
from matplotlib.colors import Normalize

# adjust here with the columns you want to be read as features and the column which is the target variable
def ExtractColumns(df):
    X = df.iloc[:,1:49].round(2)
    Y = df.iloc[:, 49].round(0)
    return X, Y
    
def SelectFeatures(X, Y):
    # accuracy scores will be be saved here
    all_acc = []
    # most important features for every iteration will be saved here
    MIF = []
    # r2 scores will be saved here
    all_r2 = []

    random_state = 42
    rkf = RepeatedKFold(n_splits=3, n_repeats=50, random_state=random_state)
    rkf.get_n_splits(X, Y)

    for train, test in rkf.split(X):
        X_train, X_test = X.iloc[train, :], X.iloc[test, :]
        Y_train, Y_test = Y.iloc[train], Y.iloc[test]

        rf = RandomForestRegressor(random_state=42, n_estimators=150, n_jobs=-1)
        rf.fit(X_train, Y_train)
        pred_values = rf.predict(X_test)

        f_i = list(zip(X.columns, rf.feature_importances_))
        f_i.sort(key=lambda x: -x[1])
        MIF.append(f_i)

        # calculate R² score 
        r2 = r2_score(Y_test, pred_values)
        print(f'R² Score : {r2:.2f}')
        all_r2.append(r2)
    return all_r2, MIF    

def GetTopFeatures(MIF, top_n=3, common_n=5, csv_filename='top_features.csv'):
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

    figure(figsize=(15, 7), dpi=80)

    # unpack the elements and frequencies
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

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature', 'Importance'])
        for feature, importance in most_common:
            writer.writerow([feature, importance])
    
    return most_common    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features and save top features")
    parser.add_argument("-i", "--input", required=True, help="Path to the input file (CSV or XLSX)")
    parser.add_argument("-o", "--output", default="top_features.csv", help="Path to save top features (CSV)")

    args = parser.parse_args()
    if args.input.endswith(".xlsx"):
        df = pd.read_excel(args.input)
    elif args.input.endswith(".csv"):
        df = pd.read_csv(args.input)

    # perform feature extraction and selection
    X, Y = ExtractColumns(df)
    all_acc, MIF = SelectFeatures(X, Y)
    top_Features = GetTopFeatures(MIF, top_n=3, common_n=5)
    feature_df = pd.DataFrame(top_Features, columns=["Feature"])
    feature_df.to_csv(args.output, index=False)
    print(f"Top features saved to {args.output}")
