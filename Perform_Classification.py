#Import libraries
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_fscore_support
from sklearn.model_selection import RepeatedKFold
import statistics

# perform the classification
def PerformClassification(X1, Y, csv_filename='Predictions.csv'):

    # initialize lists to store the metrics
    all_fpr = []
    all_tpr = []
    all_auc = []
    all_acc = []
    all_classification_reports = []
    all_precisions = []
    all_recalls = []

    
    all_predictions = pd.DataFrame(columns=['Fold', 'Actual_Labels', 'Predicted_Labels'])
    random_state = 42
    rkf = RepeatedKFold(n_splits=3, n_repeats=50, random_state=random_state)

    for i, (train_index, test_index) in enumerate(rkf.split(X1)):
        print(f"Fold {i}:")
        X_train, X_test = X1.iloc[train_index], X1.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state) 
        rfc.fit(X_train, Y_train)
        pred_values = rfc.predict(X_test)
        
        # write and store classification report
        report = classification_report(Y_test, pred_values, output_dict=True)
        all_classification_reports.append(report)
        print(classification_report(Y_test, pred_values))
        
        # get class probabilities for ROC curve
        probas_ = rfc.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(Y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        
        acc = accuracy_score(Y_test, pred_values)
        print(f'Accuracy: {acc:.4f}')
        print(f'AUC: {roc_auc:.4f}')
        
        precision, recall, _, _ = precision_recall_fscore_support(Y_test, pred_values, average=None)
        all_acc.append(acc)
        all_tpr.append(tpr)
        all_fpr.append(fpr)
        all_auc.append(roc_auc)
        all_precisions.append(precision)
        all_recalls.append(recall)

        # add the results to the dataframe        
        all_predictions = pd.concat([
            all_predictions,
            pd.DataFrame({
                'Fold': [i],
                'Actual_Labels': [Y_test.tolist()],
                'Predicted_Labels': [pred_values.tolist()]
            })
        ], ignore_index=True)
    
    avg_acc = np.mean(all_acc)
    avg_auc = np.mean(all_auc)
    
    print(f"\nFinal Results over {len(all_acc)} folds:")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average AUC: {avg_auc:.4f}")

    all_predictions.to_csv(csv_filename, index=False)
    return all_acc, all_tpr, all_fpr, all_auc, all_precisions, all_recalls

all_acc, all_tpr, all_fpr, all_auc, all_precisions, all_recalls = PerformClassification(X1, Y)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perform classification using selected features")
    parser.add_argument("-i", "--input", required=True, help="Path to the top features file")
    parser.add_argument("-d", "--dataset", required=True, help="Path to the dataset file")

    args = parser.parse_args()

    # Perform classification
    PerformClassification(args.features, args.dataset)
