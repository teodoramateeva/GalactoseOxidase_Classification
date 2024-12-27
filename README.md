#### Quick start 
Prerequisites (it is good practice to create a dedicated python venv, although not a must):

```bash
pip install \
  pandas \
  seaborn \
  scikit-learn \
  matplotlib \
  scipy \
  numpy \
  itertools \
  statistics
```

You will also need to import all required estimators from scikit-learn, based on which model you're interested in, (RandomForestClassifier, GradientBoostingClassifier), as well as the other python libraries related to calculating performance metrics.


#### **Description**

This module reads the extracted, averaged features from 3 replicas of Transition State (TS) Molecualar Dynamics (MD) simulations and performs classification. It uses the features from MD to classify variants of Galactose Oxidase, based on whether the variant is predicted to slow down or enhance the rate for the catalytic conversion of the alcohol substrate. Since the Random Forest and other ensemble algorithms are a type of intrinsic algorithms which perform automatic feature selection during the training of the model, you can use all your features and still get the same accuracy as you would if you pre-select the features. However, since the point of this study is to engineer an enzyme, knowing which features the model learns from, is important for the process. 

#### **Extracted features file**

All of the features extracted from the MD simulations need to be organised in a .csv or .xslx file in a format similar to the example_features.xslx file where the first column contains the names of the Galactose Oxidase Variants. All subsequent columns are different features, in this example, interatomic distances between two active site atoms. The final column should be your target variable, in this example it is binary (0 - rate is similar or better than the WT enzyme; 1 - rate is slower than WT enzyme).

#### Clone the repository to your local machine:
```bash
git https://github.com/teodoramateeva/GalactoseOxidase_Classification
```
#### Make sure you are located in the correct folder: 

```bash
cd GalactoseOxidase_Classification
```

#### To get the most important features and then subsequently use only those for classification, run:
Depending on your case, you might want to skip this step.

```bash
python Feature_extraction.py -i example_features/example_features.xlsx -o top_features.csv

```
This will save a .csv file with the best features in your current directory.

#### Then, run from the command line: 

```bash
python Perform_Classification.py -i top_features.csv -d example_features/example_features.xlsx

```
This script will use features to fit a Random Forest and give you the accuracy of prediction. You can further modify the .py scripts, depending on what metrics and predictions you want saved. The accuracy should remain the same if you fit all of the features, but you should be aware of the number of features you're using or you can cause overfitting.
