#### Quick start 
Prerequisites (it is advised to create a dedicated python venv or conda env)

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

You will also need to import all the estimators from scikit-learn, as well as the other python libraries related to calculating performance metrics.


#### **Description**

This module reads the extracted, averaged features from several replicas of Molecualar Dynamics (MD) simulations and selects the top features. It then uses those features to fit a Random Forest Model to classify variants of Galactose Oxidase, based on whether the variant is predicted to slow down or enhance the rate for the catalytic conversion of the alcohol substrate.

#### **Extracted features file**

All of the features extracted from the MD simulations need to be organised in a .csv file in a format similar to the example_features.csv file where the first column contains the names of the Galactose Oxidase Variants. All subsequent columns are different features, for example, interatomic distances between two active site atoms. The final column should be your target variable.


#### Run from the command line: 

```bash
python __main__.py -i example_features.csv 
```
This will give you the option to save a plot of the most important features in your dataset. Once you have saved the .png file or you have opted out, you will get a printout of the most important features. The features will also get saved to a .csv file. 

#### Then, run from the command line: 

```bash
python Perform_Classification.py -i example_features.csv 
```
This script will use the best features to fit a Random Forest and give you the averaged accuracy of prediction for 100 test folds.
