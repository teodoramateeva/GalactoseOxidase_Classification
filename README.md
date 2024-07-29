#### **Description**

This module reads the extracted, averaged features from several replicas of Molecualar Dynamics (MD) simulations and selects the top features. It then uses those features to fit a Random Forest Model to classify variants of Galactose Oxidase, based on whether the variant is predicted to slow down or enhance the rate for the catalytic conversion of the alcohol substrate.


#### **Run from the command line: **

```bash
python __main__.py -i example_features.csv 
```

