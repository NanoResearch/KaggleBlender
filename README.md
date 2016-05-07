# KaggleBlender

The FeatureEnigeer takes the train.csv and test.csv from the BNP Paribas Kaggle challenge (too big for git) and converts the categorical features and extracts additional features, which are then saved in a numpy file.
The SingleModel python file runs a single sklearn model on the data from the FeatureEnigeer script.
The ModelBlender runs multiple models on the data and blends the data using logistic regression.