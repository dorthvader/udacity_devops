# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This project loads data, performs exploratory data analysis, creates features, and trains random forest and logistic regression models. The goal is to predict customer churn.

The default state for running churn_library.py is intended for the bank_data.csv file and predict customer churn project. However, the functions and file itself could be used for other datasets as well. Some modifications might need to be made for other uses.

## Running Files

### churn_library.py

The only required argument is --data-path. This path should be from the root of your project. Here is an example if the file is in a folder named "data."

        python3 src/churn_library.py --data-path="data/bank_data.csv"

Other arguments can be passed if using the file for other files with similar data. The defaults are set for this project specifically, so no need to pass anything other than  --data-path for the bank dataset.

###test_churn_library.py

This file uses pytest, so from the root of the project use the command !pytest
