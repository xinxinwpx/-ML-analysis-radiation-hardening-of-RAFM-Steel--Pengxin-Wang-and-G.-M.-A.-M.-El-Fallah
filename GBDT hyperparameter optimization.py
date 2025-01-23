#Advanced Machine Learning Analysis of Radiation Hardening in Reduced-Activation Ferritic/Martensitic Steels
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')

# Load data
data = pd.read_csv('Data set.csv')

# Separate independent variables (X) and dependent variable (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter ranges
n_estimators_range = np.arange(5000, 20001, 5000)
learning_rate_range = np.linspace(0.0001, 0.001, 5)
max_depth_range = np.arange(5, 12)

# Initialize storage for results
results_train = []
results_test = []

# Iterate through combinations of parameters
for n_estimators in n_estimators_range:
    for learning_rate in learning_rate_range:
        for max_depth in max_depth_range:
            # Initialize and train the model
            gboost = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
            gboost.fit(X_train, y_train)

            # Predictions and evaluation on training set
            y_train_pred = gboost.predict(X_train)
            rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
            r2_train = r2_score(y_train, y_train_pred)
            pcc_train = np.corrcoef(y_train, y_train_pred)[0, 1]

            # Predictions and evaluation on testing set
            y_test_pred = gboost.predict(X_test)
            rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
            r2_test = r2_score(y_test, y_test_pred)
            pcc_test = np.corrcoef(y_test, y_test_pred)[0, 1]

            # Store results
            results_train.append((n_estimators, learning_rate, max_depth, rmse_train, r2_train, pcc_train))
            results_test.append((n_estimators, learning_rate, max_depth, rmse_test, r2_test, pcc_test))

# Convert results to DataFrame
results_train_df = pd.DataFrame(results_train, columns=['n_estimators', 'learning_rate', 'max_depth', 'rmse', 'r2', 'pcc'])
results_test_df = pd.DataFrame(results_test, columns=['n_estimators', 'learning_rate', 'max_depth', 'rmse', 'r2', 'pcc'])

# Save results to Excel files
results_train_df.to_excel('GBDT_train_results.xlsx', index=False)
results_test_df.to_excel('GBDT_test_results.xlsx', index=False)
