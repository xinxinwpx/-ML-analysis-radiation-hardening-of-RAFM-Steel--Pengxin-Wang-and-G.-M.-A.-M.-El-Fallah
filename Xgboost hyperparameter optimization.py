#Advanced Machine Learning Analysis of Radiation Hardening in Reduced-Activation Ferritic/Martensitic Steels
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
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
max_depth_range = np.arange(5, 21, 1)
eta_range = np.linspace(0.00035, 0.0015, 5)

# Initialize result storage
results_train = []
results_test = []

# Iterate over parameter combinations
for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        for eta in eta_range:
            # Initialize and train the model
            xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=eta, random_state=42)
            xgb.fit(X_train, y_train)

            # Predict and evaluate on the training set
            y_train_pred = xgb.predict(X_train)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            pcc_train = np.corrcoef(y_train, y_train_pred)[0, 1]
            r2_train = r2_score(y_train, y_train_pred)

            # Predict and evaluate on the testing set
            y_test_pred = xgb.predict(X_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            pcc_test = np.corrcoef(y_test, y_test_pred)[0, 1]
            r2_test = r2_score(y_test, y_test_pred)

            # Store results
            results_train.append((n_estimators, max_depth, eta, rmse_train, pcc_train, r2_train))
            results_test.append((n_estimators, max_depth, eta, rmse_test, pcc_test, r2_test))

# Convert results to DataFrames
results_train_df = pd.DataFrame(results_train, columns=['n_estimators', 'max_depth', 'eta', 'rmse', 'pcc', 'r2'])
results_test_df = pd.DataFrame(results_test, columns=['n_estimators', 'max_depth', 'eta', 'rmse', 'pcc', 'r2'])

# Save results to Excel files
results_train_df.to_excel('XGB_train_results.xlsx', index=False)
results_test_df.to_excel('XGB_test_results.xlsx', index=False)
