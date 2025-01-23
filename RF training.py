#Advanced Machine Learning Analysis of Radiation Hardening in Reduced-Activation Ferritic/Martensitic Steels
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Read the Excel file
excel_file = "Data set.xlsx"
data = pd.read_excel(excel_file)

# Extract independent variables (X) and dependent variable (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Random Forest model with best hyperparameters
best_rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=30,
    min_samples_split=5,
    max_features=10,
    random_state=42
)

# Train the model
best_rf_model.fit(X_train_scaled, y_train)
rf_predictions = best_rf_model.predict(X_test_scaled)

# Calculate evaluation metrics
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print(f"Random Forest - MSE: {rf_mse:.2f}, RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}, R^2: {rf_r2:.2f}")

# Plot a scatter plot of actual vs. predicted values
plt.scatter(y_test, rf_predictions, label=f'RF - MSE: {rf_mse:.2f}, RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}, R^2: {rf_r2:.2f}', alpha=0.5)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.title("Scatter Plot of Actual vs Predicted Values")

# Save the scatter plot image
plt.savefig("scatter_plot_rf.png")

# Display the scatter plot
plt.show()

# Create a DataFrame to store evaluation metrics
metrics_data = {
    'Model': ['Random Forest'],
    'MSE': [rf_mse],
    'RMSE': [rf_rmse],
    'MAE': [rf_mae],
    'R^2': [rf_r2]
}
metrics_df = pd.DataFrame(metrics_data)

# Save evaluation metrics to an Excel file
metrics_df.to_excel("RF_metrics.xlsx", index=False)

# Create a DataFrame to store scatter plot data
scatter_data = {
    'Actual Values': y_test,
    'RF Predictions': rf_predictions,
}
scatter_df = pd.DataFrame(scatter_data)

# Save scatter plot data to an Excel file
scatter_df.to_excel("RF_scatter.xlsx", index=False)
