#Advanced Machine Learning Analysis of Radiation Hardening in Reduced-Activation Ferritic/Martensitic Steels
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load Excel file
excel_file = "Data set.xlsx"
data = pd.read_excel(excel_file)

# Extract independent variables (X) and dependent variable (y)
X = data.iloc[:, :-1]  # All columns except the last column
y = data.iloc[:, -1]   # Last column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create GBDT model using the provided optimal hyperparameters
best_gbdt_model = GradientBoostingRegressor(learning_rate=0.05, max_depth=3, n_estimators=1000)
best_gbdt_model.fit(X_train_scaled, y_train)
gbdt_predictions = best_gbdt_model.predict(X_test_scaled)

# Calculate evaluation metrics
gbdt_mse = mean_squared_error(y_test, gbdt_predictions)
gbdt_rmse = np.sqrt(gbdt_mse)
gbdt_mae = mean_absolute_error(y_test, gbdt_predictions)
gbdt_r2 = r2_score(y_test, gbdt_predictions)

print(f"GBDT - MSE: {gbdt_mse:.2f}, RMSE: {gbdt_rmse:.2f}, MAE: {gbdt_mae:.2f}, R^2: {gbdt_r2:.2f}")

# Plot scatter plot of actual vs predicted values
plt.scatter(y_test, gbdt_predictions, label=f'GBDT - MSE: {gbdt_mse:.2f}, RMSE: {gbdt_rmse:.2f}, MAE: {gbdt_mae:.2f}, R^2: {gbdt_r2:.2f}', alpha=0.5)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.title("Scatter Plot of Actual vs Predicted Values")

# Save the plot as an image
plt.savefig("scatter_plot_gbdt.png")

# Display the plot
plt.show()

# Create a DataFrame to store evaluation metrics
metrics_data = {
    'Model': ['GBDT'],
    'MSE': [gbdt_mse],
    'RMSE': [gbdt_rmse],
    'MAE': [gbdt_mae],
    'R^2': [gbdt_r2]
}
metrics_df = pd.DataFrame(metrics_data)

# Save evaluation metrics to an Excel file
metrics_df.to_excel("Radiation_Hardening_GBDT_metrics.xlsx", index=False)

# Create a DataFrame to store scatter plot data
scatter_data = {
    'Actual Values': y_test,
    'GBDT Predictions': gbdt_predictions,
}
scatter_df = pd.DataFrame(scatter_data)

# Save scatter plot data to an Excel file
scatter_df.to_excel("Radiation_Hardening_GBDT_scatter.xlsx", index=False)
