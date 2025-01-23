#Advanced Machine Learning Analysis of Radiation Hardening in Reduced-Activation Ferritic/Martensitic Steels
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('TkAgg')

# Define ResMLP class
class ResMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ResMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.matching_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.matching_layer(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out += residual
        out = self.fc3(out)
        return out

# Load the Excel file
excel_file = "Data set.xlsx"
data = pd.read_excel(excel_file)

# Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert to Tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Define the ResMLP model with corrected hyperparameters
input_size = X_train.shape[1]
output_size = 1
hidden_size = 512  # Corrected hyperparameter
lr = 0.0005  # Corrected hyperparameter
batch_size = 64  # Corrected hyperparameter
num_epochs = 1000  # Corrected hyperparameter

# Create ResMLP model
model = ResMLP(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training the ResMLP model
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    train_predictions = model(X_train).numpy()
    test_predictions = model(X_test).numpy()

# Calculate evaluation metrics (training set)
train_mse = mean_squared_error(y_train.numpy(), train_predictions)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train.numpy(), train_predictions)
train_r2 = r2_score(y_train.numpy(), train_predictions)
train_pcc, _ = pearsonr(y_train.numpy().flatten(), train_predictions.flatten())  # Pearson Correlation Coefficient

# Calculate evaluation metrics (testing set)
test_mse = mean_squared_error(y_test.numpy(), test_predictions)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test.numpy(), test_predictions)
test_r2 = r2_score(y_test.numpy(), test_predictions)
test_pcc, _ = pearsonr(y_test.numpy().flatten(), test_predictions.flatten())  # Pearson Correlation Coefficient

# Print results
print(f"ResMLP Train - MSE: {train_mse:.2f}, RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R^2: {train_r2:.2f}, PCC: {train_pcc:.2f}")
print(f"ResMLP Test - MSE: {test_mse:.2f}, RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R^2: {test_r2:.2f}, PCC: {test_pcc:.2f}")

# Scatter plot of actual vs predicted values
plt.scatter(y_test.numpy(), test_predictions, label=f'ResMLP - MSE: {test_mse:.2f}, RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R^2: {test_r2:.2f}', alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.title("Scatter Plot of Actual vs Predicted Values for ResMLP")

# Save plot
plt.savefig("scatter_plot_resmlp.png")

# Show plot
plt.show()

# Save metrics to Excel
metrics_data = {
    'Model': ['ResMLP'],
    'Train_MSE': [train_mse],
    'Train_RMSE': [train_rmse],
    'Train_MAE': [train_mae],
    'Train_R^2': [train_r2],
    'Train_PCC': [train_pcc],
    'Test_MSE': [test_mse],
    'Test_RMSE': [test_rmse],
    'Test_MAE': [test_mae],
    'Test_R^2': [test_r2],
    'Test_PCC': [test_pcc]
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_excel("ResMLP1_metrics.xlsx", index=False)

# Save scatter data to Excel
scatter_data = {
    'Actual Values': y_test.numpy().flatten(),
    'ResMLP Predictions': test_predictions.flatten(),
}
scatter_df = pd.DataFrame(scatter_data)
scatter_df.to_excel("ResMLP1_scatter.xlsx", index=False)
