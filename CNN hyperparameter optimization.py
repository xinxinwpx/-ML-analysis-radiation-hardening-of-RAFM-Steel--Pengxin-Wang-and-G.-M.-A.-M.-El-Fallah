#Advanced Machine Learning Analysis of Radiation Hardening in Reduced-Activation Ferritic/Martensitic Steels
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import itertools


# Define a CNN model for regression
class CNNRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNRegression, self).__init__()
        # Define a convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Define a fully connected layer
        self.fc1 = nn.Linear(64 * input_size, 128)  # Flattened size after convolutions
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for 1D ConvNet
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Flatten for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


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


# Training function
def train_model(lr, batch_size, num_epochs):
    input_size = X_train.shape[1]
    output_size = 1
    model = CNNRegression(input_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train).numpy()
        test_predictions = model(X_test).numpy()

    # Calculate metrics (training set)
    train_mse = mean_squared_error(y_train.numpy(), train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train.numpy(), train_predictions)
    train_pcc = np.corrcoef(y_train.numpy().flatten(), train_predictions.flatten())[0, 1]

    # Calculate metrics (testing set)
    test_mse = mean_squared_error(y_test.numpy(), test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test.numpy(), test_predictions)
    test_pcc = np.corrcoef(y_test.numpy().flatten(), test_predictions.flatten())[0, 1]

    return (train_mse, train_rmse, train_r2, train_pcc), (test_mse, test_rmse, test_r2, test_pcc)


# Define parameter grid
param_grid = {
    'lr': [0.01, 0.001, 0.0005, 0.0001, 0.00001],
    'batch_size': [16, 32, 64, 128],
    'num_epochs': [500, 1000, 1500, 2000]
}

# Manual grid search
best_params = None
best_test_mse = float('inf')

# Store all results
results_list = []

# Get all combinations of hyperparameters
param_combinations = list(itertools.product(param_grid['lr'], param_grid['batch_size'], param_grid['num_epochs']))

# Train for all combinations
for lr, batch_size, num_epochs in param_combinations:
    print(f"Training with lr={lr}, batch_size={batch_size}, num_epochs={num_epochs}")
    train_metrics, test_metrics = train_model(lr, batch_size, num_epochs)

    train_mse, train_rmse, train_r2, train_pcc = train_metrics
    test_mse, test_rmse, test_r2, test_pcc = test_metrics

    print(f"Train - MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, R^2: {train_r2:.4f}, PCC: {train_pcc:.4f}")
    print(f"Test - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R^2: {test_r2:.4f}, PCC: {test_pcc:.4f}")

    # Save results
    results_list.append({
        'lr': lr,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'Train_MSE': train_mse,
        'Train_RMSE': train_rmse,
        'Train_R^2': train_r2,
        'Train_PCC': train_pcc,
        'Test_MSE': test_mse,
        'Test_RMSE': test_rmse,
        'Test_R^2': test_r2,
        'Test_PCC': test_pcc
    })

    # Update best model
    if test_mse < best_test_mse:
        best_test_mse = test_mse
        best_params = {
            'lr': lr,
            'batch_size': batch_size,
            'num_epochs': num_epochs
        }

# Output best parameters
print(f"Best parameters: {best_params}")
print(f"Best Test MSE: {best_test_mse:.4f}")

# Save results
results_df = pd.DataFrame(results_list)
results_df.to_excel("cnn_model_training_results.xlsx", index=False)
