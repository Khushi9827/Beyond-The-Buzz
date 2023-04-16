# Importing the necessary libraries
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preprocess the dataset
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Normalize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define the neural network architecture


class MLP(nn.Module):
    def _init_(self):
        super(MLP, self)._init_()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Create an instance of the MLP model
model = MLP()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the neural network
num_epochs = 50
batch_size = 64

for epoch in range(num_epochs):
    # Set the model to train mode
    model.train()

    # Iterate over batches of data
    for i in range(0, len(X_train), batch_size):
        # Get the mini-batch of data
        x_batch = torch.FloatTensor(X_train[i:i+batch_size])
        y_batch = torch.FloatTensor(y_train[i:i+batch_size]).reshape(-1, 1)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x_batch)

        # Compute the loss
        loss = criterion(y_pred, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        y_val_pred = model(torch.FloatTensor(X_val))
        y_val_pred = (y_val_pred > 0.5).float().numpy().flatten()
        accuracy = np.mean(y_val_pred == y_val)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.3f}")

# Make predictions on the test set
X_test = scaler.transform(test_df.iloc[:, 1:].values)
y_test_pred = model(torch.FloatTensor(X_test)).detach().numpy().flatten()

# Save the predictions to a CSV file
submission_df = pd.DataFrame({'Id': test_df['Id'], 'Prediction': y_test_pred})
submission_df.to_csv('predictions.csv', index=False)
