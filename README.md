# Beyond-The-Buzz
## Description
The Multi-Layer Perceptron (MLP) is a feedforward neural network that consists of multiple layers of nodes. In this model, we have used a 3-layer MLP with fully connected layers.

The input layer of the MLP consists of the input features. The hidden layers are the intermediate layers that process the input features and produce a representation that can be used to make predictions. The output layer produces the final output of the MLP.

In each layer of the MLP, the nodes perform a linear transformation on the input followed by a non-linear activation function. In our model, we have used ReLU activation functions in the first two layers and sigmoid activation function in the last layer.

The ReLU activation function is a simple non-linear function that returns the input if it is positive and zero otherwise. The sigmoid activation function is a smooth non-linear function that maps the input to a probability between 0 and 1.

During training, the weights of the MLP are updated using gradient descent to minimize a loss function. In our model, we have used binary cross-entropy loss as our loss function, which is commonly used for binary classification problems.
We have used the Adam optimizer for gradient descent, which is an adaptive learning rate optimization algorithm that computes individual adaptive learning rates for different parameters based on the first and second moments of the gradients.

We have also used standardization to preprocess the input features, which involves scaling the features to have zero mean and unit variance. This helps to improve the performance of the MLP by making the optimization process more efficient.

During training, we have used mini-batch stochastic gradient descent, which involves processing the input data in small batches rather than all at once. This helps to reduce the memory requirements and improve the convergence of the optimization algorithm.

In summary, our MLP classifier uses fully connected layers with ReLU and sigmoid activation functions, binary cross-entropy loss, Adam optimizer, and standardization for preprocessing.

## Mathematical Calculation

The input layer of the MLP takes a vector x of size n, where n is the number of input features. The first hidden layer consists of m nodes, and the weight matrix W1 has size (n x m), where each row represents the weights for a single node in the first hidden layer. The bias vector b1 has size m, where each element represents the bias for a single node in the first hidden layer. The output of the first hidden layer is given by:

h1 = ReLU(x * W1 + b1)
where * represents matrix multiplication and ReLU is the rectified linear unit activation function, defined as:

ReLU(z) = max(0, z)

The second hidden layer consists of p nodes, and the weight matrix W2 has size (m x p), where each row represents the weights for a single node in the second hidden layer. The bias vector b2 has size p, where each element represents the bias for a single node in the second hidden layer. The output of the second hidden layer is given by:

h2 = ReLU(h1 * W2 + b2)

The output layer consists of a single node, and the weight vector W3 has size (p x 1), where each element represents the weight for the output node. The bias term b3 is a scalar that represents the bias for the output node. The final output of the MLP is given by:

y = sigmoid(h2 * W3 + b3)

where sigmoid is the sigmoid activation function, defined as:

sigmoid(z) = 1 / (1 + exp(-z))

During training, the weights and biases of the MLP are optimized to minimize the binary cross-entropy loss function, which is defined as:

L(y, y_hat) = - (y * log(y_hat) + (1 - y) * log(1 - y_hat))

where y is the true label (0 or 1) and y_hat is the predicted probability (output of the MLP).

The Adam optimizer is used to update the weights and biases of the MLP during training. The update rule for the weights and biases at iteration t is given by:

W_t+1 = W_t - alpha * m_t / (sqrt(v_t) + eps)
b_t+1 = b_t - alpha * m_t / (sqrt(v_t) + eps)
where alpha is the learning rate, m_t and v_t are the first and second moments of the gradients, and eps is a small constant to avoid division by zero.

In summary, the MLP classifier involves matrix multiplication, non-linear activation functions, binary cross-entropy loss, and the Adam optimizer for weight updates.

## Code Explanation
Here is the first block that imports the necessary libraries and loads the dataset. For this task, I will be using PyTorch as my preferred library for building the neural network classifier.
```
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
```
Next, I will preprocess the dataset by splitting it into input and target variables. I will also split the data into training and validation sets using the train_test_split() function from scikit-learn. Additionally, I will normalize the input features using the StandardScaler() function to improve the training of the neural network.
```
# Preprocess the dataset
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```
In the next block of code, I will define the neural network architecture using PyTorch's nn.Module class. For this task, I will use a multi-layer perceptron (MLP) with three fully connected layers. I will use the ReLU activation function for the hidden layers and the sigmoid activation function for the output layer, as this is a binary classification problem.

```
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
```
In the next block, I will define the loss function and the optimizer for training the neural network. For this task, I will use the binary cross-entropy loss function and the Adam optimizer.

```
# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
Next, I will train the neural network on the training set using mini-batch gradient descent. I will iterate over the training data for a fixed number of epochs, and for each epoch, I will randomly sample mini-batches of data and update the model parameters using backpropagation.

```
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
        optimizer.zero_grad
```
