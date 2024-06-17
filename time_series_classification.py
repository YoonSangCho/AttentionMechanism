import numpy as np
import pandas as pd


'''
Part 1: Generating Simulation Time-Series Data
Here's a Python code snippet to generate synthetic time-series data for classification problems:
'''

# The generate_time_series_data function creates synthetic time-series data with specified samples, timestamps, features, and classes.

def generate_time_series_data(n_samples=1000, n_timestamps=50, n_features=3, n_classes=2, random_state=42):
    np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_timestamps, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    return X, y

# Example usage:
X, y = generate_time_series_data()
print("Feature Shape:", X.shape)  # Should be (1000, 50, 3)
print("Labels Shape:", y.shape)   # Should be (1000,)

'''
Part 2: Attention Mechanism to Highlight Causal Time Indices and Axes
We will create an attention mechanism and a simple neural network to predict the class of the time-series data.
We also highlight the important time indices and features. 
The attention weights will help us identify which parts of the input are most influential in the classification decision.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(input_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        attn_weights = torch.tanh(self.attention(x))
        attn_weights = self.context_vector(attn_weights).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        weighted_sum = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        
        return weighted_sum, attn_weights

class TimeSeriesClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TimeSeriesClassifier, self).__init__()
        self.attention = Attention(input_dim, hidden_dim)
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        attn_output, attn_weights = self.attention(x)
        output = self.classifier(attn_output)
        return output, attn_weights

# Parameters
n_samples = 1000
n_timestamps = 50
n_features = 3
n_classes = 2
hidden_dim = 64
learning_rate = 0.001
num_epochs = 10

# Generate data
X, y = generate_time_series_data(n_samples, n_timestamps, n_features, n_classes)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Model, Loss, Optimizer
model = TimeSeriesClassifier(input_dim=n_features, hidden_dim=hidden_dim, output_dim=n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs, attn_weights = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Visualize Attention Weights for a sample
sample_idx = 0
model.eval()
with torch.no_grad():
    _, attn_weights = model(X[sample_idx].unsqueeze(0))

attn_weights = attn_weights.squeeze().numpy()
plt.plot(attn_weights)
plt.title('Attention Weights')
plt.xlabel('Time Index')
plt.ylabel('Attention Weight')
plt.show()
