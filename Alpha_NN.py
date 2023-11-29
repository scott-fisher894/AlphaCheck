import torch
import numpy as np
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

class ChessEvaluationNN(nn.Module):
    def __init__(self):
        super(ChessEvaluationNN, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)  # Convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # Another convolutional layer
        self.fc1 = nn.Linear(32 * 8 * 8, 128)                    # Fully connected layer
        self.fc2 = nn.Linear(128, 1)                             # Output layer

    def forward(self, x):
        # Apply the layers
        x = F.relu(self.conv1(x))   # Activation function after convolution
        x = F.relu(self.conv2(x))   # Activation function after convolution
        x = x.view(-1, 32 * 8 * 8)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))     # Activation function after fully connected layer
        x = self.fc2(x)             # Output layer
        x = x.squeeze(-1)           # Remove extra dimension
        return x

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()  # Set the model to training mode

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()   # Clear the gradients

            # Forward pass: Compute predicted outputs by passing inputs to the model
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass: Compute gradient of the loss with respect to model parameters
            loss.backward()

            # Perform a single optimization step (parameter update)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_duration = time.time() - start_time
        print(f'Epoch {epoch + 1}/{epochs} completed in {epoch_duration:.2f} seconds - Loss: {epoch_loss:.4f}')

def evaluate_model(model, eval_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # No gradient calculation
        for inputs, labels in eval_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(eval_loader.dataset)

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(filepath, model):
    model.load_state_dict(torch.load(filepath))
    return model

if __name__ == "__main__":
    # Load dataset and labels from .npy files
    chess_dataset = np.load('chess_dataset.npy')
    chess_labels = np.load('chess_labels.npy')

    # Ensure the dataset is in the correct shape (N, C, H, W)
    chess_dataset = np.transpose(chess_dataset, (0, 3, 1, 2))

    # Convert numpy arrays to torch tensors
    chess_dataset = torch.tensor(chess_dataset, dtype=torch.float32)
    chess_labels = torch.tensor(chess_labels, dtype=torch.float32)

    # Create a TensorDataset
    dataset = TensorDataset(chess_dataset, chess_labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = ChessEvaluationNN()
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Call the training function
    train_model(model, train_loader, criterion, optimizer, epochs=3)

    # Optionally, save the model
    save_model(model, 'chess_model.pth')
"""
if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
"""