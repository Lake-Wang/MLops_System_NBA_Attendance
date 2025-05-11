# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# Model utilities
# Define the MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_size, dropout=0.5):
        super(MLP, self).__init__()
        
        # Stitch together layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_layer_size[0])) # from input_size -> hidden layer size
        layers.append(nn.ReLU()) # ReLU activation
        layers.append(nn.Dropout(dropout)) # Regularize with dropout
        
        # For rest of hidden layer transformations, repeat the pattern
        for i in range(len(hidden_layer_size)-1):
            layers.append(nn.Linear(hidden_layer_size[i], hidden_layer_size[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Final layer (get to 1 node for regression task)
        layers.append(nn.Linear(hidden_layer_size[-1], 1))
        
        # Stitch together the layers to define the MLP
        self.model = nn.Sequential(*layers)
    
    # Feed forward
    def forward(self, x):
        # Normalize and FF
        x = F.normalize(x)
        return self.model(x)
    

# Training loop
def train(model, dataloader, criterion, optimizer, device="cpu"):
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Apply gradient clipping for stable training
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    
    return epoch_loss

# Evaluate
def validate(model, dataloader, criterion, device="cpu"):
    model.eval()
    running_loss = 0.0

    # For eval
    model_outputs = []
    model_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Grab results in list
            model_outputs.append(outputs)
            model_labels.append(labels)

    epoch_loss = running_loss / len(dataloader)

    # Store results
    all_model_outputs = torch.cat(model_outputs, dim=0)
    all_labels = torch.cat(model_labels, dim=0)

    # Compute R^2 and RMSE
    ss_residuals = torch.sum((all_labels - all_model_outputs) ** 2)
    mean_label = torch.mean(all_labels)
    ss_total = torch.sum((all_labels - mean_label) ** 2)
    r2 = 1 - (ss_residuals / ss_total)
    rmse = np.sqrt(epoch_loss)
    
    return r2, rmse, epoch_loss