# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import time
import os
from utils import *

### Imports for MLFlow
import mlflow
import mlflow.pytorch
import subprocess

# Set ml-flow experiment on model 2
mlflow.set_experiment("model2")

# Configuration
config = {
    "total_epochs": 1000,
    "patience": 100,
    "batch_size": 32,
    "lr": 1e-4,
    "hidden_layer_size": [256, 128, 64, 32],
    "dropout": 0.15
}

torch.manual_seed(42)

# Import data from previous model
result_df = pd.read_csv('model1_out.csv')

base_data_dir = os.getenv("NBA_DATA_DIR", "nba_data")

X_train = pd.read_csv(os.path.join(base_data_dir, 'train/X_train_model2.csv'))
X_train = X_train.merge(result_df, on='gameId', how='inner')
X_test = pd.read_csv(os.path.join(base_data_dir, 'train/X_test_model2.csv'))
X_test = X_test.merge(result_df, on='gameId', how='inner')
Y_train = pd.read_csv(os.path.join(base_data_dir, 'train/Y_train_model2.csv'))
Y_test = pd.read_csv(os.path.join(base_data_dir, 'train/Y_test_model2.csv'))

X_save_cols = X_train.columns

# Read data
# Convert to tensors, pass to dataloader
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
Y_train = torch.tensor(Y_train.values, dtype=torch.float32)
Y_test = torch.tensor(Y_test.values, dtype=torch.float32)

train_data = TensorDataset(X_train, Y_train)
test_data = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

# Prepare the training loop
model = MLP(X_train.shape[1], config['hidden_layer_size'], config['dropout'])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),
                       lr=config['lr'])

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['total_epochs'] / 5,
    eta_min=config['lr'] / 5
)


# Get model param count
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

### Before we start training - start an MLFlow run
try: 
    mlflow.end_run() # end pre-existing run, if there was one
except:
    pass
finally:
    mlflow.start_run(run_name="Model2", log_system_metrics=True) # Start MLFlow run
    # automatically log GPU and CPU metrics
    # Note: to automatically log AMD GPU metrics, you need to have installed pyrsmi
    # Note: to automatically log NVIDIA GPU metrics, you need to have installed pynvml

    # Let's get the output of rocm-info or nvidia-smi as a string...
gpu_info = next(
    (subprocess.run(cmd, capture_output=True, text=True).stdout for cmd in ["nvidia-smi", "rocm-smi"] if subprocess.run(f"command -v {cmd}", shell=True, capture_output=True).returncode == 0),
    "No GPU found."
)
# ... and send it to MLFlow as a text file
mlflow.log_text(gpu_info, "gpu-info.txt")


# Log hyperparameters - the things that we *set* in our experiment configuration
mlflow.log_params(config)

best_test_loss = float('inf')
patience_counter = 0


for epoch in range(config["total_epochs"]):
    # Track time
    epoch_start_time = time.time()
    train_loss = train(model, train_loader, criterion, optimizer, device)
    _, _, test_loss = validate(model, test_loader, criterion, device)
    scheduler.step()
    epoch_time = time.time() - epoch_start_time
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    

    mlflow.log_metrics(
        {"epoch_time": epoch_time,
         "train_loss": train_loss,
         "test_loss": test_loss,
         "trainable_params": trainable_params,
         }, step=epoch)
    
    # Check for improvement in Test loss
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        torch.save(model, "attendance_model.pth")
        print("   Test loss improved. Model saved.")
        mlflow.pytorch.log_model(model, "best_model2")
    else:
        patience_counter += 1
        print(f"   No improvement in Test loss. Patience counter: {patience_counter}")
    
    if patience_counter >= config['patience']:
        print("  Early stopping triggered.")
        mlflow.log_metric("early_stopping_epoch", str(epoch))
        break

# Evaluate on test set
test_r2, test_rmse, test_loss = validate(model, test_loader, criterion, device)
print(f"Test R^2: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}, Test Model Loss: {test_loss:.4f}")


# Log test metrics to MLFlow
mlflow.log_metrics(
    {"test_loss": test_loss,
    "test_r2": test_r2,
    "test_rmse": test_rmse
    })

mlflow.end_run()


# Extract predictions for merger with weather for attendance
model.eval()
extract_df = X[X_save_cols].values
game_ids = df['gameId'].values

# Prepare for model
X_tensor = torch.FloatTensor(extract_df).to(device)

# Make predictions
predictions = []
with torch.no_grad():
    # Processsing in batches
    batch_size = config['batch_size']

    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i+batch_size]

        # FF, convert to numpy
        batch_preds = model(batch)
        predictions.append(batch_preds.cpu().numpy())

# Combine all predictions
all_predictions = np.concatenate(predictions, axis=0)

# Return results
result_df = pd.DataFrame({
    'gameId': game_ids,
    'predicted_attendance': all_predictions.flatten()
})

# Write to .csv
result_df.to_csv('model2_out.csv', index=False)