from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

if __package__:
    from .model import PendulumModel
else:
    from model import PendulumModel

def train(dataset_path: str,
          model_dir: str = "pendulum/models",
          hidden_size: int = 64,
          num_hidden_layers: int = 1,
          learning_rate: float = 0.001,
          train_batch_size: int = 64,
          eval_batch_size: int = 64,
          num_epochs: int = 20) -> str:

    # Set the device to GPU if available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load the dataset
    with np.load(dataset_path) as dataset:
        data = dataset["dataset"]

    # Split input and output
    X = data[:, :5]  # Input features: l, m, b, theta0, omega0
    y = data[:, 5:]  # Output features: theta_next, omega_next
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")

    # Split the dataset into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size = train_batch_size,
        shuffle = True,
        pin_memory = True
    )

    eval_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size = eval_batch_size,
        shuffle = False,
        pin_memory = True
    )

    # Define the model
    model = PendulumModel(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers
    ).to(device)

    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Evaluation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(eval_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8e}, Val Loss: {val_loss:.8e}")

    # Save the model
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model_{len(data)}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model_path
