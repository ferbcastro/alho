''' Train an undercomplete autoencoder on the provided CSV data
    and export the encoded data to a new CSV file.
'''

import sys
import os

import pandas as pd

from sklearn.preprocessing import StandardScaler

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from models.autoencoder import UndercompleteAE
import training.loss as loss

params_count = len(sys.argv) - 1

if params_count < 5:
    print("Usage: python3 train.py path_to_csv batch_size num_epochs encoding_dim cuda_src")
    print("  path_to_csv: Path to the CSV file containing the data")
    print("  batch_size: Batch size for training")
    print("  num_epochs: Number of backprop cycles")
    print("  encoding_dim: Encoding dimensions")
    print("  cuda_src: Set nvidia gpu")
    sys.exit(1)

CSV_SOURCE = sys.argv[1]
BATCH_SIZE = int(sys.argv[2])
NUM_EPOCHS = int(sys.argv[3])
ENCODING_DIM = int(sys.argv[4])
CUDA_SRC = int(sys.argv[5])

def setup():
    """Sets up the environment by reading the CSV file and preparing the data."""

    df = pd.read_csv(CSV_SOURCE)

    df = df.drop('url', axis=1)
    remaining_labels = df.pop('label')

    X = df

    return X, remaining_labels

def train(X: pd.DataFrame, batch_size: int = 32):
    """Trains the autoencoder on the provided data using batch processing."""

    # Converting to PyTorch tensor
    X_tensor = torch.FloatTensor(X)

    # Select gpu by id
    torch.cuda.set_device(CUDA_SRC)
    # Set to use cpu if gpu not available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = X_tensor.to(device)

    # Create dataset and dataloader for batch processing
    dataloader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    # Setting random seed for reproducibility
    torch.manual_seed(42)

    # Number of cols/input features
    input_size = X.shape[1]
    # Desired number of output dimensions
    encoding_dim = ENCODING_DIM
    model = UndercompleteAE(input_size, encoding_dim).to(device)

    # Loss function and optimizer
    criterion = loss.FocalLoss()
    optimizer = Adam(model.parameters(), lr=0.003, weight_decay=0)

    # Training the autoencoder
    num_epochs = NUM_EPOCHS
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_data in dataloader:
            batch_X = batch_data[0].to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_X)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    # Generate encoded data for the entire dataset
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        encoded_data = model.encoder(X_tensor).detach().numpy()

    return encoded_data, model

def export_data(encoded_data, filename="encodedData.csv", labels: pd.DataFrame=None) -> None:
    """Exports the encoded data to a CSV file."""

    encoded_df = pd.DataFrame(encoded_data)

    if labels is not None:
        encoded_df = encoded_df.join(labels)

    encoded_df.to_csv(filename, index=False)

    print(f"Encoded data exported to {filename}")


def main():
    """Main function to train the autoencoder and export encoded data."""

    # List available gpus
    print('Available gpus:', torch.cuda.device_count())
    X, labels = setup()
    print('Dataset loaded')
    encoded_data, model = train(X, batch_size=BATCH_SIZE)

    export_data(encoded_data, labels=labels)
    model.export()

if __name__ == "__main__":
    main()
