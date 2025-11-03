import sys
import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)
from models.autoencoder import UndercompleteAE
from torch.utils.data import DataLoader, TensorDataset

def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)

    if 'url' not in df.columns or 'label' not in df.columns:
        print("CSV error read")

    df = df.drop(['url', 'label'], axis = 1)
    features_tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)

    return features_tensor, df.shape[1]


def evaluate(model, dataloader, device, list):
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_X in dataloader:
            batch_X = batch_X[0].to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_X)

            list.append(loss.item())
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_list = []

    # aqui vai o path do arquivo teste
    test_path = sys.argv[1]
    X_tensor, dim = load_csv_data(test_path)
    X_tensor.to(device)

    model = UndercompleteAE(input_dim=dim, latent_dim=1024).to(device)
    model.load_state_dict(torch.load("./UCAE_state", map_location=device))
    print("Modelo carregado")

    test_dataset = TensorDataset(X_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # avaliar modelo
    reconstruction_loss = evaluate(model, test_loader, device, loss_list)
    print(f"Loss medio no conjunto de teste: {reconstruction_loss:.6f}")
    np.save('loss.npy', np.array(loss_list))