import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)
from models.autoencoder import UndercompleteAE
from models.loss import FocalLoss

from torch.utils.data import DataLoader, TensorDataset

if len(sys.argv) < 2:
    print('Usage python3 test_set_analysis cuda_src')
    print('\t cuda_src: gpu id')
    sys.exit(1)

CUDA_SRC = int(sys.argv[1])

test_path='bigrams/test_set_bigrams.csv'

model_tags = [3, 33, 57, 81]
model_enc = [16, 32, 256, 512]
model_comp = [0.2, 0.2, 0.2, 0.2]
model_dir = './UCAE_512'

torch.cuda.set_device(CUDA_SRC)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device is', device)

print('Loading test dataframe...')
test_df = pd.read_csv(test_path).drop(['label', 'url'], axis=1)
print('Test dataframe loaded')
input_dims = test_df.shape[1]
print('Test dataframe dimensions', test_df.shape)

print('Creating test loader...')
test_arr = test_df.to_numpy()
test_tensor = torch.from_numpy(test_arr).float()
test_tensor = test_tensor.to(device)
test_loader = DataLoader(TensorDataset(test_tensor), batch_size=512, shuffle=False)
print('Test loader created')

for i, tag in enumerate(model_tags):
    model_state_path = f'{model_dir}/UCAE_state_{tag}'
    model = UndercompleteAE(input_dim=input_dims, latent_dim=model_enc[i], compression_rate=model_comp[i]).to(device)

    print('Analysis for model:', tag)

    try:
        model.load_state_dict(torch.load(model_state_path, map_location=device))
        model.eval()

        reconstructed = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch[0].to(device)
                dec = model(batch)
                reconstructed.append(dec.cpu().numpy())

        rec = np.concatenate(reconstructed)

        mse = mean_squared_error(test_arr.astype(np.float32), rec)
        mae = mean_absolute_error(test_arr.astype(np.float32), rec)
        rmse = np.sqrt(mse)
        print(f'mse={mse:.5f}, mae={mae:.5f}, rmse={rmse:.5f}')

        rec_tensor = torch.tensor(rec)

        bce = F.binary_cross_entropy(rec_tensor, test_tensor, reduction='mean').item()
        focal_criterion = FocalLoss()
        focal_loss = focal_criterion(rec_tensor, test_tensor).item()
        print(f'bce={bce:.5f}, focal_loss={focal_loss:.5f}')

        # Erro percentual absoluto médio (MAPE)
        mape = np.mean(np.abs((test_arr - rec) / (test_arr + 1e-8))) * 100
        acc = test_tensor * rec_tensor + (1 - test_tensor) * (1 - rec_tensor)
        print(f'mape={mape:.5f}, acc={acc:.5f}')

    except Exception as e:
        print(f'Error processing model {tag}: {e}')
        continue

    print(f'Analysis for model {tag} done!')





