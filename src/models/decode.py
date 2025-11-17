import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from models.autoencoder import UndercompleteAE

if len(sys.argv) < 2:
    print('Usage: python3 encode.py cuda_src')
    print('\t cuda_src: gpu id')
    sys.exit(1)

CUDA_SRC = int(sys.argv[1])

test_path = 'bigrams/test_set_bigrams.csv'
models_dir = 'UCAE/'
model_tags = [4] # change
model_encodings = [32]
compress_rates = [0.2]

print('Loading dataframes...')

test_df = test_df.drop('label', axis=1)
print('Dataframes loaded')
test_size = test_df.size()
test_sum = test.sum().sum()
ones_rate = test_sum / test_size
print(f'Ones rate {ones_rate:.4f}')

print('Creating tensors...')
test_tensor = torch.from_numpy(test_df.to_numpy()).float()

torch.cuda.set_device(CUDA_SRC)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device is', device)

if train_df.shape[1] != test_df.shape[1]:
    print('Number of cols not equal! Exiting...')
    sys.exit(1)

input_dims = test_df.shape[1]
print('Input dims is', input_dims)
train_tensor = train_tensor.to(device)
test_tensor = test_tensor.to(device)

for i, tag in enumerate(model_tags):
    print('Encoding for model', tag)
    model = UndercompleteAE(input_dim=input_dims, latent_dim=model_encodings[i], compression_rate=compress_rates[i]).to(device)
    model_state_path = f'{models_dir}/UCAE_state_{tag}'
    model.load_state_dict(torch.load(model_state_path, map_location=device))

    model.eval()

    try:
        model.load_state_dict(torch.load(model_state_path, map_location=device))
        model.eval()

        with torch.no_grad():
            test_decode = model.decode(test_tensor)
            test_array = test_encode.cpu().numpy()
            df = pd.DataFrame(test_array)
            df.to_csv(f'test_dec_{tag}.csv', index=False)

        print(f'Test decoded and exported for model {tag}')

    except Exception as e:
        print(f'Error processing model {tag}: {e}')
        continue
