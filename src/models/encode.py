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

concat_paths = ['bigrams/train_set_bigrams.csv', 'bigrams/validation_set_bigrams.csv']
test_path = 'bigrams/test_set_bigrams.csv'
models_dir = 'UCAE/'
model_tags = [1, 4, 7, 10] # change
model_encodings = [32, 64, 128, 512]
compress_rates = [0.2, 0.2, 0.2, 0.2, 0.7]

print('Loading dataframes...')
df_to_concat = []
for path in concat_paths:
    df_to_concat.append(pd.read_csv(path))
concat_df = pd.concat(df_to_concat, axis=0, ignore_index=True).drop('url', axis=1)
concat_label = concat_df.pop('label')
print('Concat dimensions:', concat_df.shape)

test_df = pd.read_csv(test_path).drop('url', axis=1)
test_label = test_df.pop('label')
print('Dataframes loaded')

train_df = concat_df
train_label = concat_label
print('Creating tensors...')

train_tensor = torch.from_numpy(train_df.to_numpy()).float()
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
            train_encode = model.encode(train_tensor)
            train_array = train_encode.cpu().numpy()
            df = pd.DataFrame(train_array)
            df['label'] = train_label.reset_index(drop=True).values
            df.to_csv(f'train_enc_{tag}.csv', index=False)

            test_encode = model.encode(test_tensor)
            test_array = test_encode.cpu().numpy()
            df = pd.DataFrame(test_array)
            df['label'] = test_label.reset_index(drop=True).values
            df.to_csv(f'test_enc_{tag}.csv', index=False)

        print(f'Train and test encoded and exported for model {tag}')

    except Exception as e:
        print(f'Error processing model {tag}: {e}')
        continue

    print('Train and test encoded and exported')
