import sys
import os
import itertools

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

import training.train as ml

if params_count < 2:
    print("Usage: python3 train.py path_to_csv cuda_src")
    print("  path_to_csv: Path to the CSV file containing the data")
    print("  cuda_src: Set nvidia gpu")
    sys.exit(1)

CSV_SOURCE = sys.argv[1]
CUDA_SRC = int(sys.argv[2])

param_grid = {
    'epochs': [10, 20, 30],
    'batches': [16, 32],
    'encoding': [64, 128, 256, 512]
}

def permute(dct):
    labels = list(dct.keys())
    values = list(dct.values())
    for comb in itertools.product(*values):
        yield dict(zip(labels, comb))

"""
Perform grid search. It may take too long.
"""

# List available gpus
print('Available gpus:', torch.cuda.device_count())

X, labels = ml.setup(csv=CSV_SOURCE, cuda_src=CUDA_SRC)
print('Dataset loaded')

# Set best to 2^64
best_loss = 1 << 64

for params in permute(param_grid):
    b = params['epochs']
    ep = params['batches']
    ec = params['encoding']
    model, loss = ml.train(X, batch_size=b, encoding_dim=ec, num_epochs=ep)

    if loss < best_loss:
        print(f'New best loss: {loss} for {params}')
        best_loss = loss
        best_model = model

best_model.export()