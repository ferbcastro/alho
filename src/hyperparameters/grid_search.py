import sys
import os
import itertools
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

import training.train as ml

params_count = len(sys.argv) - 1

if params_count < 2:
    print("Usage: python3 train.py path_to_train_csv path_to_test_csv cuda_src")
    print("  path_to_train_csv: Path to the CSV file containing train data")
    print("  path_to_test_csv: Path to the CSV file containing test data")
    print("  cuda_src: Set nvidia gpu")
    sys.exit(1)

TRAIN_SOURCE = sys.argv[1]
TEST_SOURCE = sys.argv[2]
CUDA_SRC = int(sys.argv[3])

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

print('Loading datasets...')
train, test = ml.setup(TRAIN_SOURCE, TEST_SOURCE, cuda_src=CUDA_SRC)
print('Datasets loaded')

best_acc = -1

for params in permute(param_grid):
    b = params['epochs']
    ep = params['batches']
    ec = params['encoding']
    print(f'Computing model for {params}')
    model = ml.train(X, batch_size=b, encoding_dim=ec, num_epochs=ep)
    print(f'Model complete')

    acc = ml.test(model, test)

    if acc > best_acc:
        print(f'New best model found!')
        best_acc = acc
        best_model = model

best_model.export()