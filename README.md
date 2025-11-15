# ALHO
Autoencoder-driven Learning against Hostile phishing URLs

# Pytorch para CUDA 13.0

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130

# Diagnostico do sistema
import torch
import sys

print("=== Diagnostico do Sistema ===")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")