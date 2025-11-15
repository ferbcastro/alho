import torch
import torch.nn as nn

""" TODO : Make generic class that takes into a configuration
object and generates a feed forward net, then the encoder
and decoder can be instaces with diffents objects. Also
will make fine tunning easier
"""


class Encoder(nn.Module):
    def __init__(self, input_dim=9860, latent_dim=1024, compression_rate=0.5):
        super().__init__()

        self.layers = nn.Sequential()

        current_dim = input_dim

        while int(current_dim * compression_rate) > latent_dim:
            next_dim = int(current_dim * compression_rate)
            self.layers.append(nn.Linear(current_dim, next_dim))
            self.layers.append(nn.ReLU())
            current_dim = next_dim

        self.layers.append(nn.Linear(current_dim, latent_dim))

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, output_dim=9860, latent_dim=1024, decompression_rate=2):
        super().__init__()

        self.layers = nn.Sequential()

        current_dim = latent_dim

        while int(current_dim * decompression_rate) < output_dim:
            next_dim = int(current_dim * decompression_rate)
            self.layers.append(nn.Linear(current_dim, next_dim))
            self.layers.append(nn.ReLU())
            current_dim = next_dim

        self.layers.append(nn.Linear(current_dim, output_dim))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


class UndercompleteAE(nn.Module):
    def __init__(self, input_dim=9860, latent_dim=1024, compression_rate=0.5):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, compression_rate)
        self.decoder = Decoder(input_dim, latent_dim, 1 / compression_rate)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def export(self, path: str = "./UCAE_state") -> None:
        torch.save(self.state_dict(), path)
