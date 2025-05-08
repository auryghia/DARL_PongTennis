# Train AAE + discriminator to algin embeddings
from models.encoder import Encoder
from models.decoder import Decoder
import torch.nn as nn

encoder = Encoder(latent_dim=128)
decoder = Decoder(latent_dim=128)
discriminator = nn.Sequential(
    nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid()
)
