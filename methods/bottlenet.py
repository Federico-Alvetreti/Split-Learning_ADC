from timm.models import VisionTransformer
import torch.nn as nn
from copy import deepcopy
import numpy as np
import torch


# Creates n copies of a simple ffn  
def get_ffn(input_size, output_size, n_layers, n_copy, drop_last_activation):
    # Compute the hidden dimensions of the network
    shapes = np.linspace(input_size, output_size, num=n_layers + 1, endpoint=True, dtype=int)

    # Create the model as blocks of linear + ReLU 
    model = []
    for s in range(len(shapes) - 1):
        model.append(nn.Linear(shapes[s], shapes[s + 1]))
        model.append(nn.ReLU())
    if drop_last_activation:
        model = model[:-1]
    model = nn.Sequential(*model)

    # Create the number of copies specifeid in n_copy 
    if n_copy == 1:
        return model
    else:
        models = []
        for _ in range(n_copy):
            _model = deepcopy(model)
            for m in _model.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            models.append(_model)

        return models


# Main Encoder 
class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_layers=2,
                 drop_last_activation=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Resolve float output_size
        self.output_size = round(output_size * input_size) if isinstance(output_size, float) else output_size
        if self.output_size == 0:
            raise ValueError("Can't train with this compression.")

        # Get the real and complex feed forward networks 
        self.ffn = get_ffn(input_size=input_size,
                           output_size=self.output_size,
                           n_layers=n_layers,
                           drop_last_activation=drop_last_activation,
                           n_copy=1)

    def forward(self, x, *args, **kwargs):
        # Get real and complex parts of the input
        x = self.ffn(x)

        return x


# Main Decoder
class Decoder(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 n_layers,
                 drop_last_activation=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get the decoding ffn
        self.decoder_ffn = get_ffn(input_size=input_size,
                                   output_size=output_size,
                                   n_layers=n_layers,
                                   drop_last_activation=drop_last_activation,
                                   n_copy=1)

    def forward(self, x, *args, **kwargs):

        # Decode back 
        x = self.decoder_ffn(x)

        return x


class model(nn.Module):

    def __init__(self,
                 model: VisionTransformer,
                 channel,
                 split_index,
                 n_layers,
                 compression,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Build model 
        self.model = self.build_model(model, channel, split_index, n_layers, compression)

        # Store compression 
        self.compression_ratio = compression

        # Store channel 
        self.channel = channel

        # Variable to store communication 
        self.communication = 0

        # Store name 
        self.name = "Bottelnet"

    # Function to build model
    def build_model(self, model, channel, split_index, n_layers, compression):
        # Get encoder and decoder
        encoder = Encoder(model.embed_dim, compression, n_layers)
        decoder = Decoder(encoder.output_size, model.embed_dim, n_layers)

        # Split the original model 
        blocks_before = model.blocks[:split_index]
        blocks_after = model.blocks[split_index:]

        # Add comm pipeline and compression modules 
        model.blocks = nn.Sequential(*blocks_before, encoder, channel, decoder, *blocks_after)

        return model

        # Forward

    def forward(self, x):
        batch_size = x.shape[0]
        if self.training:
            self.communication += self.compression_ratio * batch_size
        return self.model.forward(x)
