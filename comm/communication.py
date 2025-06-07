import torch
import torch.nn as nn
import math 
from typing import Sequence
from io import BytesIO
from copy import deepcopy
import numpy as np


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
    if n_copy == 1 :
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

# Classic analogic channel with gaussian noise 
class Gaussian_Noise_Analogic_Channel(nn.Module):
    def __init__(self,
                  snr_range:float):
        super().__init__()
        self.snr_range = snr_range
        self.dims = -1


    # Adds Additive White Gaussian Noise on tensors 
    def add_awgn_noise(self, tensor: torch.Tensor) -> torch.Tensor:

        # Get random snr in [-snr, snr]
        random_snr = torch.empty(1).uniform_(-self.snr_range, self.snr_range).item()  
        self.actual_snr = random_snr
        # Estimate signal power
        signal_power = torch.linalg.norm(tensor, ord=2, dim=self.dims, keepdim=True)
        size = math.prod([tensor.size(dim=d) for d in self.dims]) if isinstance(self.dims, Sequence) else tensor.size(dim=self.dims)
        signal_power = signal_power / size

        # Compute noise power for the desired SNR
        noise_power = signal_power / (10 ** (random_snr / 10))
        std = torch.sqrt(noise_power)

        # Sample & scale noise
        noise = torch.randn_like(tensor) * std
        noisy_tensor = tensor + noise
        
        return noisy_tensor

    def forward(self, input: torch.Tensor):

        # If in training mode add noise 
        if self.training:
            input = self.add_awgn_noise(input)

        # Simple backward noise hook 
        def _grad_hook(grad):
            grad = self.add_awgn_noise(grad)
            return grad
            
        # Register hook
        if input.requires_grad:
            input.register_hook(_grad_hook)

        return input

# Main Encoder 
class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_layers=2,
                 normalize=True,
                 drop_last_activation=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Resolve float output_size
        self.output_size = round(output_size * input_size) if isinstance(output_size, float) else output_size
        if self.output_size == 0: 
            raise ValueError("Can't train with this compression.")
        
        # Get the real and complex feed forward networks 
        self.real_ffn, self.complex_ffn = get_ffn(input_size=input_size,
                                                    output_size=self.output_size,
                                                    n_layers=n_layers,
                                                    drop_last_activation=drop_last_activation,
                                                    n_copy=2)
        # Store parameters 
        self.normalize = normalize


    def forward(self, x, *args, **kwargs):

        # Get real and complex parts of the input 
        real_part, complex_part = self.real_ffn(x), self.complex_ffn(x)

        # Combine them in a single complex matrix 
        x = torch.complex(real_part, complex_part)

        # Normalize if specified  
        if self.normalize:
            x = x / torch.norm(x, 2, -1, keepdim=True)

        return x

# Main Decoder 
class Decoder(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 n_layers,
                 normalize=True,
                 drop_last_activation=False,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)


        # Get the decoding ffn 
        self.decoder_ffn = get_ffn(input_size=input_size,
                                        output_size=output_size,
                                        n_layers=n_layers,
                                        drop_last_activation=drop_last_activation,
                                        n_copy=1)

        self.normalize = normalize

    def forward(self, x, *args, **kwargs):

        # Normalize if specified  
        if self.normalize:
            x = x / torch.norm(x, 2, -1, keepdim=True)

        # Concat real and imaginary part 
        x = torch.cat((x.real, x.imag), -1)

        # Decode back 
        x = self.decoder_ffn(x)

        return x

# Class that imitates substitutes encoder, decoder and channel (when setting is_channel = True) 
class Identity(nn.Module):

    def __init__(self,
                 input_size = 0,
                 output_size = 0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, input: torch.Tensor):
        return input

