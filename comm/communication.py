import torch
import torch.nn as nn
import math 
from typing import Sequence


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

