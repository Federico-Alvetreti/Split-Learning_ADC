import torch
import torch.nn as nn
from typing import Union, Tuple, Sequence
from scripts.utils import tensor_to_bytes, get_ffn
    
# Analogic Gaussian Noise Channel 
class Gaussian_Noise_Analogic_Channel(nn.Module):
    def __init__(self, snr: Union[float, Tuple[float, float]], max_symbols: int):
        super().__init__()

        assert snr is not None, "SNR must be specified."
        self.snr = snr  # Single SNR value or range
        self.total_communication = 0
        self.max_symbols = max_symbols

    def get_snr(self, batch_size: int, device: Union[str, torch.device] = "cpu"):
        """Returns a fixed or sampled SNR value."""
        if isinstance(self.snr, Sequence):  # If it's a range (tuple), sample uniformly
            r1, r2 = self.snr
            return torch.rand(batch_size, device=device) * (r1 - r2) + r2
        return self.snr  # Return fixed value if not a range

    def apply_noise(self, x, signal_power, snr):
        """Applies Gaussian noise based on SNR."""

        if isinstance(snr, torch.Tensor):
            snr = snr.view([-1] + [1] * (x.ndim - 1))  # Expand for broadcasting

        noise_power = signal_power / (10 ** (snr / 10))
        std = torch.sqrt(noise_power)
        noise = torch.randn_like(x) * std

        return x + noise

    def forward(self, x: torch.Tensor, snr=None):

        # Get activations size 
        _,R,C = x.size()
        if R*C > self.max_symbols:
            raise ValueError("Too much symbols in the channel.")

        """Adds Gaussian noise to the input tensor based on the given SNR."""
        # Add the total communication cost as the number of bits sent through the channel
        if self.training: 
            self.total_communication += tensor_to_bytes(x) 

        # Get snr value
        snr = snr if snr is not None else self.get_snr(len(x), x.device)
        
        if isinstance(snr, torch.Tensor):
            snr = snr.to(x.device)
    
        # Compute signal power along the given dimensions
        signal_power = torch.mean(x ** 2, dim= -1, keepdim=True)  

        # Apply noise
        noisy_signal = self.apply_noise(x, signal_power, snr)  # Apply noise

        return noisy_signal
    
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
        self.output_size = (max(int(input_size * output_size), 1) if isinstance(output_size, float) else output_size)

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
