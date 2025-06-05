import torch
import torch.nn as nn
from scripts.utils import tensor_to_bytes, get_ffn
import math 
from typing import Sequence


# Classic analogic channel with gaussian noise 
class Gaussian_Noise_Analogic_Channel(nn.Module):
    def __init__(self,
                  snr_range:float,
                  quantization_compression: float = 1.0):
        super().__init__()

        self.snr_range = snr_range
        self.quantization_compression = quantization_compression
        self.total_communication = 0
        self.dims = -1
        self.previous_grad = torch.zeros(0)
        self.actual_snr = 0


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

        # If in training mode update communication cost and add noise 
        if self.training:
            self.total_communication += self.quantization_compression * tensor_to_bytes(input)
            input = self.add_awgn_noise(input)

        # Simple backward noise hook 
        def _grad_hook(grad):
            if self.previous_grad.numel() == 0:
                self.previous_grad = torch.zeros_like(grad)
            print("snr=", self.actual_snr)
            # Variance across the batch (dim=0), per parameter
            real_var = grad.real.var(dim=0, unbiased=False)
            imag_var = grad.imag.var(dim=0, unbiased=False)

            # Average real and imaginary variance
            per_param_var = (real_var + imag_var) / 2

            # Return average across all parameters
            print("average_variance = ",per_param_var.mean().item())
            self.previous_grad = grad
            
            grad = self.add_awgn_noise(grad)
            self.total_communication += tensor_to_bytes(grad)
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
