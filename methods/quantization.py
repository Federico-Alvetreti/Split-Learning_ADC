
from timm.models import VisionTransformer
import torch.nn as nn
import torch 

# Simple Uniform Quantization  
class _QuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, n_bits: int):

        # Handle the case of complex tensor (when the encoder is applied)
        if torch.is_complex(x):
            real_q = _QuantizeFunction.forward(ctx, x.real, n_bits)
            imag_q = _QuantizeFunction.forward(ctx, x.imag, n_bits)
            return torch.complex(real_q, imag_q)
        
    
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        x_min = x.min()
        x_max = x.max()
        levels = 2 ** n_bits

        thresholds = torch.linspace(x_min, x_max, steps=levels + 1, device=x.device)
        centers = (thresholds[:-1] + thresholds[1:]) / 2  # Midpoints

        diffs = torch.abs(x_flat.unsqueeze(-1) - centers.unsqueeze(0))  # [batch, dim, levels]
        closest_indices = torch.argmin(diffs, dim=-1)  # [batch, dim]

        quantized_flat = centers[closest_indices]
        quantized_x = quantized_flat.view_as(x)

        ctx.save_for_backward(x)  # optional, for custom gradient

        

        return quantized_x

    # Return unchanged gradient 
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None  # None for n_bits


class Quantization_Layer(nn.Module):

    def __init__(self, n_bits: int):
        super().__init__()
        self.n_bits = n_bits

    def forward(self, x: torch.Tensor):
        if self.training:
            return _QuantizeFunction.apply(x, self.n_bits)
        else:
            return x 
        

class model(nn.Module):

    def __init__(self, 
                 model: VisionTransformer,
                 encoder,
                 channel,
                 decoder,
                 split_index,
                 n_bits,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        # Build model 
        self.model = self.build_model(model, encoder, channel, decoder, split_index, n_bits)

        # Store compression 
        self.compression = n_bits / 32

        self.name = "Quantization"
        self.channel = channel

    # Function to build model 
    def build_model(self,
                    model, 
                    encoder,
                    channel,
                    decoder,
                    split_index,
                    n_bits, 
                    ):

        # Split the original model 
        blocks_before = model.blocks[:split_index]
        blocks_after = model.blocks[split_index:]

        # Add quantization layer after the encoder 
        model.blocks = nn.Sequential(*blocks_before, encoder, Quantization_Layer(n_bits), channel, decoder, *blocks_after)

        return model 

    # Forward 
    def forward(self, x):
        return self.model.forward(x)