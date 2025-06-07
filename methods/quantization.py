
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
        
        # Quantization forward from https://github.com/zfscgy/SplitLearning
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
        self.compression = self.get_compression(n_bits)

        # Store channel 
        self.channel = channel

        # Variable to store communication 
        self.communication = 0 

        # Store name 
        self.name = "Quantization"

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
    

    # Function to get the compression level of this method 
    def get_compression(self, n_bits):
        
        # Compute forward and backward compression   (following https://arxiv.org/pdf/2305.18469)
        forward_compression = n_bits / 32
        backward_compression =  1

        # Return average compression
        compression = (forward_compression + backward_compression) / 2

        return compression 

    # Forward 
    def forward(self, x):
        if self.training: 
            self.communication += self.compression
        return self.model.forward(x)