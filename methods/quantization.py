
from timm.models import VisionTransformer
import torch.nn as nn
import torch 

# Simple Uniform Quantization  
class _QuantizeFunction(torch.autograd.Function):


    @staticmethod
    def quantize(x: torch.tensor, n_bits: int):

        # Handle the case of complex tensor applying the quantization to both real and imaginary parts
        if torch.is_complex(x):
            real_q = _QuantizeFunction.quantize(x.real, n_bits)
            imag_q = _QuantizeFunction.quantize(x.imag, n_bits)
            return torch.complex(real_q, imag_q)
        
        # Get min and max
        x_min = x.min()
        x_max = x.max()

        # Get levels 
        levels = 2 ** n_bits

        # Get the step 
        scale = (x_max - x_min) / (levels - 1)

        # Quantize: map x to integers [0, levels-1]
        q_x = torch.round((x - x_min) / scale).clamp(0, levels - 1)

        # Dequantize: map back to float domain
        quantized_x = q_x * scale + x_min

        return quantized_x
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, n_bits: int):
        ctx.quantize_n_bits = n_bits
        return _QuantizeFunction.quantize(x, n_bits)

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        return _QuantizeFunction.quantize(grad_outputs, ctx.quantize_n_bits), None
    

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
    
    # Forward 
    def forward(self, x):
        if self.training: 
            self.communication += self.compression
        return self.model.forward(x)