
from timm.models import VisionTransformer
import torch.nn as nn
import numpy as np 
import torch 

# Selects the Top-K of a tensor with a random portion 
class RandomTopKModifier(nn.Module):
    def __init__(self, rate: float, random_portion: float = 0.1):
        super(RandomTopKModifier, self).__init__()
        self.rate = rate
        self.random_portion = random_portion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        k = max(1, round(self.rate * x_flat.shape[1]))
        sample_dim = x_flat.shape[1]

        if self.training:
            _, top_k_indices = torch.topk(torch.abs(x_flat), k, sorted=False)

            probs = torch.full_like(x_flat.real, self.random_portion / (sample_dim - k))
            probs = torch.scatter(probs, 1, top_k_indices,
                                  (1 - self.random_portion) / k)
            selected_indices = torch.multinomial(probs, k)
            mask = torch.scatter(torch.zeros_like(x_flat), 1, selected_indices, 1)

            return mask.view(*x.shape) * x

        else:
            return x


class model(nn.Module):

    def __init__(self, 
                 model: VisionTransformer,
                 encoder,
                 channel,
                 decoder,
                 split_index,
                 rate,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        # Build model 
        self.model = self.build_model(model, encoder, channel, decoder, split_index, rate)

        # Store compression
        self.compression = self.get_compression_level(model, rate)

        self.name = "Random_Top_K"
        self.channel = channel

    # Function to build model 
    def build_model(self, 
                    model, 
                    encoder,
                    channel,
                    decoder,
                    split_index,
                    rate, 
                    ):

        # Split the original model 
        blocks_before = model.blocks[:split_index]
        blocks_after = model.blocks[split_index:]

        # Add the RandomTopK selection after the encoder 
        model.blocks = nn.Sequential(*blocks_before, encoder, RandomTopKModifier(rate), channel, decoder, *blocks_after)

        return model 

    # Function to get the compression level of this method 
    def get_compression_level(self, model, rate):
        
        # Get activations size 
        token_length = model.embed_dim
        n_tokens = 197
        activations_size = n_tokens * token_length

        # Compute compression  
        compression = rate * (1 + np.ceil(np.log2(activations_size)) / 32)

        return compression 

        
    # Forward 
    def forward(self, x):
        return self.model.forward(x)