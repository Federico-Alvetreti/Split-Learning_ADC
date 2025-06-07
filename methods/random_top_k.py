
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
                 batch_size,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        # Build model 
        self.model = self.build_model(model, encoder, channel, decoder, split_index, rate)

        # Store compression
        self.compression = self.get_compression_level(model, rate, batch_size)

        # Store channel 
        self.channel = channel

        # Variable to store communication 
        self.communication = 0 

        # Store name 
        self.name = "Random_Top_K"


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
    def get_compression_level(self, model, rate, batch_size):

        # Get the number of elements of each batch  
        img_size = model.default_cfg['input_size'][-1]
        patch_size = model.patch_embed.patch_size[0] 
        n_tokens = (img_size // patch_size) ** 2 + 1 
        token_length = model.embed_dim
        activations_size = n_tokens * token_length
        batch_elements = activations_size * batch_size

        # Get the number of elements after the top-k 
        number_of_new_elements = max(1, round(self.rate * batch_elements))
    
        # Compute forward and backward compression   (following https://arxiv.org/pdf/2305.18469)
        forward_compression = number_of_new_elements * (1 + np.ceil(np.log2(batch_elements)) / 32)
        backward_compression = number_of_new_elements

        # Return average compression
        compression = (forward_compression + backward_compression) / 2 

        return compression 

        
    # Forward 
    def forward(self, x):
        if self.training: 
            self.communication += self.compression
        return self.model.forward(x)