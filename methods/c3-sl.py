
from timm.models import VisionTransformer
import torch.nn as nn 
import torch



# Circular convolution 
def batch_circular_convolution_fft(x, h):
    # x: [B//R, R, D]
    # h: [R, D] → need to expand to [1, R, D] for broadcasting
    h = h.unsqueeze(0)  # shape becomes [1, R, D]

    # Apply FFT along the last dimension
    x_fft = torch.fft.fft(x, dim=-1)
    h_fft = torch.fft.fft(h, dim=-1)

    # Element-wise complex multiplication in frequency domain
    result_fft = x_fft * h_fft

    # Inverse FFT and keep real part
    return torch.fft.ifft(result_fft, dim=-1).real

def batch_circular_correlation_fft(x, h):
    """
    x: Tensor of shape [B, 1, D]
    h: Tensor of shape [R, D]
    Returns: Tensor of shape [B, R, D] with circular correlation results
    """
    # Ensure shapes are compatible
    x = x  # [B, 1, D]
    h = h.unsqueeze(0)  # [1, R, D]

    # Compute FFTs
    X = torch.fft.fft(x, dim=-1)        # [B, 1, D]
    H = torch.fft.fft(h, dim=-1).conj() # [1, R, D]

    # Broadcast multiply: result is [B, R, D]
    Y = X * H

    # Inverse FFT and take real part
    return torch.fft.ifft(Y, dim=-1).real

# Main Encoder 
class Encoder(nn.Module):
    def __init__(self,
                 R,
                 keys,
                 *args, **kwargs):
        
        self.R = R
        self.keys = keys

        super().__init__(*args, **kwargs)

    def forward(self, x, *args, **kwargs):

        if self.training:

            # Flatten into B x d
            x  = torch.flatten(x, start_dim=1)    
            
            # Store dimensions 
            batch_dim, features_dim = x.shape

            # Reshape in B/R x R x d 
            x = x.reshape(batch_dim // self.R, self.R, features_dim)

            # Do batch circular convolution 
            x = batch_circular_convolution_fft(x, self.keys)

            # Sum over the R elements
            x = x.sum(dim=1)  # shape: [B//R, D]

        return x

# Main Decoder 
class Decoder(nn.Module):

    def __init__(self,
                 keys,
                 n_tokens,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)


        # Store keys 
        self.keys = keys
        self.n_tokens = n_tokens

    def forward(self, x, *args, **kwargs):

        if self.training:
            shapes = x.shape

            # from B/R x d  -> B/R x 1 x d 
            x = x.unsqueeze(1)

            # Decode into a B/R x R x D tensor 
            x = batch_circular_correlation_fft(x, self.keys)

            # Get dimensions 
            batch_over_R, R, features_dim = x.shape
            batch_size = batch_over_R * R

            # # Reshape into B x D
            # x = x.reshape(batch_size, features_dim)

            # Reshape into B X T X F
            x = x.reshape(batch_size, self.n_tokens, -1)

        return x



class model(nn.Module):

    def __init__(self, 
                 model: VisionTransformer,
                 channel,
                 split_index,
                 R,
                 batch_size,
                 *args, **kwargs): 
        
        super().__init__(*args, **kwargs)


        # Get dimensions 
        self.batch_size = batch_size
        self.n_tokens, self.token_dim = self.get_dimensions(model)

        # Instantiate keys
        self.keys = self.instantiate_keys(R, self.n_tokens * self.token_dim)
        
        # Build model 
        self.model = self.build_model(model, channel, split_index, R)

        # Store compression 
        self.compression_ratio = 1 / R

        # Store channel 
        self.channel = channel

        # Variable to store communication 
        self.communication = 0 

        # Store name 
        self.name = "Bottelnet"



    # Get the dimensions of activations of the model 
    def get_dimensions(self, model):

        img_size = model.default_cfg['input_size'][-1]
        patch_size = model.patch_embed.patch_size[0]

        n_tokens = (img_size // patch_size) ** 2 + 1 
        token_dim = model.embed_dim

        return n_tokens, token_dim




    def instantiate_keys(self, R, flat_activation_size):

        # Get device 
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create keys 
        keys = torch.normal(0, 1/flat_activation_size, size=(R, flat_activation_size)).to(device) 
        keys = keys / keys.norm(dim=1, keepdim=True)

        return keys 
    
    # Function to build model 
    def build_model(self, model, channel, split_index,R):

        # Get encoder and decoder
        encoder = Encoder(R, self.keys)
        decoder = Decoder(self.keys, self.n_tokens)

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