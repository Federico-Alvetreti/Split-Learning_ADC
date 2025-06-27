import torch
from torch import nn
from timm.models import VisionTransformer
import numpy as np 
from torch_kmeans import KMeans

  
class _FedLite_Quantizer(torch.autograd.Function):
    

    @staticmethod
    def quantize(x: torch.tensor, q: int, R: int, l:int):

        # Original activation shape (B x T x D)
        original_shape = x.shape

        B = original_shape[0]

        # Reshape into B x d
        x = x.view(B, -1)

        # Get d 
        d = x.shape[-1]

        # Reshape into B x q x d/q
        x = x.view(B, q, -1)

        # Reshape into R x  (B * q / R)  x  (d/q) 
        x = x.reshape(R, -1, d // q)

        # Do kmeans  
        kmeans = KMeans(n_clusters=l, max_iter=10, p_norm=2)
        results = kmeans(x)
        labels = results.labels
        centroids = results.centers
        new_labels = labels[..., None].expand(R, -1, d // q)
        x = torch.gather(centroids, 1, new_labels)

        # Reshape into original shape B x T x D
        x = x.view(original_shape)

        return x

    
    @staticmethod
    def forward(ctx, x: torch.Tensor, q: int, R: int, l: int, lbd: float):

        # Get compressed activations 
        quantized_x = _FedLite_Quantizer.quantize(x, q, R, l)
        
        # Store correction for backward
        ctx.correction = lbd * (x - quantized_x)
        
        return quantized_x
    
    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        return grad_outputs + ctx.correction, None, None, None, None 
    
class FedLite_Layer(nn.Module):

    def __init__(self, q: int, r: int, l: int, lbd: float = 5e-5, *args, **kwargs):
        super().__init__()

        self.q = q # Number of sub-vectors 
        self.r = r # Number of groups  
        self.l = l # Number of clusters 
        self.lbd = lbd # Correction weight 

    def forward(self, x: torch.Tensor):
        
        if self.training:
            return _FedLite_Quantizer.apply(x,self.q, self.r, self.l, self.lbd)
        else:
            return x 
        

class model(nn.Module):

    def __init__(self, 
                 model: VisionTransformer,
                 encoder,
                 channel,
                 decoder,
                 split_index,
                 q, # Number of subvectors
                 r, # Ignore
                 l, # Number of clusters 
                 batch_size, 
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        # Build model 
        self.model = self.build_model(model, encoder, channel, decoder, split_index, q, r, l)

        # Store compression
        self.compression = self.get_compression(q,r,l,batch_size)

        # Store channel 
        self.channel = channel

        # Variable to store communication 
        self.communication = 0 

        # Store name 
        self.name = "Fedlite"


    def get_compression(self, q, r, l, batch_size):

        activation_size = 192*197
        n_bits = 32

        # Compute the normal activation size 
        normal_activations_size = n_bits * activation_size * batch_size

        # Get the compressed activations size and the codebook size 
        compressed_activations_size = n_bits * activation_size  * r * l / q
        codebook_size = batch_size * q * np.log2(l)

        # Compute backward and forward compressions 
        forward_compression = (compressed_activations_size + codebook_size) / normal_activations_size
        backward_compression = compressed_activations_size / normal_activations_size

        # Compute the overall compression 
        overall_compression = (forward_compression + backward_compression) / 2

        return overall_compression

    # Function to build model 
    def build_model(self,
                    model, 
                    encoder,
                    channel,
                    decoder,
                    split_index,
                    q, r, l, 
                    ):

        # Split the original model 
        blocks_before = model.blocks[:split_index]
        blocks_after = model.blocks[split_index:]

        # Add quantization layer after the encoder 
        model.blocks = nn.Sequential(*blocks_before, encoder, FedLite_Layer(q, r, l), channel, decoder, *blocks_after)

        return model 
    
    # Forward 
    def forward(self, x):
        if self.training: 
            self.communication += self.compression
        return self.model.forward(x)
    