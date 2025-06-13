import torch
from torch import nn
from torch_kmeans import SoftKMeans
from timm.models import VisionTransformer


def get_hook(diff):
    def _grad_hook(grad):
        grad = grad + diff
        return grad

    return _grad_hook

class FedLiteQuantizer(nn.Module):
    def __init__(self, q: int, r: int, l: int, lbd:float = 5e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.q = q
        self.r = r
        self.l = l
        self.lbd = lbd

        self.hook = None

    def forward(self, x: torch.Tensor):
        # x has shape B x T x F

        B,T,F = x.shape

        if (T * F) % self.q != 0:
            raise(ValueError("q must divide T*F"))

        x_reshaped = x.view(B, -1)

        # shape B x q x d/q
        x_reshaped = x_reshaped.view(B, self.q, -1)

        # shape R x qB/R
        x_reshaped = x_reshaped.view(self.r, self.q, -1)

        kmeans = SoftKMeans(n_clusters=self.l, verbose=False)

        results = kmeans(x_reshaped)

        labels = results.labels
        centroids = results.centers

        new_labels = labels[..., None].expand(self.r, self.q, centroids.shape[-1])
        rec = torch.gather(centroids, 1, new_labels)
        rec = rec.view(B,T,F)

        diff = self.lbd * (x - rec)

        if self.hook is not None:
            self.hook.remove()
        self.hook = x.register_hook(get_hook(diff))

        print(rec.shape)
        return rec



class model(nn.Module):

    def __init__(self, 
                 model: VisionTransformer,
                 encoder,
                 channel,
                 decoder,
                 split_index,
                 q, r, l,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        # Build model 
        self.model = self.build_model(model, encoder, channel, decoder, split_index, q, r, l)

        # Store compression
        self.compression = 1

        # Store channel 
        self.channel = channel

        # Variable to store communication 
        self.communication = 0 

        # Store name 
        self.name = "Fedlite"

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
        model.blocks = nn.Sequential(*blocks_before, encoder, FedLiteQuantizer(q, r, l), channel, decoder, *blocks_after)

        return model 
    
    # Forward 
    def forward(self, x):
        if self.training: 
            self.communication += self.compression
        return self.model.forward(x)
    
