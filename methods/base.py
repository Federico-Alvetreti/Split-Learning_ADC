
from timm.models import VisionTransformer
import torch.nn as nn 

class model(nn.Module):

    def __init__(self, 
                 model: VisionTransformer,
                 channel,
                 split_index,
                 *args, **kwargs): 
        
        super().__init__(*args, **kwargs)
        
        # Build model 
        self.model = self.build_model(model, channel, split_index)

        # Store compression 
        self.compression = 1

        # Store channel 
        self.channel = channel

        # Variable to store communication 
        self.communication = 0 

        # Store name 
        self.name = "Base"

    
    # Function to build model 
    def build_model(self, model, channel, split_index):

        # Split the original model 
        blocks_before = model.blocks[:split_index]
        blocks_after = model.blocks[split_index:]

        # Add comm pipeline and compression modules 
        model.blocks = nn.Sequential(*blocks_before, channel, *blocks_after)

        return model 

    # Forward 
    def forward(self, x):
        if self.training: 
            self.communication += self.compression
        return self.model.forward(x)