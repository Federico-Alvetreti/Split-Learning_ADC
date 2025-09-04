
from timm.models import VisionTransformer
import torch.nn as nn 
import torch

# An attention class that stores class token attention scores
class Store_Scores_Attention(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.class_token_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.attn.qkv(x).reshape(B, N, 3, self.attn.num_heads, self.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.attn.q_norm(q), self.attn.k_norm(k)
        q = q * self.attn.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        self.class_token_attention = attn[:, :, 0, :].mean(dim=1).detach()  # Save attention over class token
        attn = self.attn.attn_drop(attn)
        attn_output = attn @ v
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x
  
# Delta block that does selection and updates budget 
class Delta_Block(nn.Module):

    def __init__(self, block, alfa):
        super().__init__()

        self.block = block
        self.alfa = alfa
        self.budget = 0.9
        self.delta = 0

    # Method to update delta and alfa of the block 
    def update_budget(self, input : torch.Tensor, output : torch.Tensor):

        # Get input class token 
        input_class_token = input[:, 0]

        # Get output class token 
        output_class_token = output[:, 0]

        # Get current budget 
        current_budget = self.budget 

        # Get previous delta
        previous_delta  = self.delta

        # Get new delta 
        new_delta = (torch.norm(output_class_token - input_class_token, p=2).item())**2

        # Store it
        self.delta = new_delta 

        # Update the budget
        new_budget =  current_budget + (new_delta - previous_delta) * self.alfa

        # Clip it between 0.5 and 1 and  store it 
        self.budget =  max(0.5, min(new_budget, 1))

    # Selects tokens 
    def select_tokens(self, x:torch.Tensor) -> torch.Tensor: 

        # Get shape of the output 
        B, N, _ = x.shape

        # Get class token attention scores 
        class_token_attention = self.block.attn.class_token_attention

        # Get the percentage of blocks to retain as the block budget  
        block_percentage = self.budget
        
        # Compute the number of tokens to keep (excluding class token)
        num_tokens_to_keep = int(block_percentage * N)
        
        # Sort indices based on entropy (excluding class token)
        sorted_indices = torch.argsort(class_token_attention[:, 1:], dim=1, descending=True)
        
        # Select top-k tokens per batch
        selected_indices = sorted_indices[:, :num_tokens_to_keep] + 1  # Shift to account for class token

        # Concatenate class token with selected tokens
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1)
        x = torch.cat([x[:, :1, :], x[batch_indices, selected_indices, :], ], dim=1)

        return x

    # Modified forward 
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Store input 
        input_x = x.clone()

        # Normal forward 
        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))
        x = x + self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))

        # Update budget 
        self.update_budget(input_x, x)

        # Select token
        if self.training:
            x = self.select_tokens(x)

        return x

# Alast model
class model(nn.Module):

    def __init__(self, 
                 model: VisionTransformer,
                 encoder,
                 channel,
                 decoder,
                 split_index,
                 k,
                 alfa,
                 *args, **kwargs): 
        
        super().__init__(*args, **kwargs)

        self.blocks = None 
        
        # Build model 
        self.model = self.build_model(model, encoder, channel, decoder, split_index, alfa)

        # Store compression 
        self.compression = 1 
        self.k = k

        self.name = "Alast"
        self.channel = channel
        
    
    # Function to build model 
    def build_model(self, model, encoder, channel, decoder, split_index, alfa):

        # Transform each block of the ViT in a delta block that select tokens and update its budget 
        for i, block in enumerate(model.blocks):  

            # Change the attention in order to store the class_tokens attention scores 
            block.attn = Store_Scores_Attention(block.attn)

            # Change the block into a delta block, initializing alfa as the training lr 
            model.blocks[i] = Delta_Block(block, alfa = alfa)

        # Split the original model 
        blocks_before = model.blocks[:split_index]
        blocks_after = model.blocks[split_index:]

        # Store original blocks 
        self.blocks = nn.Sequential(*blocks_before, *blocks_after)

        # Add comm pipeline and compression modules 
        model.blocks = nn.Sequential(*blocks_before, encoder, channel, decoder, *blocks_after)

        return model 

    # Freeze blocks that do not have enough budget and normalize budgets
    def freeze_blocks(self):

        # Get blocks budgets
        budgets = torch.tensor([block.budget for block in self.blocks], dtype=torch.float32)

        # Apply softmax to budgets to get probabilities
        probabilities = torch.nn.functional.softmax(budgets, dim=0)

        # Sample top-K blocks based on probabilities
        sampled_indices = torch.multinomial(probabilities, self.k, replacement=False).tolist()

        # Freeze all blocks except the sampled ones
        for i, block in enumerate(self.blocks):
            for param in block.parameters():
                param.requires_grad = i in sampled_indices  # Only sampled blocks remain trainable
        
        return 

    # Forward 
    def forward(self, x):
        self.freeze_blocks()
        return self.model.forward(x)