import torch 
import hydra
from comm_functions import train_backward_communication_pipeline, train_forward_communication_pipeline
import math 

# This baseline consists in sending trough the channel the raw data (images) adding noise 
def send_raw_data_baseline(cfg):

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)

    # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

    # Add the channel before the model 
    model = torch.nn.Sequential(channel, model)

    # Add the channel as an attribute
    model.channel = channel

    return model 

# This baseline consists in splitting the network in a certain split_index and just add noise to gradients and activations 
def simple_split_learning_baseline(cfg):


    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)
     
     # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

     # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Apply channel to the gradient 
    def backward_channel(module, grad_output):

        # Apply the channel 
        new_output = channel(grad_output[0])
        
        # Must return as a Tuple so the ","
        return (new_output, )
    
    # Apply channel to the activations 
    def forward_channel(module, args, output):

        # Apply the channel
        new_output = channel(output)

        return new_output

    # Register the hooks (we are assuning we have a ViT with blocks modules ) 
    model.blocks[split_index - 1].register_full_backward_pre_hook(backward_channel)
    model.blocks[split_index - 1].register_forward_hook(forward_channel)

    # Add the channel as an attribute
    model.channel = channel


    return model 

# This baseline consists in splitting the network in a certain split_index and just add noise to  activations (not training the edge device) 
def only_forward_split_learning_baseline(cfg):

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)
     
     # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

     # Get split index 
    split_index = cfg.hyperparameters.split_index

    
    # Apply channel to the activations 
    def forward_channel(module, args, output):

        # Apply the channel
        new_output = channel(output)

        return new_output
    
    # Register the hooks (we are assuning we have a ViT with blocks modules )
    model.blocks[split_index - 1].register_forward_hook(forward_channel)

    # Add the channel as an attribute
    model.channel = channel

    # Freeze initial encoding layers 
    for name, param in model.named_parameters():
        if name in ["cls_token", "pos_embed", "patch_embed.proj.weight", "patch_embed.proj.bias"]:
            param.requires_grad = False

    # Freeze every block  before the split_index 
    for block in model.blocks[:split_index]:
        for param in block.parameters():
            param.requires_grad = False
    


    return model 

# This baseline is built on the simple split learning scenario, where the edge device model retains just the most important tokens 
def delta(cfg):

    # Get parameters 
    alfa = cfg.method.parameters.alfa
    percentage = cfg.method.parameters.percentage


    def make_decision():

        # Get delta importances of each block 
        importances = []
        for block in model.blocks:
            importances.append(block.importance)
        
        # Compute the relative importances 
        edge_importance = sum(importances[:split_index]) 
        total_importance = sum(importances)
        importance_ratio = edge_importance / total_importance + model.time_elapsed * 0.01
        layers_ratio = split_index / len(importances)

        print(importance_ratio)
        print(layers_ratio)

        decision = importance_ratio >= layers_ratio

        if decision:
            model.time_elapsed = 0
        else:
            model.time_elapsed += 1 

        print(model.time_elapsed)
        

        for block in model.blocks[:split_index]:
            for param in block.parameters():
                param.requires_grad = decision

        model.decisions_list.append(model.decision)
        return decision 

    # Apply channel to  the gradient 
    def backward_channel(module, grad_output):

        # Update the number of iterations 
        model.iterations += 1
        if model.iterations % 5 == 0 :
            model.decision = make_decision()

        grad_output = channel(grad_output[0])

        return (grad_output, )
            
    # Apply channel to the activations 
    def forward_channel(module, input, output):
        # Apply the channel
        new_output = channel(output)

        return new_output
    
    # Delta forward hook 
    def remove_tokens(module, input, output): 
        B, N, C = output.shape
        class_tkn_attention = module.attn.class_token_attention

        # Calculate percentage of token to keep at each block in order to reach the percentage goal 
        block_percentage = percentage **(1 /  split_index)
        
        # Compute the number of tokens to keep (excluding class token)
        num_tokens_to_keep = int(block_percentage * (N - 1))
        
        # Sort indices based on entropy (excluding class token)
        sorted_indices = torch.argsort(class_tkn_attention[:, 1:], dim=1, descending=True)
        
        # Select top-k tokens per batch
        selected_indices = sorted_indices[:, :num_tokens_to_keep] + 1  # Shift to account for class token

        # Concatenate class token with selected tokens
        batch_indices = torch.arange(B, device=output.device).unsqueeze(1)

        output = torch.cat([output[:, :1, :], output[batch_indices, selected_indices, :], ], dim=1)

        return output 
    
        # Apply channel to the activations 
    
    
    def update_importance(module, input, output):

        if any(param.requires_grad for param in module.parameters()):
            # Get input and output class tokens 
            input_class_token = input[0]
            output_class_token = output

            # Compute the new delta 
            new_cls_token_delta = torch.norm(input_class_token - output_class_token, p=2).item()

            # If already done one forward compute the importance 
            if module.previous_cls_token_delta != None :
                new_importance = abs(new_cls_token_delta -  module.previous_cls_token_delta)
                module.importance += alfa * (new_importance - module.importance)

            # Update previous_cls_token_delta
            module.previous_cls_token_delta = new_cls_token_delta

        return output
    # This custom attention functions adds for each module a "class_token_attention" variable, the forward is modified just to store it.
    class CustomAttention(torch.nn.Module):
        def __init__(self, num_heads, head_dim, attn_drop, q_norm, k_norm, scale, qkv, proj, proj_drop):
            super().__init__()

            self.num_heads = num_heads
            self.head_dim = head_dim
            self.attn_drop = attn_drop
            self.q_norm = q_norm
            self.k_norm = k_norm
            self.scale = scale
            self.qkv = qkv
            self.proj = proj
            self.proj_drop = proj_drop
            self.class_token_attention = None 

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Same forward of the original class 
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn) 
            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            # Get the class token, take the average over the heads and store it 
            self.class_token_attention = attn[:, :, 0, :].mean(dim=1) 

            return x
        
    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)
    
     # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

     # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Replace the attention layers in the model for blocks before the splitting index 
    for block in model.blocks:
        block.attn = CustomAttention(
            num_heads=block.attn.num_heads,
            head_dim=block.attn.head_dim,
            attn_drop=block.attn.attn_drop,
            q_norm=block.attn.q_norm,
            k_norm=block.attn.k_norm,
            scale = block.attn.scale,
            qkv = block.attn.qkv,
            proj = block.attn.proj,
            proj_drop = block.attn.proj_drop)
        block.previous_cls_token_delta = None 
        block.importance = 1
        block.register_forward_hook(update_importance)

    # Register the channel hooks 
    model.blocks[split_index - 1].register_full_backward_pre_hook(backward_channel)
    model.blocks[split_index - 1].register_forward_hook(forward_channel)
    for block in model.blocks[:split_index]:
        block.register_forward_hook(remove_tokens)

    # Add the channel as an attribute
    model.channel = channel
    model.iterations = 0

    # Variable to store the last loss 
    model.decisions_list = []
    model.decision = 1
    model.time_elapsed = 0

    return model 
