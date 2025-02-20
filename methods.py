import torch 
import hydra
from comm_functions import train_backward_communication_pipeline, train_forward_communication_pipeline

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
    def apply_gradient_pipeline(module, grad_output):

        # Apply the channel 
        new_output = channel(grad_output[0])
        
        # Must return as a Tuple so the ","
        return (new_output, )
    
    # Apply channel to the activations 
    def apply_forward_pipeline(module, args, output):

        # Apply the channel
        new_output = channel(output)

        return new_output

    # Register the hooks (we are assuning we have a ViT with blocks modules ) 
    model.blocks[split_index - 1].register_full_backward_pre_hook(apply_gradient_pipeline)
    model.blocks[split_index - 1].register_forward_hook(apply_forward_pipeline)

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
    def apply_forward_pipeline(module, args, output):

        # Apply the channel
        new_output = channel(output)

        return new_output
    
    # Register the hooks (we are assuning we have a ViT with blocks modules )
    model.blocks[split_index - 1].register_forward_hook(apply_forward_pipeline)

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


def delta(cfg):

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)

    # Custom attention function 
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)


            print(attn.shape)

            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    # Replace the attention layer in the model
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
     
     # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

     # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Apply channel to the gradient 
    def apply_gradient_pipeline(module, grad_output):

        # Apply the channel 
        new_output = channel(grad_output[0])
        
        # Must return as a Tuple so the ","
        return (new_output, )
    
    # Apply channel to the activations 
    def apply_forward_pipeline(module, args, output):

        # Apply the channel
        new_output = channel(output)

        return new_output

    # Register the hooks (we are assuning we have a ViT with blocks modules ) 
    model.blocks[split_index - 1].register_full_backward_pre_hook(apply_gradient_pipeline)
    model.blocks[split_index - 1].register_forward_hook(apply_forward_pipeline)

    # Add the channel as an attribute
    model.channel = channel


    return model 
# def apply_DELTA(model): 
    


#     # Hook that updates budget  
#     def update_budget(module, input, output):

#         # Get input and output class tokens, shape batch_size x hidden_dim 
#         input_cls_token = model.pool(input)
#         output_cls_token = model.pool(output)

#         # Compute L2 norm along the hidden dimension (dim=1), and then the mean 
#         delta = torch.norm(input_cls_token - output_cls_token, p=2, dim=1, keepdim=True).mean()

#         # 
#         module.budget += 



#     # Initialize budgets as 1 
#     for block in model.blocks : 
#         setattr(block, "budget", 1)
#         block.register_forward_hook(update_budget)

# # Apply the  DELTA idea to split-learning
# def SL_DELTA(cfg):

#     # Initialize  model  
#     model = hydra.utils.instantiate(cfg.model)
     
#      # Get channel 
#     channel =  hydra.utils.instantiate(cfg.comm.channel)

#      # Get split index 
#     split_index = cfg.hyperparameters.split_index

#     # Apply channel to the gradient 
#     def apply_gradient_pipeline(module, grad_output):

#         # Apply the channel 
#         new_output = channel(grad_output[0])
        
#         # Must return as a Tuple so the ","
#         return (new_output, )
    
#     # Apply channel to the activations 
#     def apply_forward_pipeline(module, args, output):

#         # Apply the channel
#         new_output = channel(output)

#         return new_output
    

#     # Register the hooks (we are assuning we have a ViT with blocks modules )
#     model.blocks[split_index - 1].register_forward_hook(apply_forward_pipeline)
#     model.blocks[split_index - 1].register_full_backward_pre_hook(apply_gradient_pipeline)

#     # Add the channel as an attribute
#     model.channel = channel





























# def split_learning_with_denoising_baseline(cfg):

#     # Train forward communication pipeline  + denoiser
#     forward_communication_pipeline = train_forward_communication_pipeline(cfg)

#     # Train backward communication pipeline  + denoiser 
#     backward_communication_pipeline = train_backward_communication_pipeline(cfg)

#     # Get split index 
#     split_index = cfg.hyperparameters.split_index

#     # Apply channel to the gradient 
#     def apply_gradient_pipeline(module, grad_output):

#         # Apply the channel 
#         new_output = backward_communication_pipeline(grad_output[0])
        
#         # Must return as a Tuple so the ","
#         return (new_output, )
    
#     # Apply channel to the activations 
#     def apply_forward_pipeline(module, args, output):

#         # Apply the channel
#         new_output = forward_communication_pipeline(output)

#         return new_output
    
#     # Initialize  model  
#     model = hydra.utils.instantiate(cfg.model)

#     # Register the hooks (we are assuning we have a ViT with blocks modules )
#     model.blocks[split_index - 1].register_forward_hook(apply_forward_pipeline)
#     model.blocks[split_index - 1].register_full_backward_pre_hook(apply_gradient_pipeline)

#     # Add the channel as an attribute
#     model.channel = channel

#     return model