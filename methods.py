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
    model.blocks[split_index - 1].register_forward_hook(apply_forward_pipeline)
    model.blocks[split_index - 1].register_full_backward_pre_hook(apply_gradient_pipeline)

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


def split_learning_with_denoising_baseline(cfg):

    # Train forward communication pipeline  + denoiser
    forward_communication_pipeline = train_forward_communication_pipeline(cfg)

    # Train backward communication pipeline  + denoiser 
    backward_communication_pipeline = train_backward_communication_pipeline(cfg)

    # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Apply channel to the gradient 
    def apply_gradient_pipeline(module, grad_output):

        # Apply the channel 
        new_output = backward_communication_pipeline(grad_output[0])
        
        # Must return as a Tuple so the ","
        return (new_output, )
    
    # Apply channel to the activations 
    def apply_forward_pipeline(module, args, output):

        # Apply the channel
        new_output = forward_communication_pipeline(output)

        return new_output
    
    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)

    # Register the hooks (we are assuning we have a ViT with blocks modules )
    model.blocks[split_index - 1].register_forward_hook(apply_forward_pipeline)
    model.blocks[split_index - 1].register_full_backward_pre_hook(apply_gradient_pipeline)

    # Add the channel as an attribute
    model.channel = channel

    return model