import torch 
import hydra
import torch.nn as nn
from scripts.utils import train_all, resolve_compression_rate
from scripts.modules import Store_Scores_Attention, Delta_Block, Compressor_Block
import warnings

# ----------- Baselines ----------------------------------------------------

# Normal learning without considering the channel, an upper bound to the whole project 
def classic_training(cfg):

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)
    
    # Store method name 
    model.method = cfg.method.name 

    return model 

# Classic split_learning with a channel that splits the network and autoencoder to compress the things 
def classic_split_learning(cfg):

    # Check if we can train at the current compression 
    resolve_compression_rate(cfg)

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)

    # Get split index 
    split_index = cfg.hyperparameters.split_index
    
    # Get Encoder, Decoder and Channel 
    encoder = hydra.utils.instantiate(cfg.communication.encoder, input_size=model.num_features)
    decoder = hydra.utils.instantiate(cfg.communication.decoder, input_size=2 * encoder.output_size, output_size=model.num_features)
    channel = hydra.utils.instantiate(cfg.communication.channel)

    # Add encoder, decoder  and channel to the model 
    blocks_before = model.blocks[:split_index]
    blocks_after = model.blocks[split_index:]
    model.blocks = nn.Sequential(*blocks_before, encoder, channel, decoder, *blocks_after)

    # Channel and name  attributes   
    model.channel = channel
    model.method = cfg.method.name

    # Set all trainable 
    train_all(model)

    return model   

# Baseline where the whole dataset is compressed and sent to the server where the training happens 
def JPEG(cfg):

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)

    # Store method name 
    model.method = cfg.method.name 

    return model 

# Dynamic Efficient Layer and Tokens Adaptation 
def alast(cfg):
    
    # Check if we can train at the current compression 
    resolve_compression_rate(cfg)

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)

    # Transform each block of the ViT in a delta block that select tokens and update its budget 
    for i, block in enumerate(model.blocks):  

        # Change the attention in order to store the class_tokens attention scores 
        block.attn = Store_Scores_Attention(block.attn)

        # Change the block into a delta block, initializing alfa as the training lr 
        model.blocks[i] = Delta_Block(block, alfa = cfg.optimizer.lr)

    # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Get Encoder, Decoder and Channel 
    encoder = hydra.utils.instantiate(cfg.communication.encoder, input_size=model.num_features)
    decoder = hydra.utils.instantiate(cfg.communication.decoder, input_size=2 * encoder.output_size, output_size=model.num_features)
    channel = hydra.utils.instantiate(cfg.communication.channel)

    # Add encoder, channel and decoder to the model 
    blocks_before = model.blocks[:split_index]
    blocks_after = model.blocks[split_index:]
    model.blocks = nn.Sequential(*blocks_before, encoder, channel, decoder, *blocks_after)

    # Save the original blocks of the vit to freeze them 
    model.raw_blocks = nn.Sequential(*blocks_before, *blocks_after)

    # Freeze blocks that do not have enough budget and normalize budgets
    def freeze_blocks(module, input):
        # Get blocks budgets
        budgets = torch.tensor([block.budget for block in model.raw_blocks], dtype=torch.float32)

        # Apply softmax to budgets to get probabilities
        probabilities = torch.nn.functional.softmax(budgets, dim=0)

        # Sample top-K blocks based on probabilities
        K = model.K
        sampled_indices = torch.multinomial(probabilities, K, replacement=False).tolist()

        # Freeze all blocks except the sampled ones
        for i, block in enumerate(model.raw_blocks):
            for param in block.parameters():
                param.requires_grad = i in sampled_indices  # Only sampled blocks remain trainable
        
        return 

    # Register the mechanism to freeze blocks before the mmodel 
    model.register_forward_pre_hook(freeze_blocks)

    # Set the number of training blocks
    model.K = cfg.method.parameters.K
    
    # Channel and name attributes   
    model.channel = channel
    model.method = cfg.method.name

    # Set all trainable 
    train_all(model)

    return model

# Ours static 
def ours(cfg):

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)

    # # Get compression rates 
    # batch_compression, token_compression = resolve_compression_rate(cfg)

    # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Get Encoder, Decoder and Channel 
    encoder = hydra.utils.instantiate(cfg.communication.encoder, input_size=model.num_features)
    decoder = hydra.utils.instantiate(cfg.communication.decoder, input_size=2 * encoder.output_size, output_size=model.num_features)
    channel = hydra.utils.instantiate(cfg.communication.channel)

    # Transform inot compressor block 
    model.blocks[split_index -1].attn = Store_Scores_Attention(model.blocks[split_index -1].attn)
    model.blocks[split_index -1] = Compressor_Block(model.blocks[split_index -1],
                                                    batch_compression_rate =  cfg.method.parameters.batch_compression_rate,
                                                    tokens_compression_rate = cfg.method.parameters.token_compression_rate)
    
    model.compressor_module = model.blocks[split_index -1]

    # Add encoder and decoder to the model 
    blocks_before = model.blocks[:split_index]
    blocks_after = model.blocks[split_index:]
    model.blocks = nn.Sequential(*blocks_before, encoder, channel, decoder, *blocks_after)

    # Channel and name attributes   
    model.channel = channel
    model.method = cfg.method.name

    # Set all trainable 
    train_all(model)

    return model
