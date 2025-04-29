import torch 
import hydra
import torch.nn.functional as F
import torch.nn as nn

from scripts.channel import AWGN
from scripts.utils import freeze_edge, train_all
import random 



# ----------- Baselines ----------------------------------------------------


# Normal learning without considering the channel, an upper bound to the project 
def classic_training(cfg):

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)
    
    # Store method name 
    model.method = cfg.method.name 

    return model 

# Classic split_learning with a channel that splits the network and autoencoder to compress the things 
def classic_split_learning(cfg):

    # Apply channel to the gradient 
    def backward_channel(module, grad_output):
        if model.training:
            # Apply the channel 
            grad_output = channel(grad_output[0])

       # Must return as a Tuple so the ","
        return (grad_output, )

    # Apply channel to the activations 
    def forward_channel(module, args, output):
        if model.training:
            # Apply the channel
            output = channel(output)

        return output

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)

    # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Get Encoder, Decoder and Channel 
    encoder = hydra.utils.instantiate(cfg.communication.encoder, input_size=model.num_features)
    decoder = hydra.utils.instantiate(cfg.communication.decoder, input_size=2 * encoder.output_size, output_size=model.num_features)
    channel = hydra.utils.instantiate(cfg.communication.channel)

    channel = AWGN(snr=10)

    # Register the hooks on the encoder 
    # encoder.register_full_backward_pre_hook(backward_channel)
    # encoder.register_forward_hook(forward_channel)

    # Add encoder and decoder to the model 
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
def delta(cfg):

    # Apply channel to the gradient 
    def backward_channel(module, grad_output):
        if model.training:
            # Apply the channel 
            grad_output = channel(grad_output[0])
        
        # Must return as a Tuple so the ","
        return (grad_output, )
    
    # Apply channel to the activations 
    def forward_channel(module, args, output):
        if model.training:
            # Apply the channel
            output = channel(output)

        return output

    # Token selection mechanism  
    def select_tokens(module, input, output):

        if model.training:

            # Get shape of the output 
            B, N, C = output.shape

            # Get class token attention scores 
            class_tkn_attention = module.attn.class_token_attention

            # Get the percentage of blocks to retain as the block budget  
            block_percentage = module.budget
            
            # Compute the number of tokens to keep (excluding class token)
            num_tokens_to_keep = int(block_percentage * N)
            
            # Sort indices based on entropy (excluding class token)
            sorted_indices = torch.argsort(class_tkn_attention[:, 1:], dim=1, descending=True)
            
            # Select top-k tokens per batch
            selected_indices = sorted_indices[:, :num_tokens_to_keep] + 1  # Shift to account for class token

            # Concatenate class token with selected tokens
            batch_indices = torch.arange(B, device=output.device).unsqueeze(1)

            output = torch.cat([output[:, :1, :], output[batch_indices, selected_indices, :], ], dim=1)

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

     # Function to merge the batches    
    
    # Update budget for blocks modules 
    def update_budget(module, input, output):

        # Get input class token 
        input_class_token = model.pool(input[0])

        # Get output class token 
        output_class_token = model.pool(output)

        # Get current budget 
        current_budget = module.budget 

        # Get previous delta
        previous_delta  = module.delta

        # Get new delta 
        new_delta = (torch.norm(output_class_token - input_class_token, p=2).item())**2

        # Store it
        module.delta = new_delta 

        # Update the budget
        new_budget =  current_budget + (new_delta - previous_delta) * model.alfa

        # Clip it between 0.5 and 1 and  store it 
        module.budget =  max(0.5, min(new_budget, 1)) 

        return output
    
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

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)

    # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Get Encoder, Decoder and Channel 
    encoder = hydra.utils.instantiate(cfg.communication.encoder, input_size=model.num_features)
    decoder = hydra.utils.instantiate(cfg.communication.decoder, input_size=2 * encoder.output_size, output_size=model.num_features)
    channel = hydra.utils.instantiate(cfg.communication.channel)

    # Register the hooks on the encoder 
    encoder.register_full_backward_pre_hook(backward_channel)
    encoder.register_forward_hook(forward_channel)


    # Add encoder and decoder to the model 
    blocks_before = model.blocks[:split_index]
    blocks_after = model.blocks[split_index:]
    model.blocks = nn.Sequential(*blocks_before, encoder, decoder, *blocks_after)
    model.raw_blocks = nn.Sequential(*blocks_before, *blocks_after)

    # Channel and name attributes   
    model.channel = channel
    model.method = cfg.method.name

    # Set all trainable 
    train_all(model)

    # For each block register the budget update and token selection mechanism 
    for block in model.raw_blocks:  

        # Change the attention in order to store the class_tokens attention scores 
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

        # Initialize the budget to 1 and the delta to 0 
        block.budget = 0.9
        block.delta = 0

        # Register the budget update  and token selection mechanism
        block.register_forward_hook(update_budget)
        block.register_forward_hook(select_tokens)

    # Register the mechanism to freeze blocks before the mmodel 
    model.register_forward_pre_hook(freeze_blocks)

    # Set the number of training blocks
    model.K = cfg.method.parameters.K
    
    # Initialize the alfa as the training leerning rate 
    model.alfa = cfg.optimizer.lr

    # Store the name of the method in the model
    model.method = cfg.method.name

    return model

# Ours static 
def ours_static(cfg):

    # Galore pipeline 
    def svd_channel_reconstruct(G, channel):

        # Update projection matrices every T times 
        if model.t % model.update_frequency == 0:
  
            # Get the average gradient 
            G_mean = G.mean(dim=0)

            # Wide matrix: use the right singular vectors.
            U, S, V = torch.linalg.svd(G_mean, full_matrices=False) 
            # U: [m, min(m,n)], Vh: [min(m,n), n]

            energy = S.pow(2)
            total_energy = energy.sum()
            cumulative_energy = torch.cumsum(energy, dim=0)
            r = (cumulative_energy < (cfg.method.parameters.energy_threshold * total_energy)).sum().item() + 1
            
                
            # Store left projection matrices
            model.server_left_proj_matrix = U[:, :r]
            model.device_left_proj_matrix = channel(model.server_left_proj_matrix)

            # Store right projection matrices
            model.server_right_proj_matrix = V[:r, :]
            model.device_right_proj_matrix = channel(model.server_right_proj_matrix)


        # Compute the projected representation  
        G =  torch.matmul(torch.matmul(model.server_left_proj_matrix.t().unsqueeze(0), G ), model.server_right_proj_matrix.t().unsqueeze(0))

        # Apply the channel 
        G= channel(G)

        # Reconstruct the gradient
        G = torch.matmul(torch.matmul(model.server_left_proj_matrix.unsqueeze(0), G ), model.server_right_proj_matrix.unsqueeze(0))

        model.t += 1 

        return G

    # Select tokens 
    def select_tokens(module, input, output):

        if model.training and model.tokens_compression_percentage < 1:
            B, N, C = output.shape
            class_tkn_attention = module.attn.class_token_attention

            # Calculate percentage of tokens to keep at each block in order to reach the percentage goal
            block_percentage = model.tokens_compression_percentage ** (1 / split_index)
            
            # Compute the number of tokens to keep (excluding class token)
            num_tokens_to_keep = int(block_percentage * N)
            
            # Compute average attention across the batch (excluding class token)
            avg_attention = class_tkn_attention[:, 1:].mean(dim=0)  # Shape: (N-1,)
            sorted_indices = torch.argsort(avg_attention, descending=True)
            selected_indices = sorted_indices[:num_tokens_to_keep] + 1  # Shift to account for class token
            selected_indices = selected_indices.unsqueeze(0).expand(B, -1)  # Use same indices for all samples
        
            # Concatenate class token with selected tokens
            batch_indices = torch.arange(B, device=output.device).unsqueeze(1)
            output = torch.cat([output[:, :1, :], output[batch_indices, selected_indices, :]], dim=1)
        
        return output
    
    # Function that merges batches with the same labels.
    def merge_batch(batch, labels, compression_rate):

        # Get the unique labels and the number of unique labels
        unique_labels = torch.unique(labels)

        # Create variables for the new batch and labels 
        merged_batches = []
        merged_labels = []

        # Compress same label inputs 
        for lab in unique_labels:
            # Select samples for the current label.
            group_idx = (labels == lab).nonzero(as_tuple=True)[0]
            group = batch[group_idx]
            num_samples = group.shape[0]

            # Determine the target number of samples for this group.
            target_samples = max(1, round(num_samples * compression_rate))

            # Partition the group into approximately equal chunks.
            chunks = torch.chunk(group, target_samples, dim=0)
            avg_chunks = torch.stack([chunk.mean(dim=0) for chunk in chunks])
            merged_batches.append(avg_chunks)
            merged_labels.append(torch.full((avg_chunks.shape[0],), lab, dtype=labels.dtype, device=labels.device))

        # Concatenate the merged groups from all labels (so far)
        merged_batch = torch.cat(merged_batches, dim=0)
        merged_labels = torch.cat(merged_labels, dim=0)

        return merged_batch, merged_labels
  
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

    # Apply channel to the gradient 
    def backward_channel(module, grad_output):
        if model.training:
            # Apply the channel 
            grad_output = svd_channel_reconstruct(grad_output[0], model.channel)
            
        # Must return as a Tuple so the ","
        return (grad_output, )
    
    # Apply channel to the activations 
    def forward_channel(module, args, output):

        if model.training and model.batch_compression_rate  < 1.0:
            output, model.labels = merge_batch(output, model.labels, model.batch_compression_rate)

            # Apply the channel
            output = channel(output)


        return output

        # Delta forward hook 
    
    # Freeze the network sometimes 
    def random_freeze(module, output):
        if random.random() < model.freeze_probability:
            freeze_edge(model, split_index)
        else:
            train_all(model)
        return output 
    

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)

    # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Get Encoder, Decoder and Channel 
    encoder = hydra.utils.instantiate(cfg.communication.encoder, input_size=model.num_features)
    decoder = hydra.utils.instantiate(cfg.communication.decoder, input_size=2 * encoder.output_size, output_size=model.num_features)
    channel = hydra.utils.instantiate(cfg.communication.channel)

    # Register the hooks on the encoder 
    encoder.register_full_backward_pre_hook(backward_channel)
    encoder.register_forward_hook(forward_channel)

    # Channel and name attributes   
    model.channel = channel
    model.method = cfg.method.name

    # Set all trainable 
    train_all(model)

    # Replace the attention layers in the model for blocks before the splitting index 
    for block in model.blocks[:split_index]:
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
        block.register_forward_hook(select_tokens)
    
    # Add encoder and decoder to the model 
    blocks_before = model.blocks[:split_index]
    blocks_after = model.blocks[split_index:]
    model.blocks = nn.Sequential(*blocks_before, encoder, decoder, *blocks_after)

    # Initialize methods needed variables 
    model.t = 0
    model.server_left_proj_matrix = 0
    model.server_right_proj_matrix = 0
    model.device_left_proj_matrix = 0
    model.device_right_proj_matrix = 0
    model.labels = 0

    # Get compression percentages 
    model.compression = cfg.method.parameters.compression  
    model.tokens_compression_percentage = model.compression **(1/2)
    model.batch_compression_rate = model.compression **(1/2)
    model.update_frequency = cfg.method.parameters.update_frequency

    # Freeze percentage
    model.freeze_probability = cfg.method.parameters.freeze_probability

    # Register the hooks 
    model.register_forward_pre_hook(random_freeze)

    return model


# ----------- Experiments ----------------------------------------------------

# Static batch merging based on a percentage before the split
def batch_compression(cfg):

    # Apply channel to the gradient 
    def backward_channel(module, grad_output):

        # Apply the channel 
        new_output = channel(grad_output[0])
        
        # Must return as a Tuple so the ","
        return (new_output, )
    
    # Apply channel to the activations 
    def forward_channel(module, args, output):
        if model.training and model.compression_rate < 1.0:
            output, model.labels = merge_batch(output, model.labels, model.compression_rate)

        # Apply the channel
        new_output = channel(output)

        return new_output
    
    # Merge batches 
    def merge_batch(batch, labels, compression_rate):
        # Get the shape of the batch
        batch_size, _, _ = batch.shape

        # Get the unique labels and the number of unique labels
        unique_labels = torch.unique(labels)
        number_of_unique_labels = len(unique_labels)

        # Compute the max compression obtainable compressing same label inputs 
        max_compression_rate = number_of_unique_labels / batch_size 

        # Create a variable that is true if we can achieve the needed compression rate just with same label compression 
        just_same_label_compression = int(max_compression_rate < compression_rate)

        # Create variables for the new batch and labels 
        merged_batches = []
        merged_labels = []

        # Compress same label inputs 
        for lab in unique_labels:
            # Select samples for the current label.
            group_idx = (labels == lab).nonzero(as_tuple=True)[0]
            group = batch[group_idx]
            num_samples = group.shape[0]

            # Determine the target number of samples for this group.
            target_samples = max(1, round(num_samples * compression_rate) * just_same_label_compression)

            # Partition the group into approximately equal chunks.
            chunks = torch.chunk(group, target_samples, dim=0)
            avg_chunks = torch.stack([chunk.mean(dim=0) for chunk in chunks])
            merged_batches.append(avg_chunks)
            merged_labels.append(torch.full((avg_chunks.shape[0],), lab, dtype=labels.dtype, device=labels.device))

        # Concatenate the merged groups from all labels (so far)
        merged_batch = torch.cat(merged_batches, dim=0)
        merged_labels = torch.cat(merged_labels, dim=0)


        # If same-label compression is not enough, apply cross-label compression
        if not just_same_label_compression:

            # Calculate the target final size of the batch
            target_size = int(batch_size * compression_rate)

            # # Create one-hot encoded labels
            merged_labels = F.one_hot(merged_labels, num_classes=cfg.dataset.num_classes).float()

            # Split into target number of chunks
            chunks = torch.chunk(merged_batch, target_size, dim=0)
            label_chunks = torch.chunk(merged_labels, target_size, dim=0)

            # Average each chunk
            merged_batch = torch.stack([chunk.mean(dim=0) for chunk in chunks])
            merged_labels = torch.stack([chunk.mean(dim=0) for chunk in label_chunks])  # Soft labels

        final_batch_size, _, _ = merged_batch.shape
        model.actual_compression.append(final_batch_size / batch_size) 

        print("Achieved compression:",  sum(model.actual_compression) / len(model.actual_compression))

        return merged_batch, merged_labels

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)
    
     # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

     # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Apply the channel to both forward and backward 
    model.blocks[split_index - 1].register_full_backward_pre_hook(backward_channel)
    model.blocks[split_index - 1].register_forward_hook(forward_channel)

    # Add the channel as an attribute
    model.channel = channel

    # Initialize the labels 
    model.labels = 0

    # Save the compression rate 
    model.compression_rate = cfg.method.parameters.compression_rate
    
    # Save the method name 
    model.method = cfg.method.name 

    model.actual_compression = []

    return model 

# This method uses attention to merge different batches into the same one, doesn't really work...  
def differentiable_batch_averaging(cfg):

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
    
    # A module that combines batches into a smaller dimension using an attention mechanism 
    class Attention_Batch_Compressor(nn.Module):
        def __init__(self, batch_size, num_classes, model, device):
            super().__init__()
            self.batch_size = batch_size
            self.num_queries = round(model.compression_rate * batch_size)
            self.query = nn.Parameter(torch.randn(self.num_queries, model.embed_dim).to(device))
            self.key_projection = nn.Linear(model.embed_dim, model.embed_dim).to(device)
            self.num_classes = num_classes

        def entropy_loss(self, attn_weights):
            # attn_weights: (seq_len, batch)
            eps = 1e-8

            # Compute sparsity loss 
            attn_weights_ent = attn_weights * torch.log(attn_weights + eps)
            sparsity_loss = -attn_weights_ent.sum(dim=0).mean()
            
            # Compute coverage loss 
            batch_average_attn_weights = attn_weights.mean(dim=0)
            batch_average_attn_weights_ent = batch_average_attn_weights * torch.log(batch_average_attn_weights + eps)

            coverage_loss = batch_average_attn_weights_ent.sum()

            # Final loss 
            final_loss =  0.1 * sparsity_loss + 1 * coverage_loss

            return   final_loss 
        
        def forward(self, activations):
            if model.training:
                """
                activations: (B, D, L) - batch of activations
                labels: (B,) - class indices
                """
                # Get activations shape 
                B, D, L = activations.shape

                # Derive class tokens from activations 
                class_tokens = activations[:,0,:]  

                # Project class tokens to get keys
                keys = self.key_projection(class_tokens)  # (B, D)

                # Compute attention scores between queries and keys
                attn_scores = torch.matmul(self.query, keys.T)  # (Q, B)
                attn_weights = F.softmax(attn_scores, dim=-1)   # (Q, B)

                # Transpose activations to (B, D*L)
                activations_flat = activations.view(B, -1)

                # Merge activations
                merged_activations = torch.matmul(attn_weights, activations_flat)  # (Q, D*L)
                merged_activations = merged_activations.view(self.num_queries, D, L)


                model.loss = self.entropy_loss(attn_weights)
 
                # Merge labels using attention
                with torch.no_grad():
                    one_hot_labels = F.one_hot(model.labels, num_classes=self.num_classes).float()  # (B, C)
                    model.labels = torch.matmul(attn_weights, one_hot_labels)  # (Q, C)

                return merged_activations
            else:
                return activations

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)
   
    # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

    # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Add the channel as an attribute
    model.channel = channel

    # Initialize the labels 
    model.labels = 0

    # Save the compression rate 
    model.compression_rate = cfg.method.parameters.compression_rate

    # Save the method name 
    model.method = cfg.method.name 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.loss = 0 

    # Initialize batch_compressor
    batch_compressor = Attention_Batch_Compressor(cfg.dataset.batch_size, cfg.dataset.num_classes, model,  device)

    # Apply the channel to both forward and backward
    model.blocks[split_index - 1] = torch.nn.Sequential(model.blocks[split_index - 1], batch_compressor)
    model.blocks[split_index - 1].register_full_backward_pre_hook(backward_channel)
    model.blocks[split_index - 1].register_forward_hook(forward_channel)

    return model 

# This approach selects one of three states to train dynamically between: train whole network, train just server, train with stored activations 
def adaptive_freezing(cfg):

    # Particular forward for the "C" state 
    def stored_activations_forward():

        x = model.stored_activations
        for block in model.blocks[split_index:]:
            x = block(x)
        x = model.norm(x)
        x = model.forward_head(x)
        return x
    
    # Function that updates the score of the current state based on the loss 
    def update_score(module, grad_output):

        # Get delta loss 
        delta_loss = model.last_losses[0] - model.last_losses[1]

        # Update the score 
        model.states_scores[model.state] +=  model.alpha * (delta_loss.item() - model.states_scores[model.state])

        if model.i % 3 == 0: 
            update_state()
            
        model.i += 1 

    # Function that updates the states based on a threshold  
    def update_state():
        # If the score of the current state is above the threshold go to the next one 
        if model.states_scores[model.state] > model.score_thresh / (model.state + 1):
            model.state += 1
            if model.state == 1:
                freeze_edge(model, split_index)
        else:
        # Else go back to the previous one 
            model.state -= 1
            if model.state == 0:
                train_all(model)
            
        # Make sure it is between 0 and 2 
        model.state = max(0, min(model.state, 2))

    # Apply channel to the gradient 
    def backward_channel(module, grad_output):
        # Apply the channel
        grad_output = channel(grad_output[0])

        return (grad_output, )
            
    # Apply channel to the activations 
    def forward_channel(module, input, output):
        # Apply the channel
        output = channel(output)

        # If we are in the "C" state store the noisy output as activations
        model.stored_activations = output
        return output
        
    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)
    
    # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

    # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Register the channel hooks 
    model.blocks[split_index - 1].register_full_backward_pre_hook(backward_channel)
    model.blocks[split_index - 1].register_forward_hook(forward_channel)
    model.blocks[split_index +2].register_full_backward_pre_hook(update_score)

    # Add the channel as an attribute
    model.channel = channel

    # Variable to store the last loss 
    model.last_losses = [0]
    model.state = 0
    model.states_scores = [0, 0, 0]
    model.score_thresh = cfg.method.parameters.threshold
    model.alpha = cfg.method.parameters.alpha

    model.stored_activations_forward = stored_activations_forward
    model.method = cfg.method.name 
    model.i = 0
    return model 

# GALORE 
def galore(cfg):

    def svd_channel_reconstruct(G, channel):

        # Update projection matrices every T times 
        if model.t % cfg.method.parameters.switching_frequency == 0:

            # Get the average gradient 
            G_mean = G.mean(dim=0)

            # Wide matrix: use the right singular vectors.
            U, S, V = torch.linalg.svd(G_mean, full_matrices=False) 
            # U: [m, min(m,n)], Vh: [min(m,n), n]

            energy = S.pow(2)
            total_energy = energy.sum()
            cumulative_energy = torch.cumsum(energy, dim=0)
            r = (cumulative_energy < (cfg.method.parameters.energy_threshold * total_energy)).sum().item() + 1

            # Store left projection matrices
            model.server_left_proj_matrix = U[:, :r]
            model.device_left_proj_matrix = channel(model.server_left_proj_matrix)

            # Store right projection matrices
            model.server_right_proj_matrix = V[:r, :]
            model.device_right_proj_matrix = channel(model.server_right_proj_matrix)

        # Compute the projected representation  
        G =  torch.matmul(torch.matmul(model.server_left_proj_matrix.t().unsqueeze(0), G ), model.server_right_proj_matrix.t().unsqueeze(0))

        # Apply the channel 
        G= channel(G)

        # Reconstruct the gradient
        G = torch.matmul(torch.matmul(model.server_left_proj_matrix.unsqueeze(0), G ), model.server_right_proj_matrix.unsqueeze(0))
        model.t+=1
      
        return G

    # Apply channel to the gradient 
    def backward_channel(module, grad_output):

        # Apply Galore pipeline and channel  
        new_output = svd_channel_reconstruct(grad_output[0], channel)

        # Must return as a Tuple so the ","
        return (new_output, )
    
    # Apply channel to the activations 
    def forward_channel(module, args, output):

        # Apply the channel
        new_output = channel(output)

        return new_output

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)
    
    # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

    # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Register the hooks (we are assuning we have a ViT with blocks modules ) 
    model.blocks[split_index - 1].register_full_backward_pre_hook(backward_channel)
    model.blocks[split_index - 1].register_forward_hook(forward_channel)

    # Add the channel as an attribute
    model.channel = channel

    # Store method name 
    model.method = cfg.method.name 

    # Initialize methods parameters 
    model.t = 0
    model.server_left_proj_matrix = 0
    model.server_right_proj_matrix = 0
    model.device_left_proj_matrix = 0
    model.device_right_proj_matrix = 0
    return model 








