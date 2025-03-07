import torch 
import hydra
from comm_functions import train_backward_communication_pipeline, train_forward_communication_pipeline
import torch.nn.functional as F
from utils import freeze_edge, train_all

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

    model.method = cfg.method.name 
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
        new_output = channel(output, model.training)

        return new_output

    # Register the hooks (we are assuning we have a ViT with blocks modules ) 
    model.blocks[split_index - 1].register_full_backward_pre_hook(backward_channel)
    model.blocks[split_index - 1].register_forward_hook(forward_channel)

    # Add the channel as an attribute
    model.channel = channel

    model.method = cfg.method.name 
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
    

    model.method = cfg.method.name 
    return model 

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
        if model.training:
            output, model.labels = merge_batch(output, model.labels)

        # Apply the channel
        new_output = channel(output)

        return new_output
    
    # Merge batches with the same label
    def merge_batch(batch, labels, compression_rate=1.0):

        # If no compression is desired, return the original batch.
        if compression_rate >= 1.0:
            return batch, labels

        unique_labels = torch.unique(labels)
        merged_batches = []
        merged_labels = []

        for lab in unique_labels:
            # Select samples for the current label.
            group_idx = (labels == lab).nonzero(as_tuple=True)[0]
            group = batch[group_idx]
            num_samples = group.shape[0]
            # Determine the target number of samples for this group.
            target_samples = max(1, int(num_samples * compression_rate))
            
            if target_samples >= num_samples:
                # If the target is equal or larger, keep the group unchanged.
                merged_batches.append(group)
                merged_labels.append(torch.full((num_samples,), lab, 
                                                dtype=labels.dtype, device=labels.device))
            else:
                # Partition the group into approximately equal chunks.
                # Each chunk will be averaged to represent that segment.
                chunks = torch.chunk(group, target_samples, dim=0)
                avg_chunks = torch.stack([chunk.mean(dim=0) for chunk in chunks])
                merged_batches.append(avg_chunks)
                merged_labels.append(torch.full((target_samples,), lab, 
                                                dtype=labels.dtype, device=labels.device))
        
        # Concatenate the merged groups from all labels.
        merged_batch = torch.cat(merged_batches, dim=0)
        merged_labels = torch.cat(merged_labels, dim=0)
        
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

    return model 

# Static token selection
def token_selection(cfg):

    # Apply channel to  the gradient 
    def backward_channel(module, grad_output):
        # Apply the channel
        grad_output = channel(grad_output[0])

        return (grad_output, )
            
    # Apply channel to the activations 
    def forward_channel(module, input, output):
        # Apply the channel
        output = channel(output)

        return output 

    # Select tokens 
    def select_tokens(module, input, output):

        B, N, C = output.shape
        class_tkn_attention = module.attn.class_token_attention

        if model.single_selection:
            block_percentage = model.percentage
        else:
            # Calculate percentage of token to keep at each block in order to reach the percentage goal 
            block_percentage = model.percentage **(1 /  split_index)
        
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
  
    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)
    
     # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

     # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Register the channel hooks 
    model.blocks[split_index - 1].register_full_backward_pre_hook(backward_channel)
    model.blocks[split_index - 1].register_forward_hook(forward_channel)

    model.single_selection = cfg.method.parameters.single_selection

    if model.single_selection:
        model.blocks[split_index - 1].register_forward_hook(select_tokens)

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

        if not model.single_selection:
            block.register_forward_hook(select_tokens)
    
    

    # Add the channel as an attribute
    model.channel = channel

    # Store the method name 
    model.method = cfg.method.name 

    
    model.percentage = cfg.method.parameters.percentage

    return model 

# This is based on the paper with the 3 states 
def adaptive_freezing(cfg):

    # Particular forward for the "C" state 
    def stored_activations_forward():

        x = model.stored_activations
        for block in model.blocks[split_index:]:
            x = block(x)
        x = model.norm(x)
        x = model.forward_head(x)
        return x
    
    def update_score(module, grad_output):

        # Get delta loss 
        delta_loss = model.last_losses[0] - model.last_losses[1]
        print(model.states_scores)

        # Update the score 
        model.states_scores[model.state] +=  model.alpha * (delta_loss.item() - model.states_scores[model.state])

        if model.i % 3 == 0: 
            update_state()
        model.i += 1 

    # Function that defines the states 
    def update_state():
        # Nice 
        if model.states_scores[model.state] > model.score_thresh / (model.state + 1):
            model.state += 1
            if model.state == 1:
                freeze_edge(model, split_index)
        else:
            model.state -= 1
            if model.state == 0:
                train_all(model)
                

        # Make sure it is between 0 and 2 
        model.state = max(0, min(model.state, 2))

    # Apply channel to  the gradient 
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

# This baseline combines the freeze / token_selection / batch_compression baselines 
def combined(cfg):

    # Apply channel to the gradient 
    def backward_channel(module, grad_output):

        # Apply the channel 
        new_output = channel(grad_output[0])
        
        # Must return as a Tuple so the ","
        return (new_output, )
    
    # Apply channel to the activations 
    def forward_channel(module, args, output):
        if model.training:
            output, model.labels = merge_batch(output, model.labels, model.batch_compression_rate)

        model.stored_activations = output
        # Apply the channel
        new_output = channel(output, model.training)

        return new_output

        # Delta forward hook 
    
    
    def select_tokens(module, input, output):

        B, N, C = output.shape
        class_tkn_attention = module.attn.class_token_attention


        block_percentage = model.tokens_percentage

        
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
    
        # Merge batches with the same label
    
    def merge_batch(batch, labels, compression_rate=1.0):

        # If no compression is desired, return the original batch.
        if compression_rate >= 1.0:
            return batch, labels

        unique_labels = torch.unique(labels)
        merged_batches = []
        merged_labels = []

        for lab in unique_labels:
            # Select samples for the current label.
            group_idx = (labels == lab).nonzero(as_tuple=True)[0]
            group = batch[group_idx]
            num_samples = group.shape[0]
            # Determine the target number of samples for this group.
            target_samples = max(1, int(num_samples * compression_rate))
            
            if target_samples >= num_samples:
                # If the target is equal or larger, keep the group unchanged.
                merged_batches.append(group)
                merged_labels.append(torch.full((num_samples,), lab, 
                                                dtype=labels.dtype, device=labels.device))
            else:
                # Partition the group into approximately equal chunks.
                # Each chunk will be averaged to represent that segment.
                chunks = torch.chunk(group, target_samples, dim=0)
                avg_chunks = torch.stack([chunk.mean(dim=0) for chunk in chunks])
                merged_batches.append(avg_chunks)
                merged_labels.append(torch.full((target_samples,), lab, 
                                                dtype=labels.dtype, device=labels.device))
        
        # Concatenate the merged groups from all labels.
        merged_batch = torch.cat(merged_batches, dim=0)
        merged_labels = torch.cat(merged_labels, dim=0)
        
        return merged_batch, merged_labels

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)

    # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

     # Get split index 
    split_index = cfg.hyperparameters.split_index
    
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
    
     # Get channel 
    
    # Register the hooks (we are assuning we have a ViT with blocks modules ) 
    model.blocks[split_index - 1].register_full_backward_pre_hook(backward_channel)
    model.blocks[split_index - 1].register_forward_hook(select_tokens)
    model.blocks[split_index - 1].register_forward_hook(forward_channel)
    

    # Add the channel as an attribute
    model.channel = channel
    model.labels = 0

    model.tokens_percentage  = cfg.method.parameters.tokens_percentage
    model.batch_compression_rate = cfg.method.parameters.batch_compression_rate
    model.method = cfg.method.name 

    return model

# Dynamic Efficient Layer and Tokens Adaptation 
def delta(cfg):

    # Apply channel to the gradient 
    def backward_channel(module, grad_output):

        # Apply the channel 
        new_output = channel(grad_output[0])
        
        # Must return as a Tuple so the ","
        return (new_output, )
    
    # Apply channel to the activations 
    def forward_channel(module, args, output):

        # Apply the channel
        new_output = channel(output, model.training)

        return new_output

    # DELTA forward hook that select tokens 
    def select_tokens(module, input, output): 

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
        blocks_budgets = [(i, block.budget) for i, block in enumerate(model.blocks)]

        # Get the number of training blocks K 
        K = model.K

        # Sort blocks by budget in descending order
        blocks_budgets.sort(key=lambda x: x[1], reverse=True)

        # Get indices of top K blocks
        top_k_indices = {idx for idx, _ in blocks_budgets[:K]}

        # Freeze all blocks except the top K
        for i, block in enumerate(model.blocks):
            print(block.budget)
            for param in block.parameters():
                param.requires_grad = i in top_k_indices  # Only top K blocks remain trainable
        
        return 

    # Initialize  model  
    model = hydra.utils.instantiate(cfg.model)

    # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

     # Get split index 
    split_index = cfg.hyperparameters.split_index
    
    # For each block
    for block in model.blocks:  
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
        block.budget = 1
        block.delta = 0

        # Register the DELTA mechanism
        block.register_forward_hook(update_budget)
        block.register_forward_hook(select_tokens)
        
    
     # Get channel 
    
    # Apply the channel to both forward and backward 
    model.blocks[split_index - 1].register_full_backward_pre_hook(backward_channel)
    model.blocks[split_index - 1].register_forward_hook(forward_channel)

    # Freeze blocks 
    model.register_forward_pre_hook(freeze_blocks)

    # Add the channel as an attribute of the model 
    model.channel = channel

    # Store the name of the method in the model
    model.method = cfg.method.name 
    
    # Initialize the alfa as the training leerning rate 
    model.alfa = cfg.optimizer.lr

    # Get the number of training blocks
    model.K = cfg.method.parameters.K


    return model