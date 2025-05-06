import torch 
import torch.nn as nn
from kmeans_pytorch import kmeans
import torch.nn.functional as F

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
        self.class_token_attention = attn[:, :, 0, :].mean(dim=1)  # Save attention over class token
        attn = self.attn.attn_drop(attn)
        attn_output = attn @ v
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x
  
# Delta block that does selection and update budget 
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

        # Select tokens 
        x = self.select_tokens(x)

        return x
    

class Compressor_Block(nn.Module):

    def __init__(self, block, batch_compression_rate, tokens_compression_rate, kmeans_iterations):
        super().__init__()

        self.block = block
        self.cluster_ids = None
        self.K = None 

        self.batch_compression_rate = batch_compression_rate  
        self.tokens_compression_rate = tokens_compression_rate   
        self.kmeans_iterations = kmeans_iterations         

    def merge_batches_and_select_tokens(self, x: torch.Tensor) -> torch.Tensor:

        # Get shape and set device 
        B, N, _ = x.shape
        device = x.device
        

        K = max(1, int(self.batch_compression_rate * B))
        self.K = K
        num_tokens_to_keep = max(1, int(self.tokens_compression_rate * (N - 1)))

        # Use class token attention to cluster activations 
        class_token_attention = self.block.attn.class_token_attention  
        cluster_ids, centroids = kmeans(X=class_token_attention, num_clusters=K,  distance='euclidean', device=device)

        # Store ids and set device 
        self.cluster_ids = cluster_ids.to(device)
        centroids = centroids.to(device)

        merged_inputs = []

        for k in range(K):

            mask = cluster_ids == k
            grp_x = x[mask]         # (n_members, N, D)

            token_avg = grp_x.mean(dim=0)  # (N, D)

            # Get the attention centroid that represents the cluster 
            attn_cent = centroids[k, 1:]

            # Select the top k tokens and keep them 
            topk = torch.topk(attn_cent, k=num_tokens_to_keep, largest=True, sorted=False).indices
            keep_idxs = torch.cat([torch.zeros(1, dtype=torch.long, device=device), topk + 1])
            merged_inputs.append(token_avg[keep_idxs, :])


        merged_inputs = torch.stack(merged_inputs, dim=0)  # (K, num_keep+1, D)
        return merged_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))
        x = x + self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))
        if self.training:
            x = self.merge_batches_and_select_tokens(x)
        return x

    def compress_labels(self, labels: torch.Tensor) -> torch.Tensor:
        compressed = []
    
        for k in range(self.K):
            one_hot_labels = F.one_hot(labels, num_classes=102).float()
            mask = self.cluster_ids == k  # Reuse this from forward logic
            grp_lbl = one_hot_labels[mask].mean(dim=0)  # (N, L)
            compressed.append(grp_lbl)      
        return torch.stack(compressed, dim=0)