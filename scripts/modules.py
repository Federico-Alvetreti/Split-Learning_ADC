import torch 
import torch.nn as nn
from kmeans_pytorch import kmeans
import torch.nn.functional as F
import numpy as np 

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

        # Select token
        if self.training:
            x = self.select_tokens(x)

        return x

# Block used by ours to compress batches and select tokens 
class Compressor_Block(nn.Module):

    def __init__(self, block, batch_compression_rate, tokens_compression_rate):
        super().__init__()

        self.block = block
        self.cluster_ids = None
        self.K = None 

        self.batch_compression_rate = batch_compression_rate  
        self.tokens_compression_rate = tokens_compression_rate
        self.forward_happened = False  

    def merge_batches_and_select_tokens(self, x: torch.Tensor) -> torch.Tensor:

        # Get shape and set device 
        B, N, _ = x.shape
        device = x.device
        
        K = max(2, int(self.batch_compression_rate * B))
        self.K = K
        num_tokens_to_keep = max(1, int(self.tokens_compression_rate * (N-1)))

        # Use class token attention to cluster activations 
        class_token_attention = self.block.attn.class_token_attention

        with torch.no_grad():
            cluster_ids, centroids =  kmeans(X=class_token_attention,       # (B, D)
                                            num_clusters=K,                # as before
                                            distance='euclidean',          # as before
                                            tol=1e-4,                      # convergence threshold
                                            iter_limit=10,                 # ← stop after at most 10 iters
                                            device=device,
                                            tqdm_flag=False                # turn off the printouts
                                            )

        # Make sure these are plain tensors (no grad)
        self.cluster_ids = cluster_ids.detach().to(device)
        centroids = centroids.detach().to(device)

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

    def compress_labels(self, labels, num_classes) -> torch.Tensor:
        compressed = []
        for k in range(self.K):
            one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
            mask = self.cluster_ids == k  # Reuse this from forward logic
            grp_lbl = one_hot_labels[mask].mean(dim=0)  # (N, L)
            compressed.append(grp_lbl) 
        final_labels = torch.stack(compressed, dim=0)

        return final_labels

# Quantization 
class _QuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, n_bits: int):
        if torch.is_complex(x):
            real_q = _QuantizeFunction.forward(ctx, x.real, n_bits)
            imag_q = _QuantizeFunction.forward(ctx, x.imag, n_bits)
            return torch.complex(real_q, imag_q)
        
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        x_min = x.min()
        x_max = x.max()
        levels = 2 ** n_bits

        thresholds = torch.linspace(x_min, x_max, steps=levels + 1, device=x.device)
        centers = (thresholds[:-1] + thresholds[1:]) / 2  # Midpoints

        diffs = torch.abs(x_flat.unsqueeze(-1) - centers.unsqueeze(0))  # [batch, dim, levels]
        closest_indices = torch.argmin(diffs, dim=-1)  # [batch, dim]

        quantized_flat = centers[closest_indices]
        quantized_x = quantized_flat.view_as(x)

        ctx.save_for_backward(x)  # optional, for custom gradient

        

        return quantized_x

    @staticmethod
    def backward(ctx, grad_output):
        # By default: pass-through gradient (STE)
        # You can customize this logic as needed
        grad_input = grad_output.clone()
        return grad_input, None  # None for n_bits

class QuantizationLayer(nn.Module):

    def __init__(self, n_bits: int):
        super().__init__()
        self.n_bits = n_bits

    def forward(self, x: torch.Tensor):
        if self.training:
            return _QuantizeFunction.apply(x, self.n_bits)
        else:
            return x 
        

# Random top K 
class RandomTopKModifier(nn.Module):
    def __init__(self, rate: float, random_portion: float = 0.1):
        super(RandomTopKModifier, self).__init__()
        self.rate = rate
        self.random_portion = random_portion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        k = max(1, round(self.rate * x_flat.shape[1]))
        sample_dim = x_flat.shape[1]

        if self.training:
            _, top_k_indices = torch.topk(torch.abs(x_flat), k, sorted=False)

            probs = torch.full_like(x_flat.real, self.random_portion / (sample_dim - k))
            probs = torch.scatter(probs, 1, top_k_indices,
                                  (1 - self.random_portion) / k)
            selected_indices = torch.multinomial(probs, k)
            mask = torch.scatter(torch.zeros_like(x_flat), 1, selected_indices, 1)

            return mask.view(*x.shape) * x

        else:
            return x