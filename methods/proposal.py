
from timm.models import VisionTransformer
import torch.nn as nn
import torch 

# Block used by ours to compress batches and select tokens 
class Compress_Batches_and_Select_Tokens_Block_Wrapper(nn.Module):

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

# An attention class that stores class token attention scores
class Store_Class_Token_Attn_Wrapper(nn.Module):
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

class model(nn.Module):

    def __init__(self, 
                 model: VisionTransformer,
                 encoder,
                 channel,
                 decoder,
                 split_index,
                 batch_compression, 
                 token_compression,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        # Build model 
        self.model = self.build_model(model, encoder, channel, decoder, split_index, batch_compression, token_compression)

        # Store compression 
        self.compression = batch_compression * token_compression

        self.name = "Proposal"
        self.channel = channel
    
    # Function to build model 
    def build_model(self, 
                    model, 
                    encoder,
                    channel,
                    decoder,
                    split_index,
                    batch_compression, 
                    token_compression):

        # Split the original model 
        blocks_before = model.blocks[:split_index]
        blocks_after = model.blocks[split_index:]

        # Wrap last block with our compression method 
        model.blocks[split_index -1].attn = Store_Class_Token_Attn_Wrapper(model.blocks[split_index -1].attn)
        model.blocks[split_index -1] = Compress_Batches_and_Select_Tokens_Block_Wrapper(model.blocks[split_index -1], batch_compression, token_compression)

        # Add comm pipeline and compression modules 
        model.blocks = nn.Sequential(*blocks_before, encoder, channel, decoder, *blocks_after)

        return model 

    # Forward 
    def forward(self, x):
        return self.model.forward(x)