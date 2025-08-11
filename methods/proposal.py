from timm.models import VisionTransformer
import torch.nn as nn
import torch
from kmeans_pytorch import kmeans
import torch.nn.functional as F


# Block used by our proposal to compress batches and select tokens
class Compress_Batches_and_Select_Tokens_Block_Wrapper(nn.Module):

    def __init__(self, block, batch_compression, token_compression, pooling='attention'):
        super().__init__()

        assert pooling in ['attention', 'average', 'cls'], (f"Pooling must be either in "
                                                            f"{['attention', 'average', 'cls']}"
                                                            f" but {pooling} was given.")

        self.block = block

        # Store the clusters for label reconstruction
        self.cluster_ids = None

        # Store compression rates 
        self.batch_compression = batch_compression
        self.token_compression = token_compression

        self.n_new_batches = 0
        self.n_new_tokens = 0

        self.pooling = pooling

    def merge_batches_and_select_tokens(self, x: torch.Tensor) -> torch.Tensor:

        # Get input dimensions 
        n_batches, n_tokens, _ = x.size()

        # Compute the new number of token and batches 
        self.n_new_tokens = max(1, int(self.token_compression * n_tokens))
        self.n_new_batches = max(2, int(self.batch_compression * n_batches))

        # Store device
        device = x.device

        # Get the class token attention of the batch
        if self.pooling == 'attention':
            class_token_attention = self.block.attn.class_token_attention  # n_batches x hidden_dim
        elif self.pooling == 'average':
            class_token_attention = x.mean(1)
        else:
            class_token_attention = x[:, 0]

        # Do K means clustering 
        with torch.no_grad():
            cluster_ids, centroids = kmeans(X=class_token_attention,
                                            num_clusters=self.n_new_batches,
                                            distance='euclidean',
                                            tol=1e-4,
                                            iter_limit=50,
                                            device=device,
                                            tqdm_flag=False
                                            )

        # Make sure these are plain tensors
        self.cluster_ids = cluster_ids.detach().to(device)
        centroids = centroids.detach().to(device)

        # Create variable to store output
        clustered_activations = []

        # For each cluster 
        for cluster_id in range(self.n_new_batches):
            # Get the activations that belong to that cluster
            mask = cluster_ids == cluster_id
            cluster_activations = x[mask]

            # Get the average activation 
            average_activation = cluster_activations.mean(dim=0)

            # Get the attention centroid that represents the cluster 
            cluster_class_token_attention = centroids[cluster_id, 1:]

            # Select the top k tokens and keep them 
            top_k_tokens = torch.topk(cluster_class_token_attention, k=self.n_new_tokens - 1, largest=True,
                                      sorted=False).indices
            top_k_tokens_indexes = torch.cat([torch.zeros(1, dtype=torch.long, device=device), top_k_tokens + 1])

            # Create the new activation 
            clustered_activations.append(average_activation[top_k_tokens_indexes, :])

        # Stack on the batch dimension to get the output
        output = torch.stack(clustered_activations, dim=0)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Normal block forward 
        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))
        x = x + self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))

        # When training merge batches and select tokens 
        if self.training:
            x = self.merge_batches_and_select_tokens(x)

        return x

    # Function to handle the merging of lables (since we merge batches)
    def compress_labels(self, labels, num_classes) -> torch.Tensor:

        # Get number of clusters
        clusters_ids = self.n_new_batches
        new_labels = []

        # For each cluster 
        for clusters_id in range(clusters_ids):
            # Transform to one hot labels, n_batches x num_classes
            one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()

            # Get the labels that belong to the cluster
            mask = self.cluster_ids == clusters_id
            cluster_labels = one_hot_labels[mask]

            # Average them and append 
            cluster_average_label = cluster_labels.mean(dim=0)
            new_labels.append(cluster_average_label)

            # Stack on the batch dimension
        new_labels = torch.stack(new_labels, dim=0)

        return new_labels


# An attention class that stores class token attention scores
class Store_Class_Token_Attn_Wrapper(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.class_token_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normal attention behaviour
        B, N, C = x.shape
        qkv = self.attn.qkv(x).reshape(B, N, 3, self.attn.num_heads, self.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.attn.q_norm(q), self.attn.k_norm(k)
        q = q * self.attn.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        # Store class_token attention 
        self.class_token_attention = attn[:, :, 0, :].mean(dim=1).detach()

        # Normal attention behaviour 
        attn = self.attn.attn_drop(attn)
        attn_output = attn @ v
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x


class model(nn.Module):

    def __init__(self,
                 model: VisionTransformer,
                 channel,
                 split_index,
                 desired_compression=None,
                 batch_compression=None,
                 token_compressions=None,
                 pooling='attention',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if desired_compression is None:
            assert batch_compression is not None and token_compressions is not None, \
                'Both batch_compression and token_compressions must be not None'
            compression = (batch_compression, token_compressions)
            self.compression_ratio = batch_compression * token_compressions
        else:
            assert batch_compression is None and token_compressions is None, 'desired_compression must be not None'
            compression = desired_compression
            self.compression_ratio = desired_compression

        self.compressor_module = None

        # Store compression


        # Build model 
        self.model = self.build_model(model, channel, split_index, compression, pooling)

        # Store channel 
        self.channel = channel

        # Variable to store communication 
        self.communication = 0

        # Store name 
        self.name = "Proposal"

    # Function to build model 
    def build_model(self,
                    model,
                    channel,
                    split_index,
                    compression,
                    pooling):
        # Resolve compression knowing token_compression = batch_compression ** 4
        if isinstance(compression, tuple):
            batch_compression, token_compression = compression
        else:
            batch_compression = compression ** (1 / 5)
            token_compression = batch_compression ** 4

        # Wrap last block with our compression method
        model.blocks[split_index - 1].attn = Store_Class_Token_Attn_Wrapper(model.blocks[split_index - 1].attn)
        model.blocks[split_index - 1] = Compress_Batches_and_Select_Tokens_Block_Wrapper(model.blocks[split_index - 1],
                                                                                         batch_compression,
                                                                                         token_compression,
                                                                                         pooling=pooling)
        self.compressor_module = model.blocks[split_index - 1]

        # Split the original model 
        blocks_before = model.blocks[:split_index]
        blocks_after = model.blocks[split_index:]

        # Add comm pipeline and compression modules 
        model.blocks = nn.Sequential(*blocks_before, channel, *blocks_after)

        return model

    def forward(self, x):
        batch_size = x.shape[0]
        if self.training:
            self.communication += self.compression_ratio * batch_size
        return self.model.forward(x)
