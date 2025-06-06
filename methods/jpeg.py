
from timm.models import VisionTransformer
import torch.nn as nn 

from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import hydra 
from torchvision import transforms


# Filters a dataset object to keep the images that can be compressed using JPEG with the current SNR and k 
def filter_dataset_by_jpeg(dataset, cfg):

    # Create a "raw" dataset that applies just randomcrop transform 
    raw_dataset = hydra.utils.instantiate(
        cfg.dataset.train,
        transform=transforms.Resize([224, 224]))

    # Compute the maximum bytes that can pass through the channel 
    snr = cfg.hyperparameters.snr
    k_over_n = cfg.method.parameters.k_over_n
    number_of_bits_of_full_image =  224*224*3*8 # image_size(224*224*3) * bits_per_number(8)
    max_symbols=  round(k_over_n * number_of_bits_of_full_image)   
    linear_snr =  10**(snr / 10)
    bits_per_symbol = np.log2(1 + linear_snr)
    max_bytes = max_symbols * bits_per_symbol / 8
    
    # Set max and min quality of images 
    max_quality, min_quality = 95, 1
    
    # List and dict to store the index of valid images and its resepctive max quality 
    valid_idxs   = []
    quality_map  = {}
    avg_quality = 0
    communication = 0
    print("Compressing the dataset using JPEG...")

    for idx in range(len(raw_dataset)):
        # Get image and its shape 
        sample = raw_dataset[idx]
        img = sample[0] if isinstance(sample, (tuple, list)) else sample
        
        # Find the highest quality that fits the budget
        for q in range(max_quality, min_quality - 1, -1):

            # Get the bytes of the compressed image 
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=q)
            compressed_img_bytes = buf.tell()


            # If they are lower than max bytes store the index and its best quality 
            if compressed_img_bytes <= max_bytes:
                communication += buf.tell()
                valid_idxs.append(idx)
                quality_map[idx] = q
                avg_quality+=q
                break
    
    # Get percentage of retained images and print it 
    pct = len(valid_idxs) * 100.0 / len(raw_dataset)

    if len(valid_idxs) == 0:
        raise(ValueError("Can't train at this compression.", UserWarning))
    
    avg_quality /= len(valid_idxs)
    print(f"Kept {len(valid_idxs)}/{len(raw_dataset)} images ({pct:.1f}%) with an average quality of {avg_quality}.")
    
    # Build a Dataset wrapper that returns the compressed + transformed images
    class JPEGCompressedDataset(Dataset):
        def __init__(self, raw_ds, full_ds, indices, quality_map, communication):
            self.raw_ds      = raw_ds
            self.full_ds     = full_ds 
            self.indices     = indices
            self.quality_map = quality_map
            self.total_communication = communication
            # Modify the original transform to remove any RandomCrop
            self.filtered_transform = self._remove_random_crop(self.full_ds.transform)

        def _remove_random_crop(self, transform):
            """Remove RandomCrop from a transform composition."""
            if isinstance(transform, transforms.Compose):
                new_transforms = []
                for t in transform.transforms:
                    if not isinstance(t, transforms.RandomCrop):
                        new_transforms.append(t)
                return transforms.Compose(new_transforms)
            else:
                # If transform is not a Compose, return it as-is
                return transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            orig_idx = self.indices[i]
            sample = self.raw_ds[orig_idx]
            if isinstance(sample, (tuple, list)):
                img, label = sample
            else:
                img, label = sample, None

            q = self.quality_map[orig_idx]
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            compressed = Image.open(buf).convert(img.mode)

            # Apply the original transform without RandomCrop
            out = self.filtered_transform(compressed)

            return (out, label) if label is not None else out

    return JPEGCompressedDataset(raw_dataset, dataset, valid_idxs, quality_map, communication)


class model(nn.Module):

    def __init__(self, 
                 model: VisionTransformer,
                 encoder,
                 channel,
                 decoder,
                 split_index,
                 *args, **kwargs): 
        
        super().__init__(*args, **kwargs)
        
        # Build model 
        self.model = self.build_model(model, encoder, channel, decoder, split_index)

        # Store compression 
        self.compression = 1

        self.name = "JPEG"
        self.channel = channel
    
    # Function to build model 
    def build_model(self, model, encoder, channel, decoder, split_index):

        # Split the original model 
        blocks_before = model.blocks[:split_index]
        blocks_after = model.blocks[split_index:]

        # Add comm pipeline and compression modules 
        model.blocks = nn.Sequential(*blocks_before, encoder, channel, decoder, *blocks_after)

        return model 

    # Forward 
    def forward(self, x):
        return self.model.forward(x)