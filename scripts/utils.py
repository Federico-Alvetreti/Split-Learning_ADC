import torch
import torch.nn as nn
from tqdm import tqdm
import hydra 
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import hydra 
from copy import deepcopy
from torchvision import transforms

# Standard training phase 
def training_phase(model, train_data_loader, loss, optimizer, device, plot, max_communication):

    if plot:
        print("\nTraining phase: ")

    # Initialize train loss and accuracy
    train_loss = 0.0
    train_accuracy = 0.0

    # Set the model into training mode
    model.train()

    # Counter   
    iterations = 0
    # Forward the train set
    for batch in tqdm(train_data_loader, disable=not plot):

        # Get input and labels from batch
        batch_input = batch[0].to(device)
        batch_labels = batch[1].to(device)

        # Get batch predictions
        batch_predictions = model(batch_input)

        

        # Get batch accuracy + handle batch compression for labels  
        if hasattr(model, "compressor_module"):
            batch_labels = model.compressor_module.compress_labels(batch_labels, len(train_data_loader.dataset.classes))
            batch_accuracy = 0
        else:
            batch_accuracy = torch.sum(batch_labels == torch.argmax(batch_predictions, dim=1)).item() / batch_labels.shape[0]

        # Get batch loss
        batch_loss = loss(batch_predictions, batch_labels)

        iterations+=1

        # Store them
        train_loss += batch_loss.detach().cpu().item()
        train_accuracy += batch_accuracy

        # Compute gradients
        batch_loss.backward()  

        # Update and zero out previous gradients
        optimizer.step()
        optimizer.zero_grad()

        # Store communication cost 
        if hasattr(model, "channel"):
            current_communication_cost = model.channel.total_communication
            # If the current communication cost is above the max communication break the training 
            if  current_communication_cost > max_communication:
                break
        elif model.method=="JPEG":
            current_communication_cost = train_data_loader.dataset.total_communication
            
            # If the current communication cost is above the max communication break the training 
            if  current_communication_cost > max_communication:
                break

    # Compute average loss and accuracy
    average_train_loss = train_loss / iterations
    average_train_accuracy = train_accuracy / iterations

    return average_train_loss, average_train_accuracy

# Standard validation phase
def validation_phase(model, val_data_loader, loss, device, plot):
  if plot: 
    print("Validation phase: ")

  # Set the model to evaluation mode
  model.eval()

  # Initialize loss and accuracy
  val_loss = 0.0
  val_accuracy = 0.0

  # Forward val set
  with torch.no_grad():
    for batch in tqdm(val_data_loader, disable=not plot):

      # Get input and labels from batch
      batch_input = batch[0].to(device)
      batch_labels = batch[1].to(device)

      # Get predictions
      batch_predictions = model(batch_input)

      # Get batch loss and accuracy
      batch_loss = loss(batch_predictions, batch_labels)
      batch_accuracy = torch.sum(batch_labels == torch.argmax(batch_predictions, dim=1)).item() / batch_labels.shape[0]

      # Update val_loss and val_accuracy
      val_loss += batch_loss.cpu().item()
      val_accuracy += batch_accuracy

  # Compute average loss and accuracy
  average_val_loss = val_loss / len(val_data_loader)
  average_val_accuracy = val_accuracy / len(val_data_loader)

  return average_val_loss, average_val_accuracy

# Standard training / validation cicle
def training_schedule(model, train_data_loader, val_data_loader, optimizer, max_communication, device, loss=torch.nn.CrossEntropyLoss(),  plot=True, patience=10):


    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    if hasattr(model, "channel") or model.method == "JPEG":
        communication_costs = []

    # Convergence check 
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, 450):
        torch.cuda.empty_cache()
        if plot:
            print(f"\n\nEPOCH {epoch}")

        # Training and validation phases 
        avg_train_loss, avg_train_accuracy = training_phase(model, train_data_loader, loss, optimizer, device, plot, max_communication)
        avg_val_loss, avg_val_accuracy = validation_phase(model, val_data_loader, loss, device, plot)

        # Store results 
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        if plot:
            print(f"\nTrain loss: {avg_train_loss:.4f}; Val loss: {avg_val_loss:.4f}")
            print(f"Train accuracy: {avg_train_accuracy:.2f}; Val accuracy: {avg_val_accuracy:.2f}")

        # Store communication cost 
        if hasattr(model, "channel"):
            communication_costs.append(model.channel.total_communication)
            # If the communication cost exceeds the max communication stop. 
            if communication_costs[-1] > max_communication:
                break
        # Handle the cases of JPEG and Normal training, where they need just an epoch to train 
        elif model.method=="JPEG":
            communication_costs.append(train_data_loader.dataset.total_communication)
            break
        else:
            break


        # Convergence check: early stopping if val loss doesn't improve in tot epochs 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if plot:
                    print(f"⏹️ Early stopping: no improvement in val loss for {patience} epochs.")
                break


    # Collect results
    results = {
        "Train losses": train_losses,
        "Train accuracies": train_accuracies,
        "Val losses": val_losses,
        "Val accuracies": val_accuracies,
    }
    if hasattr(model, "channel") or model.method == "JPEG":
        results["Communication cost"] = communication_costs


    return results

# Filters a dataset object to keep the images that can be compressed using JPEG with the current SNR and k 
def filter_dataset_by_jpeg(dataset, cfg):

    # Create a "raw" dataset that applies just randomcrop transform 
    raw_dataset = hydra.utils.instantiate(
        cfg.dataset.train,
        transform=transforms.Resize([224, 224]))

    # Compute the maximum bytes that can pass through the channel 
    snr = cfg.hyperparameters.snr
    k_over_n = cfg.hyperparameters.k_over_n
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

# Function to train all parameters 
def train_all(model):
    for name, param in model.named_parameters():
        param.requires_grad = True

# Measure the bytes of a tensor 
def tensor_to_bytes(x: torch.Tensor) -> int:
    buf = BytesIO()
    torch.save(x, buf)
    return buf.tell()

# Creates n_copy of a feed forward network  
def get_ffn(input_size, output_size, n_layers, n_copy, drop_last_activation):

    # Compute the hidden dimensions of the network 
    shapes = np.linspace(input_size, output_size, num=n_layers + 1, endpoint=True, dtype=int)

    # Create the model as blocks of linear + ReLU 
    model = []
    for s in range(len(shapes) - 1):
        model.append(nn.Linear(shapes[s], shapes[s + 1]))
        model.append(nn.ReLU())
    if drop_last_activation:
        model = model[:-1]
    model = nn.Sequential(*model)

    # Create the number of copies specifeid in n_copy 
    if n_copy == 1 :
       return model
    else: 
        models = []
        for _ in range(n_copy):
            _model = deepcopy(model)
            for m in _model.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            models.append(_model)

        return models


def resolve_compression_rate(cfg):

    # Compute baseline compression without any techniques
    image_size = 224 * 224 * 3
    activation_size = 192 * 197
    encoder_compression = cfg.communication.encoder.output_size
    standard_compression = encoder_compression * activation_size / image_size

    # Desired overall compression ratio
    k_over_n = cfg.hyperparameters.k_over_n
    target_rate = k_over_n / standard_compression

    
    if cfg.method.name == "ours":
        if target_rate >= 0.2:
            batch_compression, token_compression = (1, max(target_rate,1))
        else: 
            batch_compression, token_compression = ( 5 * (target_rate / 5 )**0.5, (target_rate / 5 )**0.5)
    
        return (batch_compression, token_compression)
    else:
        if target_rate < 1: 
            raise(ValueError("Can't train at this compression.", UserWarning))
    

