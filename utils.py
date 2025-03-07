import torch 
from tqdm import tqdm
import hydra 
import os

# Standard training phase 
def training_phase(model, train_data_loader, loss, optimizer, device, plot):
  if plot:
    print("\nTraining phase: ")

  # Initialize train loss and accuracy
  train_loss = 0.0
  train_accuracy = 0.0

  # Set the model into training mode
  model.train()

  # Forward the train set
  for batch in tqdm(train_data_loader, disable=not plot):

    # Get input and labels from batch
    batch_input = batch[0].to(device)
    batch_labels = batch[1].to(device)

    # Handle batch_compression method 
    if hasattr(model, "labels"):
      model.labels = batch_labels 

    # Get batch predictions
    batch_predictions = model(batch_input)

    # Handle batch_compression method 
    if hasattr(model, "labels"):
      batch_labels = model.labels

    # Get batch loss and accuracy
    batch_loss = loss(batch_predictions, batch_labels)
    batch_accuracy = torch.sum(batch_labels == torch.argmax(batch_predictions, dim=1)).item() / batch_labels.shape[0]

    # Store the losses in the model if required  
    if hasattr(model, "last_losses"):
       model.last_losses.append(batch_loss)
       model.last_losses = model.last_losses[-2:]

    # Store them
    train_loss += batch_loss.item()
    train_accuracy += batch_accuracy

    # Backpropagation
    batch_loss.backward()

    # Update and zero out previous gradients
    optimizer.step()
    optimizer.zero_grad()

    # Freeze edge state "C"
    if model.method == "freeze":
      while model.state == 2:

        # Forward last activations 
        batch_predictions = model.stored_activations_forward()
        
        # Batch loss 
        batch_loss = loss(batch_predictions, batch_labels)
        model.last_losses.append(batch_loss)
        model.last_losses = model.last_losses[-2:]

        # Backpropagation
        batch_loss.backward()

        # Update and zero out previous gradients
        optimizer.step()
        optimizer.zero_grad()



  # Compute average loss and accuracy
  average_train_loss = train_loss / len(train_data_loader)
  average_train_accuracy = train_accuracy / len(train_data_loader)

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
def training_schedule(model, train_data_loader, val_data_loader, optimizer, n_epochs, device,  loss = torch.nn.CrossEntropyLoss(), plot = True):

  # Initialize train losses and accuracies
  train_losses = []
  train_accuracies = []

  # Initialize val losses and accuracies
  val_losses = []
  val_accuracies = []
  
  # Inizialize communication cost 
  if hasattr(model, "channel"):
    communication_costs = []
  

  # For each epoch
  for epoch in range(1, n_epochs + 1, 1):
    if plot:
      print("\n\nEPOCH " + str(epoch))

    # Training phase
    avg_train_loss, avg_train_accuracy = training_phase(model, train_data_loader, loss, optimizer, device, plot)

    # Validation phase
    avg_val_loss, avg_val_accuracy = validation_phase(model, val_data_loader, loss, device, plot)

    # Store train loss and accuracy
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)

    # Store val loss and accuracy
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)

    # Store epoch communication cost 
    if hasattr(model, "channel"):
      communication_costs.append(model.channel.total_communication)

    if plot:
      # Print losses
      print("\nTrain loss: " + str(avg_train_loss) + "; Val loss: " + str(avg_val_loss))

      # Print accuracies
      print("Train accuracy: " + str(avg_train_accuracy) + "; Val accuracy: " + str(avg_val_accuracy))

    

  # Store results in a dictionary 
  results = {
    "Train losses": train_losses,
    "Train accuracies": train_accuracies,
    "Val losses": val_losses,
    "Val accuracies": val_accuracies,
}

  if hasattr(model, "channel"):
      results["Communication cost"] = communication_costs

  return results

# Return a pretrained model of the type set in the cfg 
def load_pretrained_model(cfg):
    # Get model name and dataset
    model_name = cfg.model.model_name
    dataset = cfg.dataset.name
    
    # Define model path
    model_path = f"pretrained_models/{dataset}/{model_name}.pth"
    
    # Check if the model already exists
    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = hydra.utils.instantiate(cfg.model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    print(f"\n\nModel file {model_path} not found.")
    print(f"Pretraining {model_name} on {dataset}  \n\n")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Set device  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get hyperparameters 
    batch_size = cfg.schema.batch_size
    epochs = cfg.schema.pre_training_epochs
    
    # Get datasets 
    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test)
    
    # Get dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)
    
    # Initialize model  
    model = hydra.utils.instantiate(cfg.model).to(device)
    
    # Get optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    # Train 
    _ = training_schedule(model, train_dataloader, test_dataloader, optimizer, epochs, device, plot = True)
    
    # Store model weights
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model

# Function used to freeze edge layers 
def freeze_edge(model,split_index):
    # Freeze initial encoding layers 
    for name, param in model.named_parameters():
        if name in ["cls_token", "pos_embed", "patch_embed.proj.weight", "patch_embed.proj.bias"]:
            param.requires_grad = False

    # Freeze every block  before the split_index 
    for block in model.blocks[:split_index]:
        for param in block.parameters():
            param.requires_grad = False

# Function to train all parameters 
def train_all(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
