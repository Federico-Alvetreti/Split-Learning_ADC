import torch 
from tqdm import tqdm
import timm 

# Standard training phase 
def training_phase(model, train_data_loader, loss, optimizer, device):

  print("\nTraining phase: ")

  # Initialize train loss and accuracy
  train_loss = 0.0
  train_accuracy = 0.0

  # Set the model into training mode
  model.train()

  # Forward the train set
  for batch in tqdm(train_data_loader):

    # Get input and labels from batch
    batch_input = batch[0].to(device)
    batch_labels = batch[1].to(device)

    # Compute last layers, get batch predictions
    batch_predictions = model(batch_input)

    # Get batch loss and accuracy
    batch_loss = loss(batch_predictions, batch_labels)
    batch_accuracy = torch.sum(batch_labels == torch.argmax(batch_predictions, dim=1)).item() / batch_labels.shape[0]

    # Store them
    train_loss += batch_loss.item()
    train_accuracy += batch_accuracy

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
def validation_phase(model, val_data_loader, loss, device):

  print("Validation phase: ")

  # Set the model to evaluation mode
  model.eval()

  # Initialize loss and accuracy
  val_loss = 0.0
  val_accuracy = 0.0

  # Forward val set
  with torch.no_grad():
    for batch in tqdm(val_data_loader):

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
def training_schedule(model, train_data_loader, val_data_loader, optimizer, n_epochs, device,  loss = torch.nn.CrossEntropyLoss()):

  # Initialize train losses and accuracies
  train_losses = []
  train_accuracies = []

  # Initialize val losses and accuracies
  val_losses = []
  val_accuracies = []

  # For each epoch
  for epoch in range(1, n_epochs + 1, 1):

    print("\n\nEPOCH " + str(epoch))

    # Training phase
    avg_train_loss, avg_train_accuracy = training_phase(model, train_data_loader, loss, optimizer, device)

    # Validation phase
    avg_val_loss, avg_val_accuracy = validation_phase(model, val_data_loader, loss, device)

    # Store train loss and accuracy
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)

    # Store val loss and accuracy
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)

    # Print losses
    print("\nTrain loss: " + str(avg_train_loss) + "; Val loss: " + str(avg_val_loss))

    # Print accuracies
    print("Train accuracy: " + str(avg_train_accuracy) + "; Val accuracy: " + str(avg_val_accuracy))



  results = {"Train losses" : train_losses,
             "Train accuracies" : train_accuracies,
             "Val losses" : val_losses,
             "Val accuracies" : val_accuracies}

  return results



def train_communication_pipeline(model, communication_pipeline,
                                 train_data_loader, val_data_loader, 
                                 model_optimizer, comm_optimizer,
                                 n_epochs, device, loss = torch.nn.CrossEntropyLoss()):
    
    
    comm_loss = torch.nn.MSELoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    train_comm_losses = []
    val_comm_losses = []

    # Functions to store activations and gradients
    def activation_hook(module, input, output):
        communication_tensors.append(output.detach())
    
    def gradient_hook(module, grad_input, grad_output):
        communication_tensors.append(grad_output[0].detach())
    
    # Register hooks on model layers
    for layer in model.modules():
        if isinstance(layer, timm.models.vision_transformer.Block):
            layer.register_forward_hook(activation_hook)
            layer.register_backward_hook(gradient_hook)


    for _ in range(n_epochs):

      print("\n\nEPOCH " + str(_))
       
      # Training original model 
      print("TRAINING PHASE: ")

      # Initialize train loss and accuracy
      train_loss = 0.0
      train_accuracy = 0.0

      # Set the model into training mode
      model.train()

      # Forward the train set
      for batch in tqdm(train_data_loader):
        
        # Get batch size 
        batch_size = batch[0].shape[0]
        
        # List where to store activations and gradients
        communication_tensors = []

        # Get input and labels from batch
        batch_input = batch[0].to(device)
        batch_labels = batch[1].to(device)

        # Compute last layers, get batch predictions
        batch_predictions = model(batch_input)

        # Get batch loss and accuracy
        batch_loss = loss(batch_predictions, batch_labels) / batch_size
        batch_accuracy = torch.sum(batch_labels == torch.argmax(batch_predictions, dim=1)).item() / batch_size

        # Store them
        train_loss += batch_loss.item()
        train_accuracy += batch_accuracy

        # Backpropagation
        batch_loss.backward()

        # Update and zero out previous gradients
        model_optimizer.step()
        model_optimizer.zero_grad()

        # Training commmunication pipeline 

        # Set the communication pipeline in training mode 
        communication_pipeline.train()
        
        # Initialize train loss 
        comm_train_loss = 0
        
        # For every gradient / activation in the list 
        for x in communication_tensors:

            # Reconstructed signal 
            reconstructed = communication_pipeline(x)

            # MSE loss 
            comm_batch_loss = comm_loss(reconstructed, x)

            # Backpropagation 
            comm_batch_loss.backward()

            # Update and zero out previous gradients
            comm_optimizer.step()
            comm_optimizer.zero_grad()

            # Store loss 
            comm_train_loss += comm_batch_loss.item() / batch_size 


      # Compute average loss and accuracy
      average_train_comm_loss = comm_train_loss / (len(train_data_loader) * len(communication_tensors))
      average_train_loss = train_loss / len(train_data_loader)
      average_train_accuracy = train_accuracy / len(train_data_loader)


      print("VALIDATION PHASE: ")

      # Initialize loss and accuracy
      val_loss = 0.0
      val_accuracy = 0.0

      # Forward val set
      for batch in tqdm(val_data_loader):

        # Get batch size 
        batch_size = batch[0].shape[0]
        
        # List where to store activations and gradients
        communication_tensors = []

        # Get input and labels from batch
        batch_input = batch[0].to(device)
        batch_labels = batch[1].to(device)

        # Get predictions
        batch_predictions = model(batch_input)

        # Get batch loss and accuracy
        batch_loss = loss(batch_predictions, batch_labels) / batch_size
        batch_accuracy = torch.sum(batch_labels == torch.argmax(batch_predictions, dim=1)).item() / batch_size

        # Backpropagation
        batch_loss.backward()

        # Free memory used to store gradients (not doing the optimizer.step() in validation phase)
        model.zero_grad(set_to_none=True)

        # Update val_loss and val_accuracy
        val_loss += batch_loss.cpu().item()
        val_accuracy += batch_accuracy

        # Set the communication pipeline in evaluation mode 
        communication_pipeline.eval()
        
        # Initialize train loss 
        comm_val_loss = 0
        
        # For every gradient / activation 
        for x in communication_tensors:

          # Reconstructed signal 
          reconstructed = communication_pipeline(x)

          # MSE loss 
          comm_batch_loss = comm_loss(reconstructed, x)

          # Store loss 
          comm_val_loss += comm_batch_loss.item() / batch_size 


      # Compute average loss and accuracy
      average_val_comm_loss = comm_val_loss / (len(val_data_loader) * len(communication_tensors))
      average_val_loss = val_loss / len(val_data_loader)
      average_val_accuracy = val_accuracy / len(val_data_loader)
      

      # Store results 
      train_losses.append(average_train_loss)
      train_accuracies.append(average_train_accuracy)
      val_losses.append(average_val_loss)
      val_accuracies.append(average_val_accuracy)
      train_comm_losses.append(average_train_comm_loss)
      val_comm_losses.append(average_val_comm_loss)

      # Print losses
      print("\nTrain loss: " + str(average_train_loss) + "; Val loss: " + str(average_val_loss))

      # Print accuracies
      print("Train accuracy: " + str(average_train_accuracy) + "; Val accuracy: " + str(average_val_accuracy))

      # Print communication loss 
      print("Train communication loss: " + str(average_train_comm_loss) + "; Val communication loss: " + str(average_val_comm_loss))



    results = {"Train losses" : train_losses,
             "Train accuracies" : train_accuracies,
             "Val losses" : val_losses,
             "Val accuracies" : val_accuracies,
             "Train comm losses" : train_comm_losses,
             "Val comm accuracies" : val_comm_losses,
             }

    return results
