import torch 
from tqdm import tqdm

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
             "Val accuracies" : val_accuracies,
             "Communication cost" : model.channel.total_communication}

  return results
