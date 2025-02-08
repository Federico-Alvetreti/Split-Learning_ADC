import torch 
import hydra
import copy 

# This baseline consists in sending trough the channel the raw data (images) adding noise 
def send_raw_data_baseline(model, cfg):

    # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

    # Add the channel before the model 
    model = torch.nn.Sequential(channel, model)

    # Add the channel as an attribute
    model.channel = channel

    return model 


# This baseline consists in splitting the network in a certain split_index and just add noise to gradients and activations 
def simple_split_learning_baseline(model, cfg):
     
     # Get channel 
    channel =  hydra.utils.instantiate(cfg.comm.channel)

     # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Apply channel to the gradient 
    def apply_gradient_pipeline(module, grad_output):

        # Apply the channel 
        new_output = channel(grad_output[0])
        
        # Must return as a Tuple so the ","
        return (new_output, )
    
    # Apply channel to the activations 
    def apply_forward_pipeline(module, args, output):

        # Apply the channel
        new_output = channel(output)

        return new_output
    

    # Register the hooks (we are assuning we have a ViT with blocks modules )
    model.blocks[split_index - 1].register_forward_hook(apply_forward_pipeline)
    model.blocks[split_index - 1].register_full_backward_pre_hook(apply_gradient_pipeline)

    # Add the channel as an attribute
    model.channel = channel


    return model 


def split_learning_with_denoising_baseline(model, cfg):

    # Get model name 
    model_name = cfg.model.model_name

    # Get dataset 
    dataset= cfg.dataset.name
    
    # Save model weights
    torch.save(comm_model.state_dict(), "pretrained_models/" +  "flowers-102/" + model_name)

    # Model used to train the denoising autoencoders
    copy_model = copy.deepcopy(model)
    
    
    torch.nn.MSELoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    train_comm_losses = []
    val_comm_losses = []

    # Functions to store activations and gradients
    def activation_hook(module, input, output):
        activations.append(output.detach())
    
    def gradient_hook(module, grad_output):
        gradients.append(grad_output[0].detach())
    
    # Register hooks on model layers
    model.blocks[split_index -1].register_forward_hook(activation_hook)
    model.blocks[split_index -1].register_full_backward_pre_hook(gradient_hook)


    for _ in range(n_epochs):

      print("\n\nEPOCH " + str(_ + 1))
       
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
        activations = []
        gradients = []

        # Get input and labels from batch
        batch_input = batch[0].to(device)
        batch_labels = batch[1].to(device)

        # Compute last layers, get batch predictions
        batch_predictions = model(batch_input)

        # Get batch loss and accuracy
        batch_loss = loss(batch_predictions, batch_labels)
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
        forward_communication_pipeline.train()
        backward_communication_pipeline.train()

        # Initialize train loss 
        forward_comm_train_loss = 0.0
        backward_comm_train_loss = 0.0
        
        # For every gradient / activation in the list 
        for i in range(k):

          for x in activations:
              
              mean = x.mean()
              std = x.std()
              x = (x - mean) / (std + 1e-8)

              # Reconstructed signal 
              reconstructed = forward_communication_pipeline(x)

              # MSE loss 
              forward_comm_batch_loss = comm_loss(reconstructed, x)

              # Backpropagation 
              forward_comm_batch_loss.backward()

              # Update and zero out previous gradients
              forward_communication_optimizer.step()
              forward_communication_optimizer.zero_grad()

              # Store loss 
              forward_comm_train_loss += forward_comm_batch_loss.item()

          for x in gradients:
              
              mean = x.mean()
              std = x.std()
              x = (x - mean) / (std + 1e-8)

              # Reconstructed signal 
              reconstructed = backward_communication_pipeline(x)

              # MSE loss 
              backward_comm_batch_loss = comm_loss(reconstructed, x)

              # Backpropagation 
              backward_comm_batch_loss.backward()

              # Update and zero out previous gradients
              backward_communication_optimizer.step()
              backward_communication_optimizer.zero_grad()

              # Store loss 
              backward_comm_train_loss += backward_comm_batch_loss.item()

      # Compute average loss and accuracy
      average_train_forward_comm_loss = forward_comm_train_loss / (len(train_data_loader) * len(activations) * k)
      average_train_backward_comm_loss = backward_comm_train_loss / (len(train_data_loader) * len(gradients) * k)
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
        activations = []
        gradients = []


        # Get input and labels from batch
        batch_input = batch[0].to(device)
        batch_labels = batch[1].to(device)

        # Get predictions
        batch_predictions = model(batch_input)

        # Get batch loss and accuracy
        batch_loss = loss(batch_predictions, batch_labels) 
        batch_accuracy = torch.sum(batch_labels == torch.argmax(batch_predictions, dim=1)).item() / batch_size

        # Backpropagation
        batch_loss.backward()

        # Free memory used to store gradients (not doing the optimizer.step() in validation phase)
        model.zero_grad(set_to_none=True)

        # Update val_loss and val_accuracy
        val_loss += batch_loss.cpu().item()
        val_accuracy += batch_accuracy


        # Set the communication pipeline in training mode 
        forward_communication_pipeline.eval()
        backward_communication_pipeline.eval()

        # Initialize train loss 
        forward_comm_val_loss = 0.0
        backward_comm_val_loss = 0.0
        
        # For every gradient / activation in the list 
        for i in range(k):

          for x in activations:
            
              mean = x.mean()
              std = x.std()
              x = (x - mean) / (std + 1e-8)

              # Reconstructed signal 
              reconstructed = forward_communication_pipeline(x)

              # MSE loss 
              forward_comm_batch_loss = comm_loss(reconstructed, x)

              # Store loss 
              forward_comm_val_loss += forward_comm_batch_loss.item()

          for x in gradients:
              
              mean = x.mean()
              std = x.std()
              x = (x - mean) / (std + 1e-8)

              # Reconstructed signal 
              reconstructed = backward_communication_pipeline(x)

              # MSE loss 
              backward_comm_batch_loss = comm_loss(reconstructed, x)

              # Store loss 
              backward_comm_val_loss += backward_comm_batch_loss.item()

      # Compute average loss and accuracy
      average_val_forward_comm_loss = forward_comm_val_loss / (len(val_data_loader) * len(activations) * k)
      average_val_backward_comm_loss = backward_comm_val_loss / (len(val_data_loader) * len(gradients) * k)
      average_val_loss = train_loss / len(train_data_loader)
      average_val_accuracy = train_accuracy / len(train_data_loader)


      # Print losses
      print("\nTrain loss: " + str(average_train_loss) + "; Val loss: " + str(average_val_loss))

      # Print accuracies
      print("Train accuracy: " + str(average_train_accuracy) + "; Val accuracy: " + str(average_val_accuracy))

      # Print communication loss 
      print("Train communication forward loss: " + str(average_train_forward_comm_loss) + "; Val communication forward loss: " + str(average_val_forward_comm_loss))
      print("Train communication backward loss: " + str(average_train_backward_comm_loss) + "; Val communication backward loss: " + str(average_val_backward_comm_loss))

    results = {"Train losses" : train_losses,
             "Train accuracies" : train_accuracies,
             "Val losses" : val_losses,
             "Val accuracies" : val_accuracies,
             "Train comm losses" : train_comm_losses,
             "Val comm accuracies" : val_comm_losses,
             }

    return results