# Libraries
import os
import hydra 
import torch 
import json 
from tqdm import tqdm

# Custom functions 
from omegaconf import OmegaConf


def flatten_params(params):
    if isinstance(params, dict):
        return "_".join(f"{k}={v}" for k, v in params.items())
    return str(params)
_safe_globals = {
    "__builtins__": None,   # disable all other builtins
    "round": round,
}

# Now eval("…") will have access to round()
OmegaConf.register_new_resolver(
    "eval",
    lambda expr: eval(expr, _safe_globals, {})
)
OmegaConf.register_new_resolver("flatten_params", flatten_params)


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

        # Check communication
        if model.communication > max_communication: 
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
def training_schedule(model, train_data_loader, val_data_loader, optimizer, max_communication, device, hydra_output_dir, loss=torch.nn.CrossEntropyLoss(),  plot=True, save_model=True):

    # Lists to store results 
    train_losses, train_accuracies, val_losses, val_accuracies, communication_cost = [], [], [], [], []

    best_val_accuracy = 0
    results_file = os.path.join(hydra_output_dir, "training_results.json")
    final_results_file = os.path.join(hydra_output_dir, "final_training_results.json")
    results = {}

    for epoch in range(1, 1000):
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
        communication_cost.append(model.communication)

        # Plot main results 
        if plot:
            print(f"\nTrain loss: {avg_train_loss:.4f}; Val loss: {avg_val_loss:.4f}")
            print(f"Train accuracy: {avg_train_accuracy:.2f}; Val accuracy: {avg_val_accuracy:.2f}")

        # Check communication 
        if model.communication > max_communication:
            break 

        # Save the best model 
        # if avg_val_accuracy > best_val_accuracy:
        #
        #     best_val_accuracy = avg_val_accuracy
        #     # Save the model checkpoint
        #     model_file = os.path.join(hydra_output_dir, "best_model.pt")
        #     torch.save(model.state_dict(), model_file)

        # Collect results
        results = {
            "Train losses": train_losses,
            "Train accuracies": train_accuracies,
            "Val losses": val_losses,
            "Val accuracies": val_accuracies,
            "Communication cost": communication_cost,
            "Compression" : model.compression}

        # Store results

        # Save the results dictionary as a JSON file
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

    if save_model:
        model_file = os.path.join(hydra_output_dir, "final_model.pt")
        torch.save(model.state_dict(), model_file)

    with open(final_results_file, "w") as f:
        json.dump(results, f, indent=4)

    return 


# Hydra configuration 
@hydra.main(config_path="configs",
            version_base='1.2',
            config_name="default")
def main(cfg):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get dataset parameters
    batch_size = cfg.dataset.batch_size
    max_communication = cfg.dataset.max_communication

    # Get datasets
    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    val_dataset = hydra.utils.instantiate(cfg.dataset.test)

    # Set seed for reproducibility
    for seed in [42, 51, 114]:

        torch.manual_seed(seed)

        # Get dataloaders
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, drop_last=True, batch_size=batch_size, num_workers = 16)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size, num_workers = 16)

        # Get model
        model = hydra.utils.instantiate(cfg.model)

        # Get channel
        channel = hydra.utils.instantiate(cfg.communication.channel)

        # Apply method to the model
        model = hydra.utils.instantiate(cfg.method.model,
                                        channel = channel,
                                        split_index = cfg.hyperparameters.split_index,
                                        model=model).to(device)

        # Get optimizer
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

        # Print model, dataset and method
        print(f"\n\nTraining seed {seed}: \n\n  --model: {cfg.model.model_name} \n  --dataset: {cfg.dataset.name} \n  --communication: {cfg.communication.name} \n  --method: {cfg.method.name} \n  --compression: {model.compression} \n")
        print(f"\n\nParameters:  {cfg.method.parameters}\n\n  ")

        # Get the current Hydra output directory
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        print(hydra_output_dir)

        if seed != 42:
            hydra_output_dir = hydra_output_dir.replace('prova', f'prova_{seed}')
            print(hydra_output_dir)
        else:
            continue

        # Train
        training_schedule(model, train_dataloader, val_dataloader, optimizer, max_communication, device, hydra_output_dir,
                          save_model=seed == 42)

    return


# At the very bottom
if __name__ == "__main__":
    main()
