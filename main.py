# Libraries 
import hydra 
import torch 
import os 
import json 

# Custom functions 
from utils import training_schedule

@hydra.main(config_path="configs",
            version_base='1.2',
            config_name="default")


def main(cfg):

    # Set device  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get hyperparameters 
    batch_size = cfg.schema.batch_size
    epochs = cfg.schema.epochs

    # Get datasets 
    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test)

    # Get dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True,batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)

    # Initialize  model  
    comm_model = hydra.utils.instantiate(cfg.model)

    # Apply method 
    comm_model = hydra.utils.call(cfg.method, model=comm_model, cfg = cfg ).to(device)

    # Get optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=comm_model.parameters())

    # Train 
    results = training_schedule(comm_model, train_dataloader, test_dataloader, optimizer, epochs, device)

    # Get the current Hydra output directory
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Define the results file path inside Hydra's directory
    results_file = os.path.join(hydra_output_dir, "training_results.json")

    # Save the results dictionary as a JSON file
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    return

if __name__ == "__main__":
    main()