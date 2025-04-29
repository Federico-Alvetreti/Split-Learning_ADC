# Libraries 
import hydra 
import torch 
import os 
import json 

# Custom functions 
from scripts.utils import training_schedule, filter_dataset_by_jpeg
from omegaconf import OmegaConf


def flatten_params(params):
    if isinstance(params, dict):
        return "_".join(f"{k}={v}" for k, v in params.items())
    return str(params)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("flatten_params", flatten_params)


# Hydra configuration 
@hydra.main(config_path="configs",
            version_base='1.2',
            config_name="default")

def main(cfg):

    # Print model, dataset and method
    print(f"\n\nTraining {cfg.model.model_name} on {cfg.dataset.name} with an SNR of {cfg.hyperparameters.snr} using {cfg.method.name}. \n")
    
    # Set seed for reproducibility 
    torch.manual_seed(42) 

    # Set device  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get hyperparameters 
    batch_size = cfg.dataset.batch_size
    epochs = cfg.dataset.epochs 

    # Get datasets 
    to_download = not os.path.exists(cfg.dataset.train.root)

    train_dataset = hydra.utils.instantiate(cfg.dataset.train, download=to_download, _convert_="partial")
    val_dataset = hydra.utils.instantiate(cfg.dataset.test, _convert_="partial")

    # Compress the train dataset when using JPEG method 
    if cfg.method.name == "JPEG":
        train_dataset = filter_dataset_by_jpeg(train_dataset, cfg)
        if train_dataset == 0:
            return


    # Get dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True,batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size)

    # Get model 
    comm_model = hydra.utils.call(cfg.method.function, cfg = cfg).to(device)

    # Get optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=comm_model.parameters())

    # Train 
    results = training_schedule(comm_model, train_dataloader, val_dataloader, optimizer, epochs, device)

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