# Libraries 
import hydra 
from copy import deepcopy
import torch 
from torch import nn
import os 

# Custom functions 
from comm_functions import CommunicationPipeline
from utils import training_schedule

@hydra.main(config_path="configs",
            version_base='1.2',
            config_name="default")

def main(cfg):

    # Set device  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    
    # Get hyperparameters 
    batch_size = cfg.schema.batch_size
    epochs = cfg.schema.epochs

    # Get datasets 
    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test)

    # Get dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True,batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)

    # Get pretrained model 
    pretrained_model =  hydra.utils.instantiate(cfg.model).to(device)

    # Build the communication pipeline 

    # Get encoder 
    encoder = hydra.utils.instantiate(cfg.comm.encoder, input_size=pretrained_model.num_features)

    # Get channel 
    channel = hydra.utils.instantiate(cfg.comm.channel)

    # Get decoder 
    decoder = hydra.utils.instantiate(cfg.comm.decoder, input_size=encoder.output_size, output_size=pretrained_model.num_features)

    # Get pipeline 
    communication_pipeline = CommunicationPipeline(channel=channel, encoder=encoder, decoder=decoder).to(device)


    # Build the communication model 

    # Get the split index 
    split_index = cfg.hyperparameters.split_index

    # Get the blocks before and after the communication pipeline 
    blocks_before = pretrained_model.blocks[:split_index]
    blocks_after = pretrained_model.blocks[split_index:]
    
    # Build the model 
    comm_model = deepcopy(pretrained_model)
    comm_model.blocks = nn.Sequential(*blocks_before, communication_pipeline, *blocks_after)

    # Prepare for training 
    optimizer = hydra.utils.instantiate(cfg.optimizer,params=comm_model.parameters())

    # Train 
    results = training_schedule(comm_model, train_dataloader, test_dataloader, optimizer, epochs, device)

    # Get the current Hydra output directory
    hydra_output_dir = os.getcwd()

    # Define the results file path inside Hydra's directory
    results_file = os.path.join(hydra_output_dir, "training_results.json")

    # Save the results dictionary as a JSON file
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    return

if __name__ == "__main__":
    main()