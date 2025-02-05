# Libraries 
import hydra 
from copy import deepcopy
import torch 
from torch import nn
import os 
import json 

# Custom functions 
from comm_functions import CommunicationPipeline
from utils import training_schedule, train_communication_pipeline

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

    # Train the encoder / decoder 
    _ = train_communication_pipeline(pretrained_model, communication_pipeline, train_dataloader, test_dataloader,
                                 hydra.utils.instantiate(cfg.optimizer,params=pretrained_model.parameters()),
                                 hydra.utils.instantiate(cfg.optimizer,params=communication_pipeline.parameters()),
                                 20, device)
    
    del pretrained_model  # Delete the model
    del _                 # Delete the results 
    torch.cuda.empty_cache()  # Free up GPU memory


    # Freeze communication pipeline parameters by setting requires_grad=False
    for param in communication_pipeline.parameters():
        param.requires_grad = False

    # Get the split index 
    split_index = cfg.hyperparameters.split_index

    # Initialize communication model  
    comm_model = hydra.utils.instantiate(cfg.model).to(device)

    # Get the blocks before and after the communication pipeline 
    blocks_before = comm_model.blocks[:split_index]
    blocks_after = comm_model.blocks[split_index:]

    # Build the model 
    comm_model.blocks = nn.Sequential(*blocks_before, communication_pipeline, *blocks_after)
    
    # # Impose the gradient to pass trough the communcation pipeline
    # def apply_gradient_pipeline(module, grad_output):

    #     # Apply the pipeline
    #     grad_output_with_pipeline = communication_pipeline(grad_output[0])

    #     return (grad_output_with_pipeline, )

    # # Register the hook on the ouptut gradient of the block before the channel 
    # comm_model.blocks[split_index - 1].register_full_backward_pre_hook(apply_gradient_pipeline)


    # Get optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,params=comm_model.parameters())

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