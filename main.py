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
    encoder = hydra.utils.instantiate(cfg.comm.encoder, input_size= pretrained_model.num_features) if cfg.comm.encoder else None

    # Get channel 
    channel = hydra.utils.instantiate(cfg.comm.channel)

    # Get decoder 
    decoder = hydra.utils.instantiate(cfg.comm.decoder, input_size=encoder.output_size, output_size=pretrained_model.num_features) if cfg.comm.decoder else None

    # Get pipelines 
    forward_communication_pipeline = CommunicationPipeline(channel=channel, encoder=encoder, decoder=decoder).to(device)
    backward_communication_pipeline = deepcopy(forward_communication_pipeline)


    # Get optimizers
    forward_communication_optimizer = hydra.utils.instantiate(cfg.optimizer,params=forward_communication_pipeline.parameters())
    backward_communication_optimizer = hydra.utils.instantiate(cfg.optimizer,params=backward_communication_pipeline.parameters())
    pretrained_model_optimizer =  hydra.utils.instantiate(cfg.optimizer,params=pretrained_model.parameters())

    # Get the split index 
    split_index = cfg.hyperparameters.split_index

    # Train the encoder / decoder 
    _ =  train_communication_pipeline(pretrained_model,
                                forward_communication_pipeline,
                                backward_communication_pipeline,
                                pretrained_model_optimizer,
                                forward_communication_optimizer,
                                backward_communication_optimizer,
                                split_index, 100,
                                train_dataloader, test_dataloader, 
                                20, device, loss = torch.nn.CrossEntropyLoss())


    
    
    del pretrained_model  # Delete the model
    del _                 # Delete the results 
    torch.cuda.empty_cache()  # Free up GPU memory



    # Initialize communication model  
    comm_model = hydra.utils.instantiate(cfg.model).to(device)

    # Apply communcation pipeline to both the forward and the backward

    # Impose the gradient to pass trough the communcation pipeline
    def apply_gradient_pipeline(module, grad_output):

        # Apply the pipeline
        grad_output_with_pipeline = backward_communication_pipeline(grad_output[0])
        
        # Must return as a Tuple so the ","
        return (grad_output_with_pipeline, )
    
    def apply_forward_pipeline(module, args, output):

        # Apply the pipeline
        output_with_pipeline = forward_communication_pipeline(output)

        return output_with_pipeline
    
    # Register the hooks 
    comm_model.blocks[split_index - 1].register_forward_hook(apply_forward_pipeline)
    comm_model.blocks[split_index - 1].register_full_backward_pre_hook(apply_gradient_pipeline)

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