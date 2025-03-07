import torch
from torch import nn
from copy import deepcopy
from typing import Union, Tuple, Sequence
import numpy as np
import hydra 
from utils import training_schedule, load_pretrained_model

# Analogic Gaussian Noise Channel 
class Gaussian_Noise_Analogic_Channel(nn.Module):
    def __init__(self, snr: Union[float, Tuple[float, float]], dims=-1):
        super().__init__()

        assert snr is not None, "SNR must be specified."
        self.snr = snr  # Single SNR value or range
        self.dims = dims  # Dimension for power calculation
        self.total_communication = 0

    def get_snr(self, batch_size: int, device: Union[str, torch.device] = "cpu"):
        """Returns a fixed or sampled SNR value."""
        if isinstance(self.snr, Sequence):  # If it's a range (tuple), sample uniformly
            r1, r2 = self.snr
            return torch.rand(batch_size, device=device) * (r1 - r2) + r2
        return self.snr  # Return fixed value if not a range

    def apply_noise(self, x, signal_power, snr):
        """Applies Gaussian noise based on SNR."""

        if isinstance(snr, torch.Tensor):
            snr = snr.view([-1] + [1] * (x.ndim - 1))  # Expand for broadcasting

        noise_power = signal_power / (10 ** (snr / 10))
        std = torch.sqrt(noise_power)
        noise = torch.randn_like(x) * std

        return x + noise

    def forward(self, x: torch.Tensor, snr=None):
        
        """Adds Gaussian noise to the input tensor based on the given SNR."""
        # Add the total communication cost 
        self.total_communication += torch.prod(torch.tensor(x.size())).item()

        # Get snr value
        snr = snr if snr is not None else self.get_snr(len(x), x.device)
        
        if isinstance(snr, torch.Tensor):
            snr = snr.to(x.device)
    
        # Compute signal power along the given dimensions
        signal_power = torch.mean(x ** 2, dim=self.dims, keepdim=True)  

        # Apply noise
        noisy_signal = self.apply_noise(x, signal_power, snr)  # Apply noise

        return noisy_signal
    
# Functions used to create encoder / decoder 
def get_layers(input_size, output_size=1.0, n_layers=2, n_copy=1, drop_last_activation=False):
    if isinstance(output_size, float):
        output_size = max(int(input_size * output_size), 1)

    shapes = np.linspace(input_size, output_size, num=n_layers + 1, endpoint=True, dtype=int)

    model = []

    for s in range(len(shapes) - 1):
        model.append(nn.Linear(shapes[s], shapes[s + 1]))
        model.append(nn.ReLU())

    if drop_last_activation:
        model = model[:-1]

    model = nn.Sequential(*model)

    models = []
    for _ in range(n_copy):
        _model = deepcopy(model)

        for m in _model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        models.append(_model)

    return shapes[-1], models

def get_cnn_layers(input_size, output_size=1.0, n_layers=2, n_copy=1, drop_last_activation=True):
    def get_sizes(os, ins, inverse=False):
        mn = (np.inf, -1, -1)
        for i in range(1, 12):
            for j in range(1, 12):

                if not inverse:
                    s = np.floor((ins - (i - 1) - 1) / j + 1)
                else:
                    s = np.floor(((ins - 1) * j + (i - 1))) + 1

                d = abs(s - os)

                if d < mn[0]:
                    mn = (d, i, j, s)

        return mn[1:]

    input_channels, w, h = input_size
    out_channels = input_channels

    if isinstance(output_size, tuple):
        out_channels = output_size[0]
        output_size = output_size[1]

    if isinstance(output_size, float):
        output_size = max(int(output_size * w), 1)

    shapes = numpy.stack((np.linspace(w, output_size, num=n_layers + 1, endpoint=True, dtype=int),
                          np.linspace(input_channels, out_channels, num=n_layers + 1, endpoint=True, dtype=int)), 1)

    model = []
    os = None

    for s in range(len(shapes) - 1):

        if shapes[s + 1][0] < shapes[s][0]:
            ks, stride, size = get_sizes(shapes[s + 1][0], shapes[s][0])
            model.append(nn.Conv2d(shapes[s][1], shapes[s+1][1], kernel_size=ks, stride=stride))
            model.append(nn.ReLU())
        else:
            ks, stride, size = get_sizes(shapes[s + 1][0], shapes[s][0], inverse=True)
            model.append(nn.ConvTranspose2d(shapes[s][1], shapes[s+1][1], kernel_size=ks, stride=stride))
            model.append(nn.ReLU())

    if drop_last_activation:
        model = model[:-1]

    model = nn.Sequential(*model)

    models = []
    for _ in range(n_copy):
        _model = deepcopy(model)

        for m in _model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        models.append(_model)

    return shapes[-1], models

# Main Encoder 
class BaseRealToComplexNN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_layers=2,
                 normalize=True,
                 transpose=False,
                 drop_last_activation=False,
                 sincos=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(input_size, tuple):
            output_size, (cc, rr) = get_cnn_layers(input_size=input_size, output_size=output_size,
                                                   n_layers=n_layers, drop_last_activation=drop_last_activation,
                                                   n_copy=2)
        else:
            output_size, (cc, rr) = get_layers(input_size=input_size, output_size=output_size,
                                               n_layers=n_layers, drop_last_activation=drop_last_activation,
                                               n_copy=2)

        self.r_fl, self.c_fl = rr, cc

        self.normalize = normalize
        self.transpose = transpose
        self.sincos = sincos
        self.output_size = output_size

    def forward(self, x, *args, **kwargs):
        if self.transpose:
            x = x.permute(0, 2, 1)

        a, b = self.r_fl(x), self.c_fl(x)

        if self.sincos:
            a, b = torch.cos(a), torch.sin(b)

        x = torch.complex(a, b)

        if self.normalize:
            x = x / torch.norm(x, 2, -1, keepdim=True)

        return x
    
# Main Decoder 
class ConcatComplexToRealNN(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 n_layers=2,
                 normalize=False,
                 transpose=False,
                 drop_last_activation=True,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)


        if isinstance(input_size, tuple):
            self.cdim = 1
            input_size = (input_size[0]*2, input_size[1], input_size[2])
            output_size, (cc, _) = get_cnn_layers(input_size=input_size, output_size=output_size[:2],
                                                  n_layers=n_layers, drop_last_activation=drop_last_activation,
                                                  n_copy=2)
        else:
            self.cdim = -1
            output_size, (cc, _) = get_layers(input_size=input_size * 2, output_size=output_size,
                                              n_layers=n_layers, drop_last_activation=drop_last_activation,
                                              n_copy=2)

        self.d_f = cc
        self.transpose = transpose
        self.normalize = normalize

    def forward(self, x=None, *args, **kwargs):
        if self.normalize:
            x = x / torch.norm(x, 2, -1, keepdim=True)

        x = torch.cat((x.real, x.imag), self.cdim)
        x = self.d_f(x)

        if self.transpose:
            x = x.permute(0, 2, 1)

        return x

class NormalizeDenormalizeModule(nn.Module):
    def __init__(self):
        super(NormalizeDenormalizeModule, self).__init__()

    def normalize(self, x):
        """Normalize the tensor by its mean and std."""
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-8), mean, std  # Return mean and std for denormalization

    def denormalize(self, x, mean, std):
        """Denormalize the tensor."""
        return x * std + mean

class CommunicationPipeline(nn.Module):
    def __init__(self, encoder, channel, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = encoder if encoder is not None else lambda x: x
        self.decoder = decoder if decoder is not None else lambda x: x
        self.channel = channel if channel is not None else lambda x: x
        # self.norm_denorm = NormalizeDenormalizeModule()  # Add the normalization/denormalization module

    def forward(self, x, snr=None):
        # self.input = x

        # Normalize input before passing through the pipeline
        # normalized_x, mean, std = self.norm_denorm.normalize(x)

        # Pass through encoder, channel, and decoder
        x = self.encoder(x)
        x = self.channel(x)
        x = self.decoder(x)

        # Denormalize output after processing
        # x = self.norm_denorm.denormalize(x, mean, std)

        return x
    

# Train communication forward 
def train_forward_communication_pipeline(cfg): 


    print("\n\nTRAINING FORWARD AUTOENCODER\n\n")

    # Get split index 
    split_index = cfg.hyperparameters.split_index

    # Set device  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get hyperparameters 
    batch_size = cfg.schema.batch_size
    epochs = cfg.comm.channel_training_epochs

    # Get datasets 
    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test)

    # Get dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True,batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)

    # Load pre-trained model 
    pretrained_model = load_pretrained_model(cfg)

    # Freeze pre-trained model parameters 
    for param in pretrained_model.parameters():
        param.requires_grad = False
    
    # Get communication pipeline 
    encoder = hydra.utils.instantiate(cfg.comm.encoder, input_size=pretrained_model.num_features)

    decoder = hydra.utils.instantiate(cfg.comm.decoder, input_size=encoder.output_size, output_size=pretrained_model.num_features)

    channel = hydra.utils.instantiate(cfg.comm.channel)
    
    communication_pipeline = CommunicationPipeline(channel=channel, encoder=encoder, decoder=decoder).to(device)  

    # Train  communication_pipeline parameters 
    for param in communication_pipeline.parameters():
        param.requires_grad = True

    # Create the comm_model 
    blocks_before = pretrained_model.blocks[:split_index]
    blocks_after = pretrained_model.blocks[split_index:]

    pretrained_model.blocks = nn.Sequential(*blocks_before, communication_pipeline, *blocks_after)

    # Get optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=pretrained_model.parameters())

    # Train the network 
    _ = training_schedule(pretrained_model, train_dataloader, test_dataloader, optimizer, epochs, device, plot = True)

    # Get the trained communication pipeline 
    trained_communication_pipeline = pretrained_model.blocks[split_index]

    return trained_communication_pipeline


# Train communication backward 
def train_backward_communication_pipeline(cfg): 

    # Get split index 
    split_index = cfg.hyperparameters.split_index

    loss = torch.nn.CrossEntropyLoss()

    # Set device  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get hyperparameters 
    batch_size = cfg.schema.batch_size
    epochs = cfg.comm.channel_training_epochs

    # Get datasets 
    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test)

    # Get dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True,batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)

    # Load pre-trained model 
    pretrained_model = load_pretrained_model(cfg)

    # Get communication pipeline 
    encoder = hydra.utils.instantiate(cfg.comm.encoder, input_size=pretrained_model.num_features)

    decoder = hydra.utils.instantiate(cfg.comm.decoder, input_size=encoder.output_size, output_size=pretrained_model.num_features)

    channel = hydra.utils.instantiate(cfg.comm.channel)
    
    communication_pipeline = CommunicationPipeline(channel=channel, encoder=encoder, decoder=decoder)   

    # Get optimizers
    comm_optimizer = hydra.utils.instantiate(cfg.optimizer, params=communication_pipeline.parameters())
    pretrained_optimizer = hydra.utils.instantiate(cfg.optimizer, params=pretrained_model.parameters())

    # Hook to store gradients 
    def hook_fn(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Register hook 
    pretrained_model.blocks[split_index - 1].register_backward_hook(hook_fn)

    # For each epoch
    for epoch in range(1, epochs + 1, 1):
        
        # Training  phase 
        communication_pipeline.train()

        # Forward the train set
        for batch in train_dataloader:

            # List to store gradient
            gradients = []

            # Get input and labels from batch
            batch_input = batch[0].to(device)
            batch_labels = batch[1].to(device)

            # Compute last layers, get batch predictions
            batch_predictions = pretrained_model(batch_input)

            # Get batch loss and accuracy
            batch_loss = loss(batch_predictions, batch_labels)
            
            # Backpropagation
            batch_loss.backward()
            
            # Zero out the gradients 
            pretrained_optimizer.zero_grad()
            
            # Get gradient
            grad_features = gradients[0]
            
            # Forward through communication pipeline
            reconstructed_grads = communication_pipeline(grad_features)
            
            # Compute loss
            comm_loss = nn.MSELoss(reconstructed_grads, grad_features)
            comm_loss.backward()
            comm_optimizer.step()
            
            total_train_loss += comm_loss.item()
        
        # Validation phase 
        communication_pipeline.eval()

        # Forward the train set
        for batch in test_dataloader:

            # List to store gradient
            gradients = []

            # Get input and labels from batch
            batch_input = batch[0].to(device)
            batch_labels = batch[1].to(device)

            # Compute last layers, get batch predictions
            batch_predictions = pretrained_model(batch_input)

            # Get batch loss and accuracy
            batch_loss = loss(batch_predictions, batch_labels)
            
            # Backpropagation
            batch_loss.backward()
            
            # Zero out the gradients 
            pretrained_optimizer.zero_grad()
            
            # Get gradient
            grad_features = gradients[0]
            
            # Forward through communication pipeline
            reconstructed_grads = communication_pipeline(grad_features)
            
            # Compute loss
            comm_loss = nn.MSELoss(reconstructed_grads, grad_features)
            
            total_val_loss += comm_loss.item()
        
        print(f"Train Loss: {total_train_loss / len(train_dataloader)}, Val Loss: {total_val_loss / len(test_dataloader)}, ")
    
    
    return communication_pipeline

