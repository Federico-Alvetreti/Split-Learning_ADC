import torch
import torch.nn as nn
from scripts.utils import tensor_to_bytes, get_ffn
from math import floor


# Adds Additive White Gaussian Noise on tensors 
def add_awgn_noise(tensor: torch.Tensor, snr: float) -> torch.Tensor:

    # Estimate signal power along the last dim (on tokens)
    signal_power = torch.mean(tensor**2, dim=-1, keepdim=True)

    # Compute noise power for the desired SNR
    noise_power = signal_power / (10 ** (snr / 10))
    std = torch.sqrt(noise_power)

    # Sample & scale noise
    noise = torch.randn_like(tensor) * std
    noisy_tensor = tensor + noise
    
    return noisy_tensor

# Classic analogic channel 
class Gaussian_Noise_Analogic_Channel(nn.Module):
    def __init__(self,
                  snr: float):
        super().__init__()
        self.snr = snr
        self.total_communication = 0

    def forward(self, input: torch.Tensor):
        # Add noise in the forward 
        self.total_communication += tensor_to_bytes(input)
        noisy_output = add_awgn_noise(input, self.snr)

        # Simple backward noise hook 
        def _grad_hook(grad):

            noisy_grad = add_awgn_noise(grad, self.snr)
            self.total_communication += tensor_to_bytes(grad)
      
            return noisy_grad
        
        # Register hook
        if noisy_output.requires_grad:
            noisy_output.register_hook(_grad_hook)

        return noisy_output
    
# Classic analogic channel with SVD on backward 
class Gaussian_Noise_Analogic_Channel_with_SVD(nn.Module):
    def __init__(self,
                 snr: float,
                 update_frequency:int,
                 energy_threshold:float):
        super().__init__()

        # Get parameters 
        self.snr = snr
        self.energy_threshold = energy_threshold  
        self.update_frequency = update_frequency # After how many backward iterations compute other projections 
        
        # Initialize timer and total communication variables 
        self.total_communication = 0
        self.t = 0 

        # Initialize projection matrices 
        self.server_right_proj_matrix = 0
        self.server_left_proj_matrix = 0
        self.device_right_proj_matrix = 0
        self.device_left_proj_matrix = 0


    def forward(self, input: torch.Tensor):

        # Add noise in the forward 
        self.total_communication += tensor_to_bytes(input)
        noisy = add_awgn_noise(input, self.snr)

        # Backward noise hook + noise + svd 
        def _svd_grad_hook(G):

            # If the update frequency has been achieved 
            if self.t % self.update_frequency == 0 : 

                # Get the average gradient 
                G_mean = G.mean(dim=0)

                # Do the Singular Value Decomposition on the average gradient 
                U, S, V = torch.linalg.svd(G_mean, full_matrices=False)  # U: [m, min(m,n)], V: [min(m,n), n]
                energy = S.pow(2)
                total_energy = energy.sum()
                cumulative_energy = torch.cumsum(energy, dim=0)
                r = (cumulative_energy < (self.energy_threshold * total_energy)).sum().item() + 1
                
                # Update server projection matrices
                self.server_left_proj_matrix = U[:, :r]
                self.server_right_proj_matrix = V[:r, :]

                # Update device projection matrices
                self.device_left_proj_matrix = add_awgn_noise(self.server_left_proj_matrix, self.snr)
                self.device_right_proj_matrix = add_awgn_noise(self.server_right_proj_matrix, self.snr)

                # Update total communication 
                self.total_communication += tensor_to_bytes(self.server_left_proj_matrix)  
                self.total_communication += tensor_to_bytes(self.server_right_proj_matrix)

            # Compute the projected representation using server projection matrices 
            G =  torch.matmul(torch.matmul(self.server_left_proj_matrix.t().unsqueeze(0), G ), self.server_right_proj_matrix.t().unsqueeze(0))

            # Apply the channel and update communication cost 
            G = add_awgn_noise(G, self.snr)
            self.total_communication += tensor_to_bytes(G)

            # Reconstruct the gradient using device projection matrices 
            G = torch.matmul(torch.matmul(self.device_left_proj_matrix.unsqueeze(0), G ), self.device_right_proj_matrix.unsqueeze(0))

            # Update timer 
            self.t += 1 

            return G
        
        # Register hook to apply the backward function
        if noisy.requires_grad:
            noisy.register_hook(_svd_grad_hook)


        return noisy
    
# Main Encoder 
class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_layers=2,
                 normalize=True,
                 drop_last_activation=True,

                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Resolve float output_size
        self.output_size = (max(floor(input_size * output_size), 1) if isinstance(output_size, float) else output_size)

        # Get the real and complex feed forward networks 
        self.real_ffn, self.complex_ffn = get_ffn(input_size=input_size,
                                                    output_size=self.output_size,
                                                    n_layers=n_layers,
                                                    drop_last_activation=drop_last_activation,
                                                    n_copy=2)
        # Store parameters 
        self.normalize = normalize


    def forward(self, x, *args, **kwargs):

        # Get real and complex parts of the input 
        real_part, complex_part = self.real_ffn(x), self.complex_ffn(x)

        # Combine them in a single complex matrix 
        x = torch.complex(real_part, complex_part)

        # Normalize if specified  
        if self.normalize:
            x = x / torch.norm(x, 2, -1, keepdim=True)

        return x

# Main Decoder 
class Decoder(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 n_layers,
                 normalize=True,
                 drop_last_activation=False,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)


        # Get the decoding ffn 
        self.decoder_ffn = get_ffn(input_size=input_size,
                                        output_size=output_size,
                                        n_layers=n_layers,
                                        drop_last_activation=drop_last_activation,
                                        n_copy=1)

        self.normalize = normalize

    def forward(self, x, *args, **kwargs):

        # Normalize if specified  
        if self.normalize:
            x = x / torch.norm(x, 2, -1, keepdim=True)

        # Concat real and imaginary part 
        x = torch.cat((x.real, x.imag), -1)

        # Decode back 
        x = self.decoder_ffn(x)

        return x
