
# Communication Efficient Split Learning of ViTs with Attention-based Double Compression

This paper proposes a novel communication-efficient Split Learning (SL) framework, named Attention-based Double Compression (ADC), which reduces the communication overhead required for transmitting intermediate Vision Transformers activations during the SL training process. ADC incorporates two parallel compression strategies. The first one merges samples’ activations that are similar, based on the average attention score calculated in the last client layer; this strategy is class-agnostic, meaning that it can also merge samples having different classes, without losing generalization ability nor decreasing final results. The second strategy follows the first and discards the least meaningful tokens, further reducing the communication cost. Combining these strategies not only allows for sending less during the forward pass, but also the gradients are naturally compressed, allowing the whole model to be trained without additional tuning or approximations of the gradients. Simulation results demonstrate that Attention-based Double Compression outperforms state-of-the-art SL frameworks by significantly reducing communication overheads while maintaining high accuracy

## Requirements and Code structure

We suggest creating a [conda](https://conda.io/) environment and installing the packages in the requirements.txt file using pip 3. 

Our adaptive ViT proposal can be find in `methods/proposal` file. It acts as a wrapper for a hugging-face ViT model. The folder also contains the other methods we used as baselines.

Entry points for our code is the file `main.py`. It can be used to train all models using the methods.

## Running the experiments

The structure for running all experiments is based on [Hydra](https://pypi.org/project/hydra-core/). Config folder contains all the necessary files to run all the experiments present in the paper. 

To run the experiments we used bash files to run slurm jobs. The main entry points is the file `./slurm/main.sh`, used to run all base experiments.

You can find slurm files in the folders `./slurm/ablations` and `./slurm/bash`. Specifically, the former contains slurm files used to launch all the ablations, while the latter contains files to run all base experiments.  

Use them as reference for running yours. You can easily extend this work or test other configurations by adding or modifying config files.
