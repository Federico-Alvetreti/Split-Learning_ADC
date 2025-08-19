#!/bin/sh
# https://stackoverflow.com/questions/27708656/pass-command-line-arguments-via-sbatch

mkdir -p ./sout/err
mkdir -p ./sout/log

export HYDRA_FULL_ERROR=1
export TQDM_LOG=1
export TQDM_LOG_INTERVAL=100

old="$IFS"
IFS='_'
str="$*"

echo "${str}"

IFS=$old

sbatch <<EOT
#!/bin/sh
#SBATCH -A IscrC_AdvCMT
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name="splitlearning_split_${str}"
#SBATCH --out="./sout/log/splitlearning_split_${str}.out"
#SBATCH --error="./sout/err/splitlearning_split_${str}.err"

# Print debug information
echo "=== Job Information ==="
echo "NODELIST="\${SLURM_NODELIST}
echo "Job ID: "\${SLURM_JOB_ID}
echo "Parameters: ${str}"
echo "Working directory: \$(pwd)"
echo "Date: \$(date)"
echo "========================"

# Change to working directory
echo "Changing to working directory..."
cd /leonardo/home/userexternal/jpomponi/Split-Learning || {
    echo "ERROR: Cannot change to /leonardo/home/userexternal/jpomponi/Split-Learning"
    exit 1
}

# Load modules
echo "Loading modules..."
#module load anaconda3
module load cuda

conda init
# conda activate ood


echo "Activating conda environment 'split_learning'..."
source activate split_learning || {
    echo "ERROR: Failed to activate conda environment 'split_learning'"
    echo "Available environments:"
    conda env list
    exit 1
}

if [ ${1} = 'base' ]; then
  srun python main.py method=${1} dataset=${3} model=${2} hyperparameters.split_index=${5} hydra=split_ablation
elif [ ${1} = 'bottlenet' ]; then
  srun python main.py method=${1} dataset=${3} model=${2} method.parameters.compression=${4} hyperparameters.split_index=${5} hydra=split_ablation
elif [ ${1} = 'c3-sl' ]; then
  srun python main.py method=${1} dataset=${3} model=${2} method.parameters.R=${4} hyperparameters.split_index=${5} hydra=split_ablation
elif [ ${1} = 'proposal' ]; then
srun python main.py method=${1} dataset=${3} model=${2} method.parameters.desired_compression=${4} hyperparameters.split_index=${5}  +hyperparameters.seeds=42 method.parameters.pooling=${6:-attention} hydra=split_ablation
elif [ ${1} = 'quantization' ]; then
  srun python main.py method=${1} dataset=${3} model=${2} method.parameters.n_bits=${4} hyperparameters.split_index=${5} hydra=split_ablation
elif [ ${1} = 'random_top_k' ]; then
  srun python main.py method=${1} dataset=${3} model=${2} method.parameters.rate=${4} hyperparameters.split_index=${5} hydra=split_ablation
elif [ ${1} = 'top_k' ]; then
  srun python main.py method=${1} dataset=${3} model=${2} method.parameters.rate=${4} hyperparameters.split_index=${5} hydra=split_ablation
else
	echo "ERROR: method not recognized"
  exit 1
fi


# Check exit status
if [ \$? -eq 0 ]; then
    echo "Python script completed successfully"
else
    echo "ERROR: Python script failed with exit code \$?"
    exit 1
fi

echo "Job completed at: \$(date)"

EOT
