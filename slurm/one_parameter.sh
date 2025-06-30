#!/bin/sh
# https://stackoverflow.com/questions/27708656/pass-command-line-arguments-via-sbatch

mkdir -p ./sout/err
mkdir -p ./sout/log

export HYDRA_FULL_ERROR=1
export TQDM_LOG=1
export TQDM_LOG_INTERVAL=100

sbatch <<EOT
#!/bin/sh
#SBATCH -A IscrC_AdvCMT
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name="splitlearning_${1}_${2}_${3}"
#SBATCH --out="./sout/log/advtra_${1}_${2}_${3}.out"
#SBATCH --error="./sout/err/advtra_${1}_${2}_${3}.err"

# Print debug information
echo "=== Job Information ==="
echo "NODELIST="\${SLURM_NODELIST}
echo "Job ID: "\${SLURM_JOB_ID}
echo "Parameters: ${1} ${2} ${3}"
echo "Working directory: \$(pwd)"
echo "Date: \$(date)"
echo "========================"

# Change to working directory
echo "Changing to working directory..."
cd /leonardo/home/userexternal/jpomponi/Split || {
    echo "ERROR: Cannot change to /leonardo/home/userexternal/jpomponi/Split-Learning"
    exit 1
}

# Load modules
echo "Loading modules..."
module load anaconda3
module load cuda

conda init
#conda activate ood

echo "Activating conda environment 'ood'..."
source activate ood || {
    echo "ERROR: Failed to activate conda environment 'ood'"
    echo "Available environments:"
    conda env list
    exit 1
}

srun python main.py method=${1} dataset=${2} model=${3} device=0


# Check exit status
if [ \$? -eq 0 ]; then
    echo "Python script completed successfully"
else
    echo "ERROR: Python script failed with exit code \$?"
    exit 1
fi

echo "Job completed at: \$(date)"

EOT
