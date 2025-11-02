#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=h100-80:2
#SBATCH --output=dp_dual_gpu_output_%j.log      # Standard output file (%j will be replaced with job ID)
#SBATCH --error=dp_dual_gpu_error_%j.log        # Standard error file (%j will be replaced with job ID)

# load conda
module load anaconda3/2024.10-1

# activate environment
conda activate minitorch-cuda-2
nvidia-smi

cd /jet/home/tatavart/11986-llmsys/llmsys_f25_hw5

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Run dual GPU data parallel training (world_size=2, batch_size=128)
echo "Running dual GPU experiment (world_size=2, batch_size=128)"
python3 project/run_data_parallel.py --world_size 2 --batch_size 128 --n_epochs 10

echo "Job finished at: $(date)"

