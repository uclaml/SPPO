#!/bin/bash

#SBATCH --job-name=dpo-no-ref
#SBATCH --account genai_interns
#SBATCH --qos genai_interns_low
##SBATCH --partition=learnai  # change as needed, e.g., lowpri on some clusters
##SBATCH --gres=gpu:1        # uncomment only if/as needed
#SBATCH --time=1-00:00:00    
#SBATCH --gpus-per-node=8
##SBATCH --cpus-per-task=48    # change as needed
## %j is the job id, q%u is the user id
#SBATCH --output=/data/home/%u/dpo-no-ref-%j.log

# Start clean
# module purge

# Load what we need
# module load conda

# source ~/.bashrc
source /data/home/yuewu96/miniconda3/etc/profile.d/conda.sh
# conda init bash
conda activate sppo


# cat /etc/hosts
./scripts/dpo.sh
