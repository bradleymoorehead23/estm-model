#!/bin/bash
#SBATCH --job-name=hvqvae        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=38G                # total memory per node
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=bcm2@princeton.edu

module purge
module load anaconda3/2022.10
module load cudatoolkit/12.0
module load cudnn/cuda-11.x/8.2.0
conda activate tf2-gpu

python h_vq_vae.py