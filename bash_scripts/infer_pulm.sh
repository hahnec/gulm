#!/bin/bash

#SBATCH --job-name="pulm"
#SBATCH --time=12:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=job_gpu
#SBATCH --account=ws_00000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load Python/3.8.6-GCCcore-10.2.0

source ~/21_rethink_ulm/venv/bin/activate

cd ~/21_rethink_ulm/scripts

python -c "import torch; print(torch.cuda.is_available())"

python ./pala_memgo+ellipse_gte+localize-2ch_script_dev_frame_batch.py
