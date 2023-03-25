#!/bin/bash

#SBATCH --job-name="pulm"
#SBATCH --time=23:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=ws_00000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gtx1080ti:1

module load Python/3.8.6-GCCcore-10.2.0

source ~/21_rethink_ulm/venv/bin/activate

cd ~/21_rethink_ulm/scripts

python -c "import torch; print(torch.cuda.is_available())"

python ./invivo_memgo+ellipse_gte+localize-2ch_script_dev_frame_batch.py
