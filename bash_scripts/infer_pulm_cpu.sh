#!/bin/bash

#SBATCH --job-name="pulm"
#SBATCH --time=23:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=job_bdw
#SBATCH --account=ws_00000
#SBATCH --partition=bdw

module load Python/3.8.6-GCCcore-10.2.0
#module load CUDA/11.3.0-GCC-10.2.0
#module load cuDNN/8.2.0.53-CUDA-11.3.0
#module load Workspace

source ~/02_pace/pulm/venv/bin/activate

cd ~/02_pace/pulm/scripts/rethink_ulm

python -c "import torch; print(torch.cuda.is_available())"

python ../scripts/pala_memgo+ellipse_gte+localize-2ch_script_dev_frame_batch.py
