#!/bin/bash

#SBATCH --job-name="pulm_render"
#SBATCH --time=12:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --qos=job_bdw
#SBATCH --account=ws_00000
#SBATCH --partition=bdw

module load Python/3.8.6-GCCcore-10.2.0

source ~/21_rethink_ulm/venv/bin/activate

cd ~/21_rethink_ulm/scripts

python ./tracking_render.py
