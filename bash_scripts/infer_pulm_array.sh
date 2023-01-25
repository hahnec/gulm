#!/bin/bash

#SBATCH --job-name="pulm_array"
#SBATCH --time=15:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=job_gpu
#SBATCH --account=ws_00000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%1

module load Python/3.8.6-GCCcore-10.2.0
#module load CUDA/11.3.0-GCC-10.2.0
#module load cuDNN/8.2.0.53-CUDA-11.3.0
#module load Workspace

source ~/02_pace/pulm/venv/bin/activate

param_store=param.txt

ch_gap=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
enlarge_factor=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')
cluster_number=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $3}')
max_iter=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $4}')

cd ~/02_pace/pulm/scripts/rethink_ulm

python -c "import torch; print(torch.cuda.is_available())"

echo "Channel gap: ${ch_gap}, Upsampling factor: ${enlarge_factor}, Cluster number:${cluster_number}, Max-iter: ${max_iter}"
python ./pala_memgo+ellipsoid+localize-2ch_script.py ch_gap=${ch_gap} enlarge_factor=${enlarge_factor} cluster_number=${cluster_number} max_iter=${max_iter} dat_num=2
