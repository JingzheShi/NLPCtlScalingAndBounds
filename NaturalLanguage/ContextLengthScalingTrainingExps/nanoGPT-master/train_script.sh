#!/bin/bash -l
#SBATCH --job-name=Training0712 # Job name
#SBATCH --output=Training0712.o%j # Name of stdout output file
#SBATCH --error=Training0712.e%j # Name of stderr error file
#SBATCH --partition=standard-g # partition name
#SBATCH --nodes=1 # Total number of nodes
#SBATCH --ntasks-per-node=1 # 8 MPI ranks per node, 8 total (1x8)
#SBATCH --gpus-per-node=8 # Allocate one gpu per MPI rank
#SBATCH --time=12:00:00 # Run time (d-hh:mm:ss)
#SBATCH --account=project_xxxxxxxxxxx # Project for billing

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

CONTEXT_LEN=$1
PERCENT_NAME=$2


CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
export MPICH_GPU_SUPPORT_ENABLED=1
source /scratch/project_xxxxxxxxxxx/venv/bin/activate
OMP_NUM_THREADS=4
export OMP_NUM_THREADS=4
srun --cpu-bind=${CPU_BIND} torchrun --standalone --nproc_per_node=8 train.py --compile=False config/LONGCONTEXT_train_gpt2_"$CONTEXT_LEN"ctl_"$PERCENT_NAME"pct_small.py > Training_longcontext_fp32_small_"$CONTEXT_LEN"ctl_"$PERCENT_NAME".txt