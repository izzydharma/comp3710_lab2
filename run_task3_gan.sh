#!/bin/bash
#SBATCH --job-name=lab2-gan
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Submit from ~/Lab_2 with: sbatch run_task3_gan.sh
# Four hours is an initial limit; adjust it using measured training times.
# Use the partition memory default: Rangpur currently advertises only 1 MB per node.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this file using sbatch from your Lab_2 folder}"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
python -u - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("Built for CUDA:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable on this allocated node. Save this job's .out and .err "
        "files to diagnose the PyTorch/driver compatibility before retrying."
    )
print("GPU:", torch.cuda.get_device_name(0))
PY

python -u task3_gan.py --device cuda --workers "${SLURM_CPUS_PER_TASK:-4}"
echo "Finished $(date)"
