#!/bin/bash

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <config_path> [-r]"
    exit 1
fi

config_path="$1"
job_name=$(basename "$config_path" .json)

cat << EOF > "slurm/${job_name}.sh"
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=$job_name.out

ml conda
conda activate evo
export REMOTE_PREFIX="/export/b12/tbai4"
python3 train.py --config $config_path --device cuda
EOF

echo "generated: ${job_name}.sh"

if [ "$2" = "-r" ]; then
    sbatch "slurm/${job_name}.sh"
fi
