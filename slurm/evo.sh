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
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=$job_name.out

ml anaconda
conda activate evo
python3 evo.py --config $config_path --device cuda
EOF

echo "generated: ${job_name}.sh"

if [ "$2" = "-r" ]; then
    sbatch "slurm/${job_name}.sh"
fi
