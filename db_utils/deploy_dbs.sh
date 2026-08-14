#!/bin/bash
#SBATCH --job-name=neo4j_fleet
#SBATCH --partition=cpu-2d
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48GB
#SBATCH --output=fleet_%j.log

echo "Starting DB fleet..."
srun --ntasks-per-node=1 python3 -u start_neo4j.py
