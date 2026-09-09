#!/bin/bash
#SBATCH --job-name=gemma3_qa
#SBATCH --time=6:00:00    
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=16G

module load ollama/0.6.0-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0


cd /scratch/s5916771/honours1/anthropomimetic-uncertainty/
source .venv/bin/ativate

ollama serve &
sleep 5
ollama pull gemma3:1b

python main.py

wait