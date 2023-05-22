#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --job-name=pm2-aggr
#SBATCH --output=/home/jkirschn/scratch/jkirschn/logs/slurm-%A_%a.out

pm2-aggr /home/jkirschn/scratch/jkirschn/runs regret

hn/scratch/flatland/logs/