#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --job-name=e2d
#SBATCH --output=/home/jkirschn/scratch/jkirschn/logs/slurm-%A_%a.out
#SBATCH --array=1-25

#pm2 --outdir /home/jkirschn/scratch/jkirschn/runs semi_bandit --seed=1 ts --n=1000

# sqrt(t)
#pm2 --outdir /home/jkirschn/scratch/jkirschn/runs semi_bandit --seed=1 e2d --n=1000

# fixed lambda
#pm2 --outdir /home/jkirschn/scratch/jkirschn/runs semi_bandit --seed=1 e2d --e2d_fixed_lambda=25 --n=1000
#pm2 --outdir /home/jkirschn/scratch/jkirschn/runs semi_bandit --seed=1 e2d --e2d_fixed_lambda=2 --n=1000
#pm2 --outdir /home/jkirschn/scratch/jkirschn/runs semi_bandit --seed=1 e2d --e2d_fixed_lambda=5 --n=1000
#pm2 --outdir /home/jkirschn/scratch/jkirschn/runs semi_bandit --seed=1 e2d --e2d_fixed_lambda=10 --n=1000

pm2 --outdir /home/jkirschn/scratch/jkirschn/runs semi_bandit --seed=1 e2d --e2d_anytime=1 --n=1000


#python /home/jkirschn/projects/def-szepesva/jkirschn/flatland/evaluation_az.py -np $SLURM_ARRAY_TASK_ID -nen 15 -nep 10 -sp -g -nm -tm=tw -id="az_${tm}_nm_g_${SLURM_ARRAY_TASK_ID}" -dl=/home/jkirschn/scratch/flatland/logs/