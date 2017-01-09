#!/usr/bin/env bash
###############################
#SBATCH --job-name=small_train
#SBATCH -d singleton  
#SBATCH --output=log.train.small 
#SBATCH --error=error.train.small 

###############################
#SBATCH --time=5:00 
#SBATCH --partition k80
#SBATCH --gres=gpu:4

###############################
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haosheng@stanford.edu

###############################
module load cuda80
module load tensorflow 
python main.py --num_layers 4 --batch_size 128 --max_size 30 --emb_size 1000 --hidden_size 1000 --epochs 10000 --lr_init 1 --dataset small --patience 100000 
