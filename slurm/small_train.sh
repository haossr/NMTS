#!/usr/bin/env bash
###############################
#SBATCH --job-name=small_train
#SBATCH -d singleton  
#SBATCH -u 
#SBATCH --output=log.train.small 
#SBATCH --error=error.train.small 

###############################
#SBATCH --time=1:00:00 
#SBATCH --partition k80
#SBATCH --gres=gpu:4

###############################
# --mail-type=END,FAIL
#SBATCH --mail-user=haosheng@stanford.edu

###############################
module load cuda80
module load tensorflow 
python main.py --num_layers 4 --batch_size 128 --max_size 30 --emb_size 1000 --hidden_size 1000 --epochs 1 --lr_init 1 --dataset small --patience 100000 --reload 1 
