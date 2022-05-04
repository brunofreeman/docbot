#!/bin/bash

#SBATCH --job-name=docbot_resnet_test_predict_smaller
#SBATCH --output=/home/%u/docbot/out/%x_%j_%u.out

#SBATCH -A CS156b
#SBATCH -t 2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

UNAME=mbatchev

cd /home/"${UNAME}"/docbot
source /groups/CS156b/2022/venvs/docbotvenv/bin/activate
python3 src/ResNet/resnet_test_predict.py
deactivate