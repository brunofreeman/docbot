#!/bin/bash

#SBATCH --job-name=docbot_one_per_predict

#SBATCH --output=/home/%u/docbot/out/%x_%j_%u.out

#SBATCH -A CS156b
#SBATCH -t 48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

UNAME=bfreeman

cd /home/"${UNAME}"/docbot
mkdir out
source /groups/CS156b/2022/venvs/docbotvenv/bin/activate
python3 src/one_per/one_per_predict.py
deactivate
