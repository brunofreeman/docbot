#!/bin/bash

#SBATCH --job-name=docbot_rnop

#SBATCH --output=/home/%u/docbot/out/rnop/out/%j_%x.out

#SBATCH -A CS156b
#SBATCH -t 72:00:00
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
python3 src/rnop/rnop.py $1 $2
deactivate
