#!/bin/bash

#SBATCH --job-name=docbot_dnop

#SBATCH --output=/home/%u/docbot/out/dnop/out/%j_%x.out

#SBATCH -A CS156b
#SBATCH -t 24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

UNAME=bfreeman

cd /home/"${UNAME}"/docbot
mkdir out
source /groups/CS156b/2022/venvs/docbotvenv/bin/activate
python3 src/dnop/dnop.py $1 $2
deactivate
