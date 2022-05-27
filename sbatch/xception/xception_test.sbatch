#!/bin/bash

# comments start with a pound followed by a space
# lines beginning with #SBATCH specified job parameters

#SBATCH --job-name=xception_test

# direct standard out and standard error of the job
# %x: job name, %j: job ID, %u: username
#SBATCH --output=/home/%u/docbot/out/%x_%j_%u.out
#SBATCH --error=/home/%u/docbot/out/%x_%j_%u.err

# tells Slurm to bill CS 156b for the job
#SBATCH -A CS156b

# estimated time to run the job (Slurm will kill the job if this limit is exceeded)
# tip: start off by overestimating and adjust as you get a better sense
#     too high: will take a long time to get scheduled
#     too low: will get killed off before completion
#SBATCH -t 6:00:00

# number of concurrent srun taks (likely will not need to modify)
#SBATCH --ntasks=1

# number of CPU threads for each task
#SBATCH --cpus-per-task=1

# total amount of system RAM for all tasks (specified with M or G)
#SBATCH --mem=32G

# request a single Tesla P100 GPU
# maximum per node is 4
# can request a newer V100, but there are only two, so they are hard to get
#SBATCH --gres=gpu:1

# get notified via email when the job begins, ends, and/or fails
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# rest of file is standard bash
UNAME=nmcalister01

# setup a Python environment and run a script (for example)
cd /home/"${UNAME}"/docbot
mkdir out
source /groups/CS156b/2022/venvs/docbotvenv/bin/activate
python3 src/xception/xception_test.py
deactivate
