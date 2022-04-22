# HPC Reference

## Logging In
- `ssh username@login.hpc.caltech.edu`
- Need Access password and 2FA

## Storage
- Each user home directory is allotted 50 GB
- `/groups/CS156b/` is allotted 300 TB to be shared among the class
	- Need to email `cwang7@caltech.edu` to be granted access permissions (include Access username in email)

## Module System
- `module avail [search term]`: prints availble modules on the HPC (optionally filtered by `search term`)
- `module load <module name>`: loads in the specified module
- `module unload <module name>`: unloads the specified module
- `module list`: lists the user's loaded modules

## Python Virtual Environment
- `source /groups/CS156b/2022/venvs/docbotvenv/bin/activate`: activate the `docbot` virtual environment
- `deactivate`: deactivate the `docbot` virtual environment
- `pip install -r requirements.txt` (from `home/username/docbot/`): ensure all virtual environment requirements are satisfied
  - To get a new package, add a line to `requirements.txt` (syntax `package~=1.2.3`) and run the above command again to install.

## Slurm
### Slurm Commands
- `sbatch <my_job.sbatch>`: submit `my_job.sbatch` (see below for format) to Slurm for running
- `squeue [-u uname]`: look at what is currently queued on the HPC (optionally filtered by username)
	- `JOBID`: unique  of job, used for killing the job
	- `NAME`: string name provided to Slurm in file descriptor
	- `USER`: user who submitted the job
	- `NODELIST`: job status
		- `(Priority)`: not running because there is a job running with higher priority
		- `(Resource)`: not running because not enough resrouces are available
		- `TIME`: real-time job running length (`D-HH:M:SS` format)
		- `hpc-[node info],...`: job is currently running on the listed nodes
- `scancel <id>`: cancel job with `JOBID=id` that you submitted
- `scontrol`: change parameters of jobs that have already been submitted

### `.sbatch` Files
`.sbatch` files are used to submit jobs to Slurm.
```bash
#!/bin/bash

# comments start with a pound followed by a space
# lines beginning with #SBATCH specified job parameters

#SBATCH --job-name=descriptive_job_name

# direct standard out and standard error of the job
# %x: job name, %j: job ID, %u: username
#SBATCH --output=/home/%u/docbot/out/%x_%j_%u.out
#SBATCH --output=/home/%u/docbot/out/%x_%j_%u.err

# tells Slurm to bill CS 156b for the job
#SBATCH -A CS156b

# estimated time to run the job (Slurm will kill the job if this limit is exceeded)
# tip: start off by overestimating and adjust as you get a better sense
#     too high: will take a long time to get scheduled
#     too low: will get killed off before completion
#SBATCH -t 1:30:00

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
UNAME=uname

# load modules needed
module load <module 1>
...
module load <module n>

# setup a Python environment and run a script (for example)
cd /home/"${UNAME}"/docbot
mkdir out
source /groups/CS156b/2022/venvs/docbotvenv/bin/activate
python3 src/docbot_train.py
deactivate
```

## Interactive Sessions
Interactive sessions can be set up via Jupyter Notebooks [here](https://interactive.hpc.caltech.edu/).
See timestamp 27:00 of the [HPC Demo](https://piazza.com/caltech/spring2022/cs156b/resources).
