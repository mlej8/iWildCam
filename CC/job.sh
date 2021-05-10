#!/bin/bash
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=16G                 # memory
#SBATCH --time=2-0                # A time limit of zero requests that no time limit be imposed. Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
#SBATCH --ntasks=1
#SBATCH --job-name=iWildCam
#SBATCH --output=logs/%j
#SBATCH --mail-user=er.li@mail.mcgill.ca
#SBATCH --mail-type=ALL

# script for launching an interactive session with Jupyter notebook on Compute Canada

source iWildCam-env/bin/activate

echo "Running $VIRTUAL_ENV/bin/notebook.sh"

srun $VIRTUAL_ENV/bin/notebook.sh