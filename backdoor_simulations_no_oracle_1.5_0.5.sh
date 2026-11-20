#!/bin/bash
#
# comment out the memory and cpu requirements
##SBATCH --mem=10G
##SBATCH --cpus-per-task=6 

# comment out to request time limit of two days
##SBATCH --time=2-00:00:00

# specify the number of tasks in the array
#SBATCH --array=0-199

# name the output and error files
#SBATCH --output=backdoor_no_oracle_1.5_0.5-%a.log
#SBATCH --error=backdoor_no_oracle_1.5_0.5-%a.err

module load python/3.11.8

# run the script using the job number as the seed
python3 backdoor_experiments.py $SLURM_ARRAY_TASK_ID 0 1.5 0.5
