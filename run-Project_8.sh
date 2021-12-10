#!/usr/bin/bash
#
# Author: Dongming Yang
# Updated: December 09, 2021
# 79: -------------------------------------------------------------------------

# slurm options: --------------------------------------------------------------
#SBATCH --job-name=Project8.py
#SBATCH --mail-user=dongming@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5GB 
#SBATCH --time=120:00
#SBATCH --account=cscar
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/%x-%j-4.log

# application: ----------------------------------------------------------------
n_procs=5

# modules 
module load tensorflow

# the contents of this script
cat run-Project_8.sh

# run the script
date

cd /home/dongming/stats_507/the_great_lake
python Project_8.py $n_procs

date
echo "Done."
