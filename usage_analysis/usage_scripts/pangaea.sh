#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=25GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20

module load python/3.6.1
python3 extractlogs_downloads_numpy_chunks_lnx.py -c usage.ini
