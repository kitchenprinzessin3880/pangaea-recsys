#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks-per-node=20

module load python/3.6.3
virtualenv --system-site-packages test
source test/bin/activate
pip install tables
python ~/pangaea-recsys/usage_analysis/usage_scripts/extractlogs_downloads_numpy_chunks.py -c ~/pangaea-recsys/usage_analysis/config/usage_linux.ini
