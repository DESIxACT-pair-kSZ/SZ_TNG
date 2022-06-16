#!/bin/bash
# Slurm sbatch options
#SBATCH -C haswell
#SBATCH -o ./hs_logs/halosky.sh.log-%j 
#SBATCH -n 4

# ex: bash SZ_TNG.sh 0 28 24 10001

# Loading the required module
source /usr/common/software/python/3.7-anaconda-2019.10/etc/profile.d/conda.sh
conda activate nbody_sbi
export PYTHONPATH="$PYTHONPATH:/opt/mods/lib/python3.6/site-packages:/opt/ovis/lib/python3.6/site-packages:/global/homes/k/kjc268/enlib:/global/homes/k/kjc268/halosky:/global/homes/k/kjc268/ostrich:/global/homes/k/kjc268/pydelfi"

# command line flags are: latin hypercube number (lh), initial snapshot number (si), final snapshot number (sf), number of bins for maps (nb)

# note: final snapshot number is the highest redshift for lightcone

# Run the script
python3 /global/homes/k/kjc268/SZ_TNG/save_fields.py -lh $1 -si $2 -sf $3
python3 /global/homes/k/kjc268/SZ_TNG/get_sz_maps.py -lh $1 -si $2 -sf $3 -nb $4
python3 /global/homes/k/kjc268/SZ_TNG/apply_beam.py -lh $1 -si $2 -sf $3 -nb $4