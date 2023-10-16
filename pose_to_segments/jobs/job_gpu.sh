#!/bin/bash
#
#
### comment lines start with ## or #+space
### slurm option lines start with #SBATCH


#SBATCH --job-name=baseline  	## job name
#SBATCH --time=0-23:00:00       ## days-hours:minutes:seconds
###SBATCH --mem=256000             ##   3GB ram (hardware ratio is < 4GB/core)
#SBATCH --mem=128000             ##   3GB ram (hardware ratio is < 4GB/core)

### SBATCH --output=job.out	## standard out file
#SBATCH --ntasks=1            ## Ntasks.  default is 1.
#SBATCH --cpus-per-task=1	## Ncores per task.  Use greater than 1 for multi-threaded jobs.  default is 1.
#SBATCH --account=iict-sp2.volk.cl.uzh    ## only need to specify if you belong to multiple tenants on ScienceCluster
###SBATCH --gres gpu:V100:1
###SBATCH --gres gpu:A100:1
#SBATCH --gres gpu:1
#SBATCH --constraint=GPUMEM32GB|GPUMEM80GB

module load anaconda3
source activate trans
# pip install .[dev]

export MEDIAPI_PATH=/shares/volk.cl.uzh/amoryo/datasets/mediapi/mediapi-skel.zip
export MEDIAPI_POSE_PATH=/shares/volk.cl.uzh/amoryo/datasets/mediapi/mediapipe_zips.zip

srun $@
