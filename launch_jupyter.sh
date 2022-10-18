#!/bin/bash
#SBATCH --job-name=JupLab
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --qos=6hours    # 30min, 6hours, 1day, 1week, infinite  --> 6hours default, slurm is backfilling so be specific with time
#SBATCH -o JupLab-%J.oe

# activate a conda environment of your choice (here we use base)
# conda init bash
# conda activate myenv
#source activate myenv
source $HOME/miniconda3/bin/activate myenv

## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

## print tunneling instructions to STDOUT
echo -e "
     Copy/Paste this in your local terminal to ssh tunnel with remote
-----------------------------------------------------------------
     ssh -N -L $ipnport:$ipnip:$ipnport panile54@login.scicore.unibas.ch
-----------------------------------------------------------------

     Then open a browser on your local machine to the following address
------------------------------------------------------------------
     https://localhost:$ipnport  (see log file for token)
------------------------------------------------------------------
     "

## start an ipcluster instance and launch jupyter server
jupyter lab --no-browser --port=$ipnport --ip=$ipnip --keyfile=$HOME/.mycerts/mycert.key --certfile=$HOME/.mycerts/mycert.pem

