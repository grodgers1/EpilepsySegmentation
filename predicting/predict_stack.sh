#!/bin/bash
#SBATCH --job-name=predict_stack     #Name of your job
#SBATCH --cpus-per-task=1    #Number of cores to reserve
#SBATCH --mem-per-cpu=32G     #Amount of RAM/core to reserve
#SBATCH --time=1-00:00:00      #Maximum allocated time
#SBATCH --qos=1day         #Selected queue to allocate your job
#SBATCH --output=predict_stack.o%j   #Path and name to the file for the STDOUT
#SBATCH --error=predict_stack.e%j    #Path and name to the file for the STDERR

source $HOME/miniconda3/bin/activate myenv

mkdir $HOME/EpilepsySegmentation/predicting/default
mkdir $HOME/EpilepsySegmentation/predicting/default/PCB5KA14drh_test

python3 $HOME/EpilepsySegmentation/UNET/predict_stack.py --predict_dataset $HOME/EpilepsySegmentation/data/for_prediction/PCB5KA14drh_test/ --save_path $HOME/EpilepsySegmentation/predicting/default/PCB5KA14drh_test/ --model_path $HOME/EpilepsySegmentation/training/checkpoints/default/best_model.pt --save_ims 1


