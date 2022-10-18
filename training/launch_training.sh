#!/bin/bash
#SBATCH --job-name=UNET_train_gpu     #Name of your job
#SBATCH --time=6:00:00      #Maximum allocated time
#SBATCH --qos=6hours         #Selected queue to allocate your job
#SBATCH --mem-per-cpu=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=rtx8000  # or titanx
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, etc
#SBATCH --output=UNET_train_gpu.o%j   #Path and name to the file for the STDOUT
#SBATCH --error=UNET_train_gpu.e%j    #Path and name to the file for the STDERR

module load CUDA/11.3.1

cp -R $HOME/EpilepsySegmentation/data/train $TMPDIR/
cp -R $HOME/EpilepsySegmentation/data/val $TMPDIR/

source $HOME/miniconda3/bin/activate myenv

time python3 $HOME/EpilepsySegmentation/UNET/train.py --model_name default --train_dataset $TMPDIR/train/ --val_dataset $TMPDIR/val/ --checkpoint_path $HOME/EpilepsySegmentation/training/checkpoints/ --save_model 1 --save_freq 400 --device cuda --batch_size 30 --epochs 400 --crop 512
