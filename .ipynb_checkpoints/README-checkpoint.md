# EpilepsySegmentation
U-Net segmentation for the dentate gyrus of healthy and epileptic mouse brain hemispheres from X-ray microtomography-based virtual histology data.

The U-Net is provided, along with scripts for splitting data, training the network, analyzing runs, and predicting stacks of images. In this document, I will provide a brief discussion of the datasets, usage, and performance. This document will also provide some specific advice for usage on [sciCORE](http://scicore.unibas.ch/), the scientific computing center at University of Basel.

The PyTorch implementation is based on code from [cosmic-cortex](https://github.com/cosmic-cortex/pytorch-UNet).

# Usage
## Conda environment
You can find a list of packages/sources in my Conda environment to create your own using [this list](./CondaEnvironment.txt) or [this yml file](./environment.yml).

Make sure that your paths are correct in your activate commends within the slurm scripts, for me it looked like this:
`source $HOME/miniconda3/bin/activate myenv`

### Troubleshooting issues with no cuda / gpu
Check your environment and make sure no pytorch installations with cpu only:
`conda list -n myenv`

If there is, then uninstall it with `conda remove -n myenv <package_name>` and then re-install a pytorch version with cuda activated

## Datasets
Data used for segmentation will be made available upon publication. In the meantime, stacks of manually labelled data can be found in `ManuallyLabelled` with sub-directories `images` and `masks` for grayscale images and segmentations, respectively. 

## Data splits
I divided the data into train/validate/test using the script `DataSplits.ipynb`. Note that the network takes a directory for training/validation with sub-directories `images` and `masks` (see `UNET/unet/dataset.py`).
## Train network
### Train on gpu
The relevant script is `UNET/train.py`. You can train the network on gpu with the script `training/launch_training.sh`. Note that the paths may have to be adjusted in that script to match your directory structure.

### Ablation tests
I provide a slurm script for launching a series of trainings on the gpu for an ablation study, see `training/example_ablation.sh`. Note that the paths may have to be adjusted in that script to match your directory structure.

## Analyzing training runs
The notebook `AnalyzeTrainingRuns.ipynb`

## Predicting stacks
The relevant script is `UNET/predict_stack.py`. You can find a template slurm script to launch this (on cpu, not using temporary storage): `predicting/predict_stack.sh`.

# Results
## Ablation study

## Comparison with manual segmentation
### Re-labelled slices

### Othogonal slices

# sciCORE usage tips
## Launching a Jupyter notebook
I used Jupyter notebooks for making data splits and analyzing training runs.

Use the included slurm script for launching a Jupyter notebook, adjust your conda environment name if needed:
`sbatch launch_jupyter.sh`

Check that it is launched:
`squeue -u <username>`

Once running, follow commands in the related `.oe` file, e.g. `more <filename.oe>`. Remember to run the ssh command from your local machine, not from within sciCORE. Try both links in case one isn't working.

## Using Fiji or ImageJ
Can be very useful for reviewing predictions.

On sciCORE, the commands are as follows:
`ml Fiji`
`ImageJ-linux64 &`

### Making a color overlay
Load your grayscale image and segmentations.
Image -> Color -> Merge Channels...

### Synchronize windows:
Analyze -> Tools -> Synchronize Windows