import os

import numpy as np
import torch

from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torch.autograd import Variable

from unet.dataset import Image2D

from skimage import io

import pandas as pd

# deal with arguments
parser = ArgumentParser()
parser.add_argument('--predict_dataset', required=True, type=str)
parser.add_argument('--save_path', required=True, type=str)
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--save_ims', default=1, type=int)
args = parser.parse_args()

if args.save_ims == 1:
    save_ims = True
else:
    save_ims = False

# set up data loader and load the trained model
predict_dataset = Image2D(args.predict_dataset)
if args.device == 'cpu':
    model = torch.load(args.model_path,map_location=torch.device('cpu'))
else:
     model = torch.load(args.model_path)
model.eval()

# directories
export_path = args.save_path
if not os.path.exists(export_path):
    os.makedirs(export_path)

# set up metrics to save
n_classes = 4

# loop over images in validation set, evaluate model, save predicted images
for batch_idx, (X_batch, image_filename) in enumerate(DataLoader(predict_dataset, batch_size=1)):
    #filename
    image_filename_short = os.path.splitext(image_filename[0])[0]

    #load data and make prediction
    X_batch = Variable(X_batch.to(args.device),)
    y_out = model(X_batch).to(args.device)

    # go from prob to segmentation
    preds = y_out.detach().clone()
    preds = torch.argmax(preds, dim=1).squeeze(dim=1)

    #save images
    if save_ims:
        #print('saving images')
        # save as label
        io.imsave(os.path.join(export_path, image_filename_short + '_predicted.png'),
                  torch.argmax(y_out, dim=1).squeeze().detach().numpy().astype(np.dtype('uint8')),
                  check_contrast=False)
