import os

import numpy as np
import torch

import argparse
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torch.autograd import Variable

from unet import utils
from unet.metrics import my_dice, my_dice_labels, my_F1, my_F1_labels, my_jaccard, my_jaccard_labels, my_crossentropy
from unet.dataset import ImageAndMask2D

from skimage import io

import pandas as pd

# deal with arguments
parser = ArgumentParser()
parser.add_argument('--validation_dataset', required=True, type=str)
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
predict_dataset = ImageAndMask2D(args.validation_dataset)
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
namesList = []
DiceList = []
DiceLabelList = []
F1List = []
F1LabelList = []
JaccardList = []
JaccardLabelList = []
CrossentropyList = []
# loop over images in validation set, evaluate model, evaluate loss, write to file, save predicted images
for batch_idx, (X_batch, y_batch, image_filename) in enumerate(DataLoader(predict_dataset, batch_size=1)):
    #filename
    image_filename_short = os.path.splitext(image_filename[0])[0]
    #print(image_filename_short)

    #load data and make prediction
    X_batch = Variable(X_batch.to(args.device),)
    y_batch = Variable(y_batch.to(args.device))
    y_batch = y_batch.long()
    #print(f'y_batch: {y_batch.shape}')
    y_out = model(X_batch).to(args.device)
    #print(f'y_out: {y_out.shape}')

    # evaluate metrics
    Dice = my_dice(y_out,y_batch)
    DiceLabel = my_dice_labels(y_out,y_batch)
    F1 = my_F1(y_out,y_batch)
    F1Label = my_F1_labels(y_out,y_batch)
    Jaccard = my_jaccard(y_out,y_batch)
    JaccardLabel = my_jaccard_labels(y_out,y_batch)
    Crossentropy = my_crossentropy(y_out,y_batch)

    # update lists
    namesList.append(image_filename_short)
    DiceList.append(Dice)
    DiceLabelList.append(DiceLabel)
    F1List.append(F1)
    F1LabelList.append(F1Label)
    JaccardList.append(Jaccard)
    JaccardLabelList.append(JaccardLabel)
    CrossentropyList.append(Crossentropy)

    #save images
    if save_ims:
        #print('saving images')
        # save as label
        io.imsave(os.path.join(export_path, image_filename_short + '_predicted.png'),
                  torch.argmax(y_out, dim=1).squeeze().detach().numpy().astype(np.dtype('uint8')),
                  check_contrast=False)
        # save logits output channels
        for c in range(y_out.shape[1]):
            this_channel = 255*(y_out[0, c, :, :].squeeze().detach().numpy())
            this_channel = this_channel.astype(np.dtype('uint8'))
            io.imsave(os.path.join(export_path, image_filename_short + f'_ch{c}.png'), this_channel,
                      check_contrast=False)

dict = {'filename': namesList, 'Dice': DiceList, 'DiceLabel': DiceLabelList, 'F1': F1List, 'F1Label': F1LabelList,
        'Jaccard': JaccardList, 'JaccardLabel': JaccardLabelList, 'Crossentropy': CrossentropyList}
df = pd.DataFrame(dict)
df.to_csv(os.path.join(export_path,'metrics.csv'))
