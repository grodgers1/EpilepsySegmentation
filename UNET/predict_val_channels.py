import os

import numpy as np
import torch

from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchmetrics

from unet.metrics import my_dice, my_F1, my_jaccard, my_crossentropy
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
n_classes = 4
namesList = []
DiceLogitsList = []
F1LogitsList = []
JaccardLogitsList = []
CrossentropyList = []
DiceSingleList = []
F1SingleList = []
JaccardSingleList = []
DiceMultiList = []
F1MultiList = []
JaccardMultiList = []

# loop over images in validation set, evaluate model, evaluate loss, write to file, save predicted images
for batch_idx, (X_batch, y_batch, image_filename) in enumerate(DataLoader(predict_dataset, batch_size=1)):
    #filename
    image_filename_short = os.path.splitext(image_filename[0])[0]
    print(image_filename_short)

    #load data and make prediction
    X_batch = Variable(X_batch.to(args.device),)
    y_batch = Variable(y_batch.to(args.device))
    y_batch = y_batch.long()
    #print(f'y_batch: {y_batch.shape}')
    y_out = model(X_batch).to(args.device)
    #print(f'y_out: {y_out.shape}')

    # channel-wise metrics
    preds = y_out.detach().clone()
    preds = torch.argmax(preds, dim=1).squeeze(dim=1)
    target = y_batch.detach().clone()
    
    # evaluate metrics
    DiceLogits = my_dice(y_out,y_batch)
    F1Logits = my_F1(y_out,y_batch)
    JaccardLogits = my_jaccard(y_out,y_batch)
    Crossentropy = my_crossentropy(y_out,y_batch)
    
    d1Single = torchmetrics.functional.dice_score(preds, target).detach().numpy()
    f1Single = torchmetrics.functional.f1_score(preds,target,num_classes=n_classes, average='micro', mdmc_average='samplewise', ignore_index=0).detach().numpy()
    j1Single = torchmetrics.functional.jaccard_index(preds,target,num_classes=n_classes,ignore_index=0).detach().numpy()
    
    d1Multi = torchmetrics.functional.dice_score(preds, target, bg=True, reduction='none')[0:n_classes].detach().numpy()
    f1Multi = torchmetrics.functional.f1_score(preds,target,num_classes=n_classes, average='none', mdmc_average='samplewise').detach().numpy()
    j1Multi = torchmetrics.functional.jaccard_index(preds,target,num_classes=n_classes,reduction='none')[0:n_classes].detach().numpy()
    
    # update lists
    namesList.append(image_filename_short)
    DiceLogitsList.append(DiceLogits)
    F1LogitsList.append(F1Logits)
    JaccardLogitsList.append(JaccardLogits)
    CrossentropyList.append(Crossentropy)

    DiceSingleList.append(d1Single)
    F1SingleList.append(f1Single)
    JaccardSingleList.append(j1Single)

    DiceMultiList.append(d1Multi)
    F1MultiList.append(f1Multi)
    JaccardMultiList.append(j1Multi)

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

dict = {'filename': namesList, 'Crossentropy': CrossentropyList, 'DiceLogits': DiceLogitsList,
        'F1Logits': F1LogitsList, 'JaccardLogits': JaccardLogitsList, 'DiceSingle': DiceSingleList,
        'F1Single': F1SingleList, 'JaccardSingle': JaccardSingleList}
df = pd.DataFrame(dict)

d1_df = pd.DataFrame(DiceMultiList,columns=['Dice_ch0','Dice_ch1','Dice_ch2','Dice_ch3'])
j1_df = pd.DataFrame(JaccardMultiList,columns=['Jaccard_ch0','Jaccard_ch1','Jaccard_ch2','Jaccard_ch3'])
f1_df = pd.DataFrame(F1MultiList,columns=['F1_ch0','F1_ch1','F1_ch2','F1_ch3'])
df = pd.concat([df,d1_df,j1_df,f1_df],axis=1)

df.to_csv(os.path.join(export_path,'metrics.csv'))
