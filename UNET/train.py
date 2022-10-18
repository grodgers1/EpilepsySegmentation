import os

import torch.optim as optim
from unet.utils import summary, summary_string

from functools import partial
from argparse import ArgumentParser

from unet.unet import UNet2D
from unet.model import Model
from unet.utils import MetricList
from unet.metrics import jaccard_index, f1_score, my_dice, my_F1, my_jaccard, my_crossentropy
from unet.metrics import my_dice_labels, my_F1_labels, my_jaccard_labels
from unet.metrics import LogNLLLoss, DiceLoss, JaccardLoss, CrossEntropyLoss
from unet.dataset import JointTransform2D, ImageToImage2D, Image2D

parser = ArgumentParser()
# input/output of net
parser.add_argument('--in_channels', default=1, type=int)
parser.add_argument('--out_channels', default=4, type=int)
# train and validation sets
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', type=str)
# for saving model
parser.add_argument('--checkpoint_path', required=True, type=str)
parser.add_argument('--save_model', default=0, type=int)
parser.add_argument('--save_freq', default=0, type=int)
parser.add_argument('--model_name', type=str, default='model')
# cpu or gpu
parser.add_argument('--device', default='cpu', type=str)
# network structure
parser.add_argument('--up_mode', default='trans', type=str) # 'trans' or 'bilinear'
parser.add_argument('--depth', default=5, type=int)
parser.add_argument('--width', default=32, type=int)
parser.add_argument('--loss', default='default', type=str) # 'default', 'dice', 'jaccard', 'crossentropy'
# training parameters
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--dropout', default=0.0, type=float)
# image transformations/augmentations
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--p_hflip', type=float, default=0.0)
parser.add_argument('--p_vflip', type=float, default=0.0)
parser.add_argument('--p_random_affine', type=float, default=0.0)
parser.add_argument('--random_affine_params', nargs='*', type=float, default=None)
parser.add_argument('--p_noise', type=float, default=0.0)
parser.add_argument('--noise_sigma', type=float, default=0.0)

args = parser.parse_args()

# Data augmentation:
aug_list = [(args.p_hflip != 0), (args.p_vflip != 0), (args.p_noise != 0), (args.p_random_affine != 0),
            (args.crop is not None)]
if any(aug_list):
    print('Summary of data transformations/augmentations:')

if args.crop is not None:
    crop = (args.crop, args.crop)
    print(f'Applying cropping to size: {crop}')
else:
    crop = None

p_vflip = args.p_vflip
if not (p_vflip == 0):
    print(f'Applying random vertical flip with probability {p_vflip}')

p_hflip = args.p_hflip
if not (p_hflip == 0):
    print(f'Applying random horizontal flip with probability {p_hflip}')

p_random_affine = args.p_random_affine
if args.random_affine_params is not None:
    random_affine_params = args.random_affine_params
else:
    # setting to default
    # rot1, rot2, trans1, trans2, scale1, scale2, shear1, shear2
    random_affine_params = (-20, 20, 0.1, 0.1, 0.9, 1.1, -10, 10)
if not (p_random_affine == 0):
    print(f'Applying the following random affine transform with probability {p_random_affine}')
    print(f'Rotation: {random_affine_params[0:2]}')
    print(f'Translate: {random_affine_params[2:4]}')
    print(f'Scale: {random_affine_params[4:6]}')
    print(f'Shear: {random_affine_params[6:8]}')

p_noise = args.p_noise
noise_sigma = args.noise_sigma
if not (p_noise == 0):
    print(f'Adding 0-mean gaussian noise with sigma {noise_sigma} with probability {p_noise}')

# Set up datasets:
tf_train = JointTransform2D(crop=crop, p_hflip=p_hflip, p_vflip=p_vflip, p_random_affine=p_random_affine,
                            random_affine_params=random_affine_params, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_hflip=0, p_vflip=0, p_random_affine=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)

# Set up network:
conv_depths = [int(args.width*(2**k)) for k in range(args.depth)]
unet = UNet2D(args.in_channels, args.out_channels, conv_depths, args.dropout, args.up_mode)
#loss = LogNLLLoss()
if (args.loss == 'dice'):
    loss = DiceLoss()
    print(f'Using loss: {args.loss}')
elif (args.loss == 'jaccard'):
    loss = JaccardLoss()
    print(f'Using loss: {args.loss}')
elif (args.loss == 'crossentropy'):
    loss = CrossEntropyLoss()
    print(f'Using loss: {args.loss}')
else:
    loss = LogNLLLoss()
    print(f'Using default loss: LogNLLLoss()')

optimizer = optim.Adam(unet.parameters(), lr=args.learning_rate)

results_folder = os.path.join(args.checkpoint_path, args.model_name)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

model = Model(unet, loss, optimizer, results_folder, device=args.device)

# make a dictionary of settings, write to file
results, params_info = summary_string(unet,(1,512,512),batch_size=args.batch_size,device=args.device)
with open(os.path.join(results_folder,'network_summary.txt'),'w') as f:
    print(results,file=f)

loggedParams = {
    "in_channels": args.in_channels,
    "out_channels": args.out_channels,
    "train_dataset": args.train_dataset,
    "val_dataset": args.val_dataset,
    "checkpoint_path": args.checkpoint_path,
    "save_model": args.save_model,
    "save_freq": args.save_freq,
    "model_name": args.model_name,
    "device": args.device,
    "loss": args.loss,
    "up_mode": args.up_mode,
    "depth": args.depth,
    "width": args.width,
    "total_learned_parameters": params_info[0].detach().numpy(),
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "dropout": args.dropout,
    "crop": crop,
    "p_vflip": p_vflip,
    "p_hflip": p_hflip,
    "p_random_affine": p_random_affine,
    "random_affine_params": random_affine_params,
    "noise_sigma": noise_sigma,
    "p_noise": p_noise
}
f = open(os.path.join(results_folder,'network_parameters.txt'),'w')
for key in loggedParams.keys():
    f.write(str(key) + ":\t" + str(loggedParams[key]))
    f.write("\n")
f.close()

# metrics to log
# metric_list = MetricList({'jaccard': partial(jaccard_index),
#                           'f1': partial(f1_score)})
metric_list = MetricList({'jaccard_old': partial(jaccard_index),
                          'f1_old': partial(f1_score),
                          'Dice': partial(my_dice,as_loss=False),
                          'DiceLabel': partial(my_dice_labels,as_loss=False),
                          'F1': partial(my_F1,as_loss=False),
                          'F1Label': partial(my_F1_labels,as_loss=False),
                          'Jaccard': partial(my_jaccard, num_classes=4, as_loss=False),
                          'JaccardLabel': partial(my_jaccard_labels, num_classes=4, as_loss=False),
                          'Crossentropy': partial(my_crossentropy, weights=None, as_loss=False)
                          })

# Run training:
print('Starting training...')
model.fit_dataset(train_dataset, n_epochs=args.epochs, n_batch=args.batch_size,
                  shuffle=True, val_dataset=val_dataset, save_freq=args.save_freq,
                  save_model=args.save_model, predict_dataset=predict_dataset,
                  metric_list=metric_list, verbose=True)