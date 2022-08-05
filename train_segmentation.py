
from seg_model_smp.models_predefined import segmentation_models_pytorch as psmp
import json
import random
import torch
import torch.optim as optim
import wandb
import numpy as np
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
import tifffile as tiff
from torch import nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable

# files
from data_loader.dataset import Dataset
from utils.criterion import DiceLoss
from utils import eval_metrics
# from model.unet import UNet


Dice = DiceLoss()
# reading config file
with open(
    "/share/projects/erasmus/pratichhya_sharma/DAoptim/DAoptim/utils/config.json",
    "r",
) as read_file:
    config = json.load(read_file)


def set_seed(seed):
    """Set all random seeds to a fixed value and take out any randomness from cuda kernels"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


# set network
net = psmp.Unet(encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=3,
                # model output channels (number of classes in your dataset)
                classes=1,
                )
# net = UNet(config["n_channel"], config["n_classes"])
net.cuda()

saving_interval = 10
base_lr = 0.1
running_loss = 0.0
testing_loss = 0.0
training_loss = 0.0
validation_loss = 0.0
mode = config["mode"]
NUM_EPOCHS = 50
lrs = []

# early stopping patience; how long to wait after last time validation loss improved.
patience = 5
the_last_loss = 100
