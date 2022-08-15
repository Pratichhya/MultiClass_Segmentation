import segmentation_models_pytorch as smp
import json
import random
import wandb
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tifffile as tiff

import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable

# files
from dataloader.dataloader import Dataset
from utils.criterion import DiceLoss
from utils import eval_metrics


DC = DiceLoss()

NUM_EPOCHS = 5
base_lr = 0.01


def set_seed(seed):
    """Set all random seeds to a fixed value and take out any randomness from cuda kernels"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


net = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=8,
    # model output channels (number of classes in your dataset)
    activation="softmax"
)
net.cuda()


def train_epoch(optimizer, dataloader):
    len_train = len(dataloader)
    f1_source, acc, IoU, K = 0.0, 0.0, 0.0, 0.0
    total_loss = 0
    net.train()
    iter_ = 0

    for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len_train):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        # zero optimizer
        optimizer.zero_grad()
        output = net(data)

        loss = DC(output, target)
        loss.backward()
        optimizer.step()

        # evaluation
        f1_source_step, acc_step, IoU_step, K_step = eval_metrics.f1_score(
            target, output)
        f1_source += f1_source_step
        acc += acc_step
        IoU += IoU_step
        K += K_step
        total_loss += loss
        # wandb.log({'train_Loss': loss,'train_F1': f1_source_step,'train_acc':acc_step,'train_IoU':IoU_step})
    return (total_loss/len_train), [f1_source/len_train, acc/len_train, IoU/len_train, K/len_train]


def eval_epoch(epochs, dataloader):
    len_train = len(dataloader)
    f1_source, acc, IoU, K = 0.0, 0.0, 0.0, 0.0
    val_loss = 0
    net.eval()
    iter_ = 0
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len_train):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            output = net(data)
            loss = DC(output, target)

            # evaluation
            f1_source_step, acc_step, IoU_step, K_step = eval_metrics.f1_score(
                target, output)
            f1_source += f1_source_step
            acc += acc_step
            IoU += IoU_step
            K += K_step
            total_loss += loss

            if epochs % 2 == 0:
                pred = np.rint(output.data.cpu().numpy()[0])
                gt = target.data.cpu().numpy()[0]
                print(f"shape of pred: {pred.shape}")
                print(f"shape of gt: {gt.shape}")
                # tiff.imwrite(
                #    os.path.join(config["eval_output"], f"rgb_val{i+1}" + ".tif"),
                #    rgb,
                # )
                # tiff.imwrite(
                #    os.path.join(config["eval_output"], f"pred_val{i+1}" + ".tif"),
                #    pred,
                # )
                images_pred = wandb.Image(
                    pred, caption="Top: Output, Bottom: Input")
                # # images_rgb = wandb.Image(rgb, caption="Top: Output, Bottom: Input")
                images_gt = wandb.Image(
                    gt, caption="Top: Output, Bottom: Input")
                wandb.log({"Ground truth": images_gt,
                           "Prediction": images_pred})
            # wandb.log({'Val_Loss': loss,'Val_F1': f1_source_step,'Val_acc':acc_step,'Val_IoU':IoU_step})
    return (total_loss/len_train), [f1_source/len_train, acc/len_train, IoU/len_train, K/len_train]


def main(net):
    parameter_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"The model has {parameter_num:,} trainable parameters")

    # set optimizer
    optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0005)

    # define the scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [1, 10, 20], gamma=0.1)

    # calling the dataloader
    d_loaders = Dataset(
        "/home/jovyan/private/MultiClassSegmentation/dataloader")
    d_loaders.array_torch()
    train_data = d_loaders.train_data
    val_data = d_loaders.val_data
    test_data = d_loaders.test_data

    for e in range(1, NUM_EPOCHS + 1):
        print("----------------------Traning phase-----------------------------")
        train_loss, acc_mat = train_epoch(optimizer, train_data)
        print(f"Training loss in average for epoch {str(e)} is {train_loss}")
        print(f"Training F1 in average for epoch {str(e)} is {acc_mat[0]}")
        print(
            f"Training Accuracy in average for epoch {str(e)} is {acc_mat[1]}")
        print(f"Training IOU in average for epoch {str(e)} is {acc_mat[2]}")
        # print(f"Training K in average for epoch {str(e)} is {acc_mat[3]}")
        wandb.log({'Train Loss': train_loss,
                   'Train_F1': acc_mat[0], 'Train_acc': acc_mat[1], 'Train_IoU': acc_mat[2], 'Train_Kappa': acc_mat[3]}, step=e)
        # (total/batch)*epoch=iteration
        del train_loss, acc_mat

        print("----------------------Evaluation phase-----------------------------")
        f1_collect = []
        valid_loss, acc_mat = eval_epoch(e, val_data)
        print(f"Evaluation loss in average for epoch {str(e)} is {valid_loss}")
        print(f"Evaluation F1 in average for epoch {str(e)} is {acc_mat[0]}")
        f1_collect.append(acc_mat[0])
        print(
            f"Evaluation Accuracy in average for epoch {str(e)} is {acc_mat[1]}")
        print(f"Evaluation IOU in average for epoch {str(e)} is {acc_mat[2]}")
        # print(f"Evaluation K in average for epoch {str(e)} is {acc_mat[3]}")
        wandb.log({'Val_Loss': valid_loss,
                   'Val_F1': acc_mat[0], 'Val_acc': acc_mat[1], 'Val_IoU': acc_mat[2], 'Val_Kappa': acc_mat[3]}, step=e)
        del valid_loss, acc_mat

    max_f1 = max(f1_collect)
    # torch.save(net.state_dict(),"saved_model/multi_seg_v0")
    print("finished")
    return max_f1


if __name__ == "__main__":
    wandb.login()
    wandb.init(project="multi_class")
    for i in range(2):
        f1 = main(net)
        with open("./f1.txt", "a") as a_file:
            a_file.write("\n")
            a_file.write(f"Unet Trial{i+1} max evaluation f1: {f1}")
