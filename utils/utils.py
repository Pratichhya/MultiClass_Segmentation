import torch
import torch.nn.functional as F
import torch.utils.data
import random
import numpy as np
import sys, os


# from torch.utils.data.sampler import BatchSampler
# Bring your packages onto the path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def model_eval(dataloader, model_g, model_f):
    model_g.eval()
    model_f.eval()
    total_samples =0
    correct_prediction = 0
    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            label = label.long().to(device)
            gen_output = model_g(img)
            pred = F.softmax(model_f(gen_output), 1)
            correct_prediction += torch.sum(torch.argmax(pred,1)==label)
            total_samples += pred.size(0)
        accuracy = correct_prediction.cpu().data.numpy()/total_samples
    return accuracy
    
    
def squared_distances(x, y):
    '''
        Compute the squared eculidean matrix 
        Adapted from Geomloss
    '''
    if x.dim() == 2:
        D_xx = (x*x).sum(-1).unsqueeze(1)  # (N,1)
        D_xy = torch.matmul( x, y.permute(1,0) )  # (N,D) @ (D,M) = (N,M)
        D_yy = (y*y).sum(-1).unsqueeze(0)  # (1,M)
    elif x.dim() == 3:  # Batch computation
        D_xx = (x*x).sum(-1).unsqueeze(2)  # (B,N,1)
        D_xy = torch.matmul( x, y.permute(0,2,1) )  # (B,N,D) @ (B,D,M) = (B,N,M)
        D_yy = (y*y).sum(-1).unsqueeze(1)  # (B,1,M)
    else:
        print("x.shape : ", x.shape)
        raise ValueError("Incorrect number of dimensions")

    return D_xx - 2*D_xy + D_yy
