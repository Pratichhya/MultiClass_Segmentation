# importing necessary packages
import os
import rasterio
import numpy as np
from tqdm import tqdm

"""to make a grid of patchsize X patchsize on top of the image and create sample patches"""

class Patch:
    def __init__(self):
        print("It's gonna take a while...grab some food until then")
        # self.DATASET = DATASET
        
    # image patching in case of single images
    def image_patching(self,data):
        self.load_data()
        print("But did you check what did they give us? AVOID SHAPE ERROR👻👻👻👻👻")
        print(f"initial data shape is: {data.shape}")
        nbands, nrows, ncols = data.shape
        imgarray = data
        patchsamples = np.zeros(
            shape =(0, nbands, self.patchsize, self.patchsize), dtype=imgarray.dtype
        )
        for i in range(int(nrows / self.patchsize)):
            for j in range(int(ncols / self.patchsize)):
                tocat = imgarray[
                    :,
                    i * self.patchsize : (i + 1) * self.patchsize,
                    j * self.patchsize : (j + 1) * self.patchsize,
                ]
                tocat = np.expand_dims(tocat, axis=0)
                patchsamples = np.concatenate((patchsamples, tocat), axis=0)
        patchsamples = patchsamples
        return patchsamples
    
    
class Unpatch:
    def __init__(self):
        print("Mosaicing all")
    
    def image_unpatch(self):
        return