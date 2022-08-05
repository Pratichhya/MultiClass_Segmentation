import os
import rasterio
import numpy as np
from tqdm import tqdm
import tifffile as tiff


class Shape:
    def __init__(self, DATASET):
        print("Good !!! Be sure with the shape before you start i.e:")
        self.DATASET = DATASET
        
        # cross checking image sizes
    def image_size(self):
        self.DATASET.load_data()
        print("shape of Xtrain: ", self.DATASET.Xtrain_main.shape)
        print("shape of Ytrain: ", self.DATASET.Ytrain_main.shape)
        print("shape of Xtest: ", self.DATASET.Xtest_main.shape)
        print("shape of Ytest: ", self.DATASET.Ytest_main.shape)



