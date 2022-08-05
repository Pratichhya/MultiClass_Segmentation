# importing necessary packages
import os
import rasterio
import numpy as np

class Augumentation:
    def __init__(self):
        print("Here we start turning around to give you augmented dataset ")
        # self.DATASET = DATASET

    # augumentation of data
    def augumentation(self,patches):
        print("Here we start turning around to give you augmented dataset👩‍💻")
        # image flipping
        self.data_fliped = patches[:, :, ::-1, :]
        # image mirroring
        self.data_mirrored =patches[:, :, :, ::-1]
        # appending all together
        self.data_fliped = np.append(patches, self.data_fliped, axis=0)
        all_images = np.append(self.data_fliped, self.data_mirrored, axis=0)
        return all_images