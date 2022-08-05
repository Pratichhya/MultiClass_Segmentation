import os
import rasterio
import numpy as np

# from .preprocess import PreProcess

class OneHotEncoding:
    def __init__(self):
        print("It's gonna take a while...grab some food until then")
        # self.DATASET = DATASET

    #applying one-hot encoding in labels
    def binary_hot_encoding(self):
        self.create_directory()
        print("one-hot 🔥 encoding")
        if self.for_what == "training_source":
            ones = np.ones(self.Ys_train.shape)
            self.ys_train_inverted = ones-self.Ys_train # Because 0-1 is 1 and 1-0 is zero
            self.Ys_train = np.concatenate((self.ys_train_inverted,self.Ys_train),axis=1)
            print(f" Shape of Ys_train is:{self.Ys_train.shape}")
        # elif self.for_what == "training_target":
        #     ones = np.ones(self.Yt_train.shape)
        #     self.yt_train_inverted = ones-self.Yt_train # Because 0-1 is 1 and 1-0 is zero
        #     self.Yt_train = np.concatenate((self.yt_train_inverted,self.Yt_train),axis=1)
        #     print(f" Shape of Yt_train is:{self.Yt_train.shape}")
        # else:
        #     ones = np.ones(self.ytest_patches.shape)
        #     self.ytest_inverted = ones-self.ytest_patches # Because 0-1 is 1 and 1-0 is zero
        #     self.Ytest = np.concatenate((self.ytest_inverted,self.ytest_patches),axis=1)
        #     print(f" Shape of Ytest is:{self.Ytest.shape}")
