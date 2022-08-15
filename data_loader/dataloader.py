# importing necessary packages

import os
import rasterio
import numpy as np
from tqdm import tqdm
import tifffile as tiff
import shutil
import json

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader, random_split

# reading config file
with open("/home/jovyan/private/MultiClassSegmentation/utils/config.json", "r",) as read_file:
    config = json.load(read_file)


class Dataset():
    def __init__(self, data_folder):
        # connecting to the folder
        print("Buckle up, here with start the journey🚲")
        self.data_folder = data_folder

    def array_torch(self):
        print("On the way to find data and make our final dataloader")
        self.Xmain = np.load(self.data_folder + "/npy/Xdata_256mul_nonorm.npy")
        self.Ymain = np.load(self.data_folder + "/npy/Ydata_256mul_nonorm.npy")
        print("----------------------Found already existing npy----------------------")
        self.Xmain = self.Xmain[:100, :, :, :]
        self.Ymain = self.Ymain[:100, :, :, :]

        print("shape of Xmain: ", self.Xmain.shape)
        print("shape of Ymain: ", self.Ymain.shape)
#         print(f"x max:{self.Xmain.max()}")
#         print(f"x min:{self.Xmain.min()}")
#         print(f"y max:{self.Ymain.max()}")
#         print(f"y min:{self.Ymain.min()}")

        print("----------------------------------------------------------------------")
        print("Since its multi class segmentation I prefer onehot encoding....")
        print(
            f"..........so before onehot encoding label shape is: {self.Ymain.shape}")

        Ymain_onehot = np.identity(8)[self.Ymain.astype(int)]
        Ymain_onehot = np.swapaxes(Ymain_onehot, 1, -1)
        self.Ymain = np.squeeze(Ymain_onehot, axis=-1)
        print(
            f"..........and after onehot encoding label shape is: {self.Ymain.shape}")

        print("----------------------------------------------------------------------")
        print("Time to split the data to train, test and validation set")

        # set aside 20% of train and test data for evaluation
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Xmain, self.Ymain, test_size=0.2, shuffle=True, random_state=8)
        # Use the same function above for the validation set
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.25, random_state=8)  # 0.25 x 0.8 = 0.2

        # printing final shape
        print(f"shape of Training Images{self.X_train.shape}")
        print(f"shape of Training Labels{self.y_train.shape}")
        print(f"shape of Validation Images{self.X_val.shape}")
        print(f"shape of Validation Labels{self.y_val.shape}")
        print(f"shape of Test Images{self.X_test.shape}")
        print(f"shape of Test Labels{self.y_test.shape}")

        # converting numpy array to pytorch dataset
        self.x_train = torch.Tensor(self.X_train.astype(np.float16))
        self.y_train = torch.Tensor(self.y_train.astype(np.float16))
        self.x_val = torch.Tensor(self.X_val.astype(np.float16))
        self.y_val = torch.Tensor(self.y_val.astype(np.float16))
        self.x_test = torch.Tensor(self.X_test.astype(np.float16))
        self.y_test = torch.Tensor(self.y_test.astype(np.float16))

        self.tensor_train = TensorDataset(self.x_train, self.y_train)
        self.train_data = DataLoader(self.tensor_train, batch_size=32,
                                     pin_memory=True, shuffle=True, worker_init_fn=np.random.seed(42))
        self.tensor_val = TensorDataset(self.x_val, self.y_val)
        self.val_data = DataLoader(self.tensor_val, batch_size=32,
                                   pin_memory=True, shuffle=True, worker_init_fn=np.random.seed(42))
        self.tensor_test = TensorDataset(self.x_test, self.y_test)
        self.test_data = DataLoader(self.tensor_test, batch_size=32,
                                    pin_memory=True, shuffle=True, worker_init_fn=np.random.seed(42))

        print("Finally atleast test dataloader section works 😌")


# if __name__ == "__main__":
#     DATASET = Dataset("/home/jovyan/private/MultiClassSegmentation/dataloader")
#     DATASET.array_torch()
