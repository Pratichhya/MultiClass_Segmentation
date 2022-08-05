# -*- coding: utf-8 -*-
# importing necessary packages
import os
import rasterio
import numpy as np
from tqdm import tqdm
import tifffile as tiff
import shutil
import json

import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader,random_split

#files
from .test_size import Shape
from .patching import Patch
from .augmentation import Augumentation
from .one_hot import OneHotEncoding
from .preprocess import PreProcess
# from test_size import Shape
# from patching import Patch
# from augmentation import Augumentation
# from one_hot import OneHotEncoding

# reading config file
with open("/share/projects/erasmus/pratichhya_sharma/DAoptim/DAoptim/utils/config.json","r",) as read_file:
    config = json.load(read_file)

class Dataset():
    def __init__(self, data_folder, patchsize, for_what):
        # connecting to the folder
        print("Buckle up, here with start the journey🚲")
        self.data_folder = data_folder
        self.train_folder = self.data_folder + "train_data"
        self.test_folder = self.data_folder + "test_data"
        self.patchsize = patchsize
        self.for_what = for_what


    # fetching training and testing folder
    def get_data(self):
        self.xs_train_img_files = []
        self.ys_train_label_files = []
        self.xt_train_img_files = []
        self.yt_train_label_files = []
        self.test_img_files = []
        self.test_label_files = []
        # collecting training images files
        if self.for_what == "training_source":
            for image in os.listdir(self.train_folder):
                if image == "source":
                    xs_train_img_folder = os.path.join(self.train_folder, image)
                    for img in os.listdir(xs_train_img_folder):
                        xs_train_img_path = os.path.join(xs_train_img_folder, img)
                        self.xs_train_img_files.append(xs_train_img_path)
                        # print(f"training images:{self.train_img_files}")
                elif image == "source_labels":
                    ys_train_labels_folder = os.path.join(self.train_folder, image)
                    for lab in os.listdir(ys_train_labels_folder):
                        ys_train_label_path = os.path.join(ys_train_labels_folder, lab)
                        self.ys_train_label_files.append(ys_train_label_path)
                        # print(f"training labels:{self.train_label_files}")
        if self.for_what == "training_target":
            for image in os.listdir(self.train_folder):
                if image == "target":
                    xt_train_img_folder = os.path.join(self.train_folder, image)
                    for img in os.listdir(xt_train_img_folder):
                        xt_train_img_path = os.path.join(xt_train_img_folder, img)
                        self.xt_train_img_files.append(xt_train_img_path)
                        # print(f"training images:{self.train_img_files}")
                elif image == "target_labels":
                    yt_train_labels_folder = os.path.join(self.train_folder, image)
                    for lab in os.listdir(yt_train_labels_folder):
                        yt_train_label_path = os.path.join(yt_train_labels_folder, lab)
                        self.yt_train_label_files.append(yt_train_label_path)
                        # print(f"training labels:{self.train_label_files}")        
        elif self.for_what == "testing":
            for image in os.listdir(self.test_folder):
                if image == "images":
                    test_img_folder = os.path.join(self.test_folder, image)
                    for img in os.listdir(test_img_folder):
                        test_img_path = os.path.join(test_img_folder, img)
                        self.test_img_files.append(test_img_path)
                        # print(f"testing images:{self.test_img_files}")
                elif image == "labels":
                    test_labels_folder = os.path.join(self.test_folder, image)
                    for lab in os.listdir(test_labels_folder):
                        test_label_path = os.path.join(test_labels_folder, lab)
                        self.test_label_files.append(test_label_path)
                        # print(f"testing labels:{self.test_label_files}")

    # reading images
    def read_images(self, image_list):
        # append all the images as numpy array after reading
        for val in image_list:
            read_img = rasterio.open(val).read()
            return read_img

    # reading labels
    def read_labels(self, label_list):
        # append all the labels as numpy after reading
        for val in label_list:
            read_labels = rasterio.open(val).read()
            return read_labels

    # loading training and testing dataset
    def load_data(self):
        print("Ok just wait here, I will go search for data🔎")
        print("wow, you go girl, you found all the dataset required for the trip🎒📷🧳⚡🍊🍹")
        self.get_data()
        if self.for_what == "training_source":
            self.Xs_train_main = self.read_images(self.xs_train_img_files)
            self.Ys_train_main = self.read_labels(self.ys_train_label_files)
        elif self.for_what == "training_target":
            self.Xt_train_main = self.read_images(self.xt_train_img_files)
            #self.Yt_train_main = self.read_labels(self.yt_train_label_files)
        elif self.for_what == "testing":
            self.Xtest_main = self.read_images(self.test_img_files)
            self.Ytest_main = self.read_labels(self.test_label_files)

    def save_images(self,path,array,name):
        for i in range(array.shape[0]):
            #Create the output files with each unique file name
            tiff.imwrite(os.path.join(path, f"{name}_{i+1}"+".tif"), array[i,:,:,:])
        print("Congratulations!!! 🎉🎉🎉  for saving all your augmented patches")

    # save the created patches in a different folder
    def create_directory(self):
        # self.save_images()
        if self.for_what == "training_source":
            # training images creation
            if os.path.exists(self.train_folder + "/source_img_patches/"):
                print("I will read source image patches from existing directory📘📖")
                self.xs_train_patches = []
                for i in os.listdir(self.train_folder + "/source_img_patches/"):
                    im = rasterio.open(self.train_folder + "/source_img_patches/"+i).read()
                    self.xs_train_patches.append(im)
                self.Xs_train = np.asarray(self.xs_train_patches)
                print("shape of previously patched images are:", self.Xs_train.shape)
                # shutil.rmtree(self.train_folder + "/source_img_patches")  # Removes all the subdirectories!
                # os.makedirs(self.train_folder + "/source_img_patches")
            else:
                self.load_data()
                os.makedirs(self.train_folder + "/source_img_patches/")
                print("working on images...grab some food until then🍜")
                self.xs_train_patches = Patch.image_patching(self,self.Xs_train_main)
                #normalize 
                self.Xs_train_norm = []
                for i in range(len(self.xs_train_patches)):
                    self.norm_im = PreProcess.bytes8caling(self,self.xs_train_patches[i,:,:,:], chanell_order = "first")
                    self.Xs_train_norm.append(self.norm_im)
                Xs_train_ax = np.asarray(self.Xs_train_norm)
                self.Xs_train =  np.moveaxis(Xs_train_ax, 3, 1)
                print(f"shape of normalized source images are {self.Xs_train.shape}")
                # self.Xs_train = Augumentation.augumentation(self,self.xs_train_patches)
                xs_file_path = self.train_folder + "/source_img_patches/"
                self.save_images(xs_file_path,self.Xs_train,"xs_train")
                print("!!!!!!!!!!!! saving source images as npy after patched and normalized !!!!!!!!!!!!!!")
                np.save(config["npy_path"]+'Xs_train.npy', self.Xs_train)

            # training labels creation
            if os.path.exists(self.train_folder + "/source_label_patches/"):
                print("I will read source labels from existing directory📗📖")
                self.ys_train_patches = []
                for i in os.listdir(self.train_folder + "/source_label_patches/"):
                    im = rasterio.open(self.train_folder + "/source_label_patches/"+i).read()
                    self.ys_train_patches.append(im)
                self.Ys_train = np.asarray(self.ys_train_patches)
                print("read; shape is:", self.Ys_train.shape)
                # shutil.rmtree(self.train_folder + "/source_label_patches")  # Removes all the subdirectories!
                # os.makedirs(self.train_folder + "/source_label_patches")
            else:
                self.load_data()
                os.makedirs(self.train_folder + "/source_label_patches/")
                print("It's still gonna take a while...have some patience🧘‍♀️")
                self.ys_train_patches = Patch.image_patching(self,self.Ys_train_main)
                self.Ys_train = self.ys_train_patches
                # self.Ys_train =Augumentation.augumentation(self,self.ys_train_patches)
                ys_file_path = self.train_folder + "/source_label_patches/"
                self.save_images(ys_file_path,self.Ys_train,"ys_train")
                np.save(config["npy_path"]+'Ys_train.npy', self.Ys_train)
            
        elif self.for_what == "training_target":
            # training images creation
            if os.path.exists(self.train_folder + "/target_img_patches/"):
                print("I will read target image patches from existing directory📔📕")
                self.xt_train_patches = []
                for i in os.listdir(self.train_folder + "/target_img_patches/"):
                    im = rasterio.open(self.train_folder + "/target_img_patches/"+i).read()
                    self.xt_train_patches.append(im)
                self.Xt_train_main = np.asarray(self.xt_train_patches)
                print("Available patched target is:", self.Xt_train_main.shape)
                print("---------------it was only for target that patched images are loaded then normalized----------")
                #normalize 
                Xt_train_norm = []
                for i in range(len(self.Xt_train_main)):
                    self.norm_imt = PreProcess.bytes8caling(self, self.Xt_train_main[i,:,:,:], chanell_order = "first")
                    Xt_train_norm.append(self.norm_imt)
                Xt_train_ax = np.asarray(Xt_train_norm)
                self.Xt_train =  np.moveaxis(Xt_train_ax, 3, 1)
                print(f"shape of normalized target images are {self.Xt_train.shape}")
                self.Xt_train =  PreProcess.selected_patches(self)
                np.save(config["npy_path"]+'Xt_train.npy', self.Xt_train)
                # shutil.rmtree(self.train_folder + "/target_img_patches")  # Removes all the subdirectories!
                # os.makedirs(self.train_folder + "/target_img_patches")
            else:
                print("I dont think you have entire image for target set😰😭😭😭😭😭")
                # self.load_data()
                # os.makedirs(self.train_folder + "/target_img_patches/")
                # print("working on images...grab some food until then🥐🍕")
                # self.xt_train_patches = Patch.image_patching(self,self.Xt_train_main)
                # # self.Xt_train = Augumentation.augumentation(self,self.xt_train_patches)
                # xt_file_path = self.train_folder + "/target_img_patches/"
                # self.save_images(xt_file_path,self.Xt_train,"xt_train")
                # np.save(config["npy_path"]+'Xt_train.npy', self.Xt_train)
                
        elif self.for_what == "testing":
            # testing image creation
            if os.path.exists(self.test_folder + "/testing_images/"):
                print("I will read testing dataset from existing directory📙📖")
                self.xtest_patches = []
                for i in os.listdir(self.test_folder + "/testing_images/"):
                    im = rasterio.open(self.test_folder + "/testing_images/"+i).read()
                    self.xtest_patches.append(im)
                self.xtest_patches = np.asarray(self.xtest_patches)
                print("read; shape is:", self.xtest_patches.shape)
                # shutil.rmtree(self.test_folder + "/testing_images")  # Removes all the subdirectories!
                # os.makedirs(self.test_folder + "/testing_images")
            else:
                # self.load_data()
                os.makedirs(self.test_folder + "/testing_images/")
                print("working on images...grab some food until then🍩🥞")
                self.xtest_patches = []
                self.xt_train_patches = []
                for i in os.listdir(self.train_folder + "/target_img_patches/"):
                    im = rasterio.open(self.train_folder + "/target_img_patches/"+i).read()
                    self.xt_train_patches.append(im)
                self.Xt_train_main = np.asarray(self.xt_train_patches)
                print("Available patched target is:", self.Xt_train_main.shape)
                print("---------------it was only for target that patched images are loaded then normalized----------")
                #normalize 
                self.Xt_train_norm = []
                for i in range(len(self.Xt_train_main)):
                    self.norm_imt = PreProcess.bytes8caling(self, self.Xt_train_main[i,:,:,:], chanell_order = "first")
                    self.Xt_train_norm.append(self.norm_imt)
                Xt_train_ax = np.asarray(self.Xt_train_norm)
                self.Xt_train =  np.moveaxis(Xt_train_ax, 3, 1)
                print(f"shape of normalized target images are {self.Xt_train.shape}")
                #filtering out the leftouts
                self.Xtest =  PreProcess.leftout(self,self.Xt_train)
                x_file_path = self.test_folder +"/testing_images"
                self.save_images(x_file_path,self.Xtest,"xtest")
                np.save(config["npy_path"]+'Xtest.npy', self.Xtest)

                
            # # testing labels creation
            # if os.path.exists(self.test_folder + "/testing_labels/"):
            #     print("I will read test labels from existing directory📗📖")
            #     # shutil.rmtree(self.test_folder + "/testing_labels")  # Removes all the subdirectories!
            #     # os.makedirs(self.test_folder + "/testing_labels")
            #     self.ytest_patches = []
            #     for i in os.listdir(self.test_folder+ "/testing_labels/"):
            #         im = rasterio.open(self.test_folder+ "/testing_labels/"+i).read()
            #         self.ytest_patches.append(im)
            #     self.ytest_patches = np.asarray(self.ytest_patches)
            #     print("read; shape is:", self.ytest_patches.shape)
            # else:
            #     self.load_data()
            #     os.makedirs(self.test_folder + "/testing_labels/")
            #     print("It's still gonna take a while...have some patience🖖")
            #     self.ytest_patches =Patch.image_patching(self,self.Ytest_main)
            #     # self.Ytest = Augumentation.augumentation(self,self.ytest_patches)
            #     y_file_path = self.test_folder +"/testing_labels/"
            #     self.save_images(y_file_path,self.ytest_patches,"ytest")
        else:
            print("First, be sure what you want")

            
    def array_torch(self):
        if os.path.exists("data_loader/npy/Xt_chicago.npy") and os.path.exists("data_loader/npy/Yt_chicago.npy") and os.path.exists("data_loader/npy/Xs_wien.npy"):
            self.Xt_train  = (np.load(self.data_folder + "/npy/Xtest_chicago.npy"))/255
            self.Xt_train =self.Xt_train[:,:3,:,:]
            # self.Xt_train  = ((np.load(self.data_folder + "npy/Xt_train.npy"))[:,:3,:,:])/255
            self.Yt_train   = np.load(self.data_folder + "/npy/Ytest_chicago.npy")
            self.Yt_train =self.Yt_train[:,:3,:,:]
            # self.Yt_train  = np.load(self.data_folder + "npy/Yt_train.npy")
            self.Xs_train = (np.load(self.data_folder + "/npy/Xtest_wien.npy"))/255
            self.Xs_train =self.Xs_train[:,:3,:,:]
            # self.Xs_train  = ((np.load(self.data_folder + "npy/Xs_train.npy"))[:,:3,:,:])/255
            self.Ys_train = np.load(self.data_folder + "/npy/Ytest_wien.npy")
            self.Ys_train =self.Ys_train[:,:3,:,:]
            # self.Ys_train  = np.load(self.data_folder + "npy/Ys_train.npy")
            
            print("----------------------ready to use dataset--------------")
            print("Found already existing npy")
            print("shape of Xs_train: ", self.Xs_train.shape)
            print("shape of Ys_train: ", self.Ys_train.shape)
            print("shape of Xt_train: ", self.Xt_train.shape)
            print("shape of Yt_train: ", self.Yt_train.shape)
            # print(f"xs max:{self.Xs_train.max()}")
            # print(f"xs min:{self.Xs_train.min()}")
            # print(f"xt max:{self.Xt_train.max()}")
            # print(f"xt min:{self.Xt_train.min()}")

        else:
            print("Poor me no npy..................")
            self.create_directory()
            

        if self.for_what == "both":
        #for train source dataset
            # OneHotEncoding.binary_hot_encoding(self)
            
            self.tensor_xs = torch.Tensor(self.Xs_train.astype(np.float16)) 
            self.tensor_ys = torch.Tensor(self.Ys_train.astype(np.float16))
            self.tensor_xs_train = TensorDataset(self.tensor_xs,self.tensor_ys) 
            
            #split into train and validation set
            sn_train_examples = int(len(self.tensor_xs_train) * config["valid_ratio"])
            sn_valid_examples = len(self.tensor_xs_train) - sn_train_examples
            self.source_train_partly, self.source_valid_partly = random_split(self.tensor_xs_train,[sn_train_examples, sn_valid_examples])
            print("--------------------------------------------------------------------")
            print(f'Number of source training examples: {len(self.source_train_partly)}')
            print(f'Number of source validation examples: {len(self.source_valid_partly)}')
            #create dataloader
            self.source_dataloader = DataLoader(self.source_train_partly,batch_size=config["batchsize"],shuffle = True,pin_memory= True,worker_init_fn=np.random.seed(42))
            self.valid_source_dataloader = DataLoader(self.source_valid_partly,batch_size=config["batchsize"],shuffle = False,pin_memory= True,worker_init_fn=np.random.seed(42))
            print("Finally atleast train and valid source dataloader section works 😌 ")
            
            
            #for train target dataset
            print("--------------------------------------------------------------------")
            print(f" Shape of Xt_train is:{self.Xt_train.shape}")
            self.tensor_xt = torch.Tensor(self.Xt_train.astype(np.float16)) 
            self.tensor_yt = torch.Tensor(self.Yt_train.astype(np.float16))
            self.tensor_xt_train = TensorDataset(self.tensor_xt,self.tensor_yt) 
            #split into train and validation set
            tn_train_examples = int(len(self.tensor_xt_train) * config["valid_ratio"])
            tn_valid_examples = len(self.tensor_xt_train) - tn_train_examples
            self.target_train_partly, self.target_valid_partly = random_split(self.tensor_xt_train,[tn_train_examples, tn_valid_examples])
            print("--------------------------------------------------------------------")
            print(f'Number of target training examples: {len(self.target_train_partly)}')
            print(f'Number of target validation examples: {len(self.target_valid_partly)}')
            #create dataloader
            self.target_dataloader = DataLoader(self.target_train_partly,batch_size=config["batchsize"],shuffle = True,pin_memory= True,worker_init_fn=np.random.seed(42)) 
            self.valid_target_dataloader = DataLoader(self.target_valid_partly,batch_size=config["batchsize"],shuffle = False,pin_memory= True,worker_init_fn=np.random.seed(42))
            print("Finally atleast train and valid target dataloader section works 😌 ")
            
            
        elif self.for_what == "training_source":
        #for train source dataset
            # OneHotEncoding.binary_hot_encoding(self)
            
            self.tensor_xs = torch.Tensor(self.Xs_train) 
            self.tensor_ys = torch.Tensor(self.Ys_train)
            self.tensor_xs_train = TensorDataset(self.tensor_xs,self.tensor_ys) 
            
            #split into train and validation set
            sn_train_examples = int(len(self.tensor_xs_train) * config["valid_ratio"])
            sn_valid_examples = len(self.tensor_xs_train) - sn_train_examples
            self.source_train_partly, self.source_valid_partly = random_split(self.tensor_xs_train,[sn_train_examples, sn_valid_examples])
            print("--------------------------------------------------------------------")
            print(f'Number of source training examples: {len(self.source_train_partly)}')
            print(f'Number of source validation examples: {len(self.source_valid_partly)}')
            #create dataloader
            self.source_dataloader = DataLoader(self.source_train_partly,batch_size=config["batchsize"],shuffle = True,pin_memory= True,worker_init_fn=np.random.seed(42))
            self.valid_source_dataloader = DataLoader(self.source_valid_partly,batch_size=config["batchsize"],shuffle = True,pin_memory= True,worker_init_fn=np.random.seed(42))
            print("Finally atleast train and valid source dataloader section works 😌 ")

                        
        elif self.for_what == "training_target":
            #for train target dataset
            print("--------------------------------------------------------------------")
            print(f" Shape of Xt_train is:{self.Xt_train.shape}")
            self.tensor_xt = torch.Tensor(self.Xt_train) 
            self.tensor_yt = torch.Tensor(self.Yt_train)
            self.tensor_xt_train = TensorDataset(self.tensor_xt,self.tensor_yt) 
            #split into train and validation set
            tn_train_examples = int(len(self.tensor_xt_train) * config["valid_ratio"])
            tn_valid_examples = len(self.tensor_xt_train) - tn_train_examples
            self.target_train_partly, self.target_valid_partly = random_split(self.tensor_xt_train,[tn_train_examples, tn_valid_examples])
            print("--------------------------------------------------------------------")
            print(f'Number of target training examples: {len(self.target_train_partly)}')
            print(f'Number of target validation examples: {len(self.target_valid_partly)}')
            #create dataloader
            self.target_dataloader = DataLoader(self.target_train_partly,batch_size=config["batchsize"],pin_memory= True,worker_init_fn=np.random.seed(42)) 
            self.valid_target_dataloader = DataLoader(self.target_valid_partly,batch_size=config["batchsize"],pin_memory= True,worker_init_fn=np.random.seed(42))
            print("Finally atleast train and valid target dataloader section works 😌 ")
            
            
        elif self.for_what == "testing":
            if os.path.exists(self.data_folder + "/npy/Xtest_chicago.npy"):
                self.Xtest = np.load(self.data_folder + "npy/Xtest.npy")/255
                self.Xtest =self.Xtest[:,:3,:,:]
                self.Ytest = np.load(self.data_folder + "npy/Ytest.npy")
                self.Ytest =self.Ytest[:,:3,:,:]
                # percentage=(np.count_nonzero(self.Ytest.reshape(self.Ytest.shape[0],-1),axis=1)/(256*256))*100
                # sel_index=np.where(percentage>1.0)
                # # print(percentage)
                # # print(sel_index[0].shape)
                # self.Ytest=self.Ytest[sel_index]
                # self.Xtest=self.Xtest[sel_index]
                
                print("Found already existing npy")
                print("shape of Xtest: ", self.Xtest.shape)
                print("shape of Ytest: ", self.Ytest.shape)
                # print("shape of Ytest: ", self.Ytest.shape)  #need to remake the dataset
            else:
                # OneHotEncoding.binary_hot_encoding(self)
                self.create_directory()
                
            #for testing dataset
            self.tensor_xp = torch.Tensor(self.Xtest.astype(np.float16)) 
            self.tensor_yp = torch.Tensor(self.Ytest.astype(np.float16)) 
            self.tensor_test = TensorDataset(self.tensor_xp,self.tensor_yp) 
            self.test_dataloader = DataLoader(self.tensor_test, batch_size=1,pin_memory= True,worker_init_fn=np.random.seed(0)) 
            print("Finally atleast test dataloader section works 😌")
        else:
            print("Please be sure you know what you are doing👀")


# if __name__ == "__main__":
#     DATASET = Dataset("/share/projects/erasmus/pratichhya_sharma/version00/data_loader/",128,"testing")
#     DATASET.array_torch()






