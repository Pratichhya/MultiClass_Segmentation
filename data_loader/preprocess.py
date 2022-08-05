# Filtering based on percentage of pixels with building pixels. for now I used 0.1 percent but you can increase them as need
import numpy as np

class PreProcess:    
    def remove(self):
        print("--------------------------------------------------------------------")
        print("Total pixels in Ys_train before removing some with all 0 is⏮️  ",self.Ys_train.size)
        print(f"Number of Non-Zeroes before is⏮️  {np.count_nonzero(self.Ys_train)}")
        print(f"Number of Zeroes before is⏮️  {self.Ys_train.size - np.count_nonzero(self.Ys_train)}")
        self.Ys_train_ck=self.Ys_train
        self.Ys_train_ck=self.Ys_train_ck.reshape(self.Ys_train_ck.shape[0],-1)
        percentage=(np.count_nonzero(self.Ys_train_ck,axis=1)/self.Ys_train_ck.shape[-1])*100
        sel_index=np.where(percentage>0.1)
        self.Ys_train=self.Ys_train[sel_index]
        self.Xs_train=self.Xs_train[sel_index]
        print("/////////////////////////////////////////////////////////////////////")
        print("Total pixels in Ys_train after removing some with all 0 is⏩   ",self.Ys_train.size)
        print(f"Number of Non-Zeroes after is⏩   {np.count_nonzero(self.Ys_train)}")
        print(f"Number of Zeroes after is⏩   {self.Ys_train.size - np.count_nonzero(self.Ys_train)}")
        print("--------------------------------------------------------------------")
        
    def single_byte_scaling(self, data, cmin=None, cmax=None, high=255, low=0):
        # """
        # Converting the input image to uint8 dtype and scaling
        # the range to ``(low, high)`` (default 0-255). If the input image already has 
        # dtype uint8, no scaling is done.
        # :param data: 16-bit image data array
        # :param cmin: bias scaling of small values (def: data.min())
        # :param cmax: bias scaling of large values (def: data.max())
        # :param high: scale max value to high. (def: 255)
        # :param low: scale min value to low. (def: 0)
        # :return: 8-bit image data array
        # """
        if data.dtype == np.uint8:
            return data
        else:
            if high > 255:
                high = 255
            if low < 0:
                low = 0
            if high < low:
                raise ValueError("`high` should be greater than or equal to `low`.")

            if cmin is None:
                cmin = data.min()
            if cmax is None:
                cmax = data.max()

            cscale = cmax - cmin
            if cscale == 0:
                cscale = 1

            scale = float(high - low) / cscale
            bytedata = (data - cmin) * scale + low
            return np.floor((bytedata.clip(low, high) + 0.5)).astype(np.uint8)
    
    def bytes8caling(self,IMG, chanell_order = "first"):
    #     assert len(IMG.shape) == 3, "input image is not RGB image, please try to consider it"
        if chanell_order == "first":
            out_img = np.dstack((PreProcess.single_byte_scaling(self, IMG[0, :,:]), PreProcess.single_byte_scaling(self, IMG[1,:,:]),  PreProcess.single_byte_scaling(self, IMG[2, :,:]), PreProcess.single_byte_scaling(self, IMG[3,:,:])))
        else:
            out_img = np.dstack((PreProcess.single_byte_scaling(IMG[:,:,0]), PreProcess.single_byte_scaling(IMG[:,:,1]),  PreProcess.single_byte_scaling(IMG[:,:,2]), PreProcess.single_byte_scaling(IMG[:,:,3])))

        return out_img
    
    def selected_patches(self):
        test_data = np.load('/share/projects/erasmus/pratichhya_sharma/version00/data_loader/npy/Xs_train.npy')
        selected_patched=[]
        for indx in range(len(test_data)):
            a = self.Xt_train[indx,:,:,:]
            selected_patched.append(a)
        self.Xt_train = np.asarray(selected_patched)
        print("There are %i number of target training patches" % (self.Xt_train.shape[0]))
        return self.Xt_train
    
    
    def source_norm(self):
        a = self.Xs_train.max()
        b = self.Xs_train.min()
        self.Xs_train = ((self.Xs_train-b)/(a-b))
        return self.Xs_train