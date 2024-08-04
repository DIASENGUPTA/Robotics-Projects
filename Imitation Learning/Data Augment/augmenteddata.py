import glob

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset



class CarlaDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir+'*.npy') #need to change to your data format
        #print(self.data_list)

        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        self.transform2= transforms.Resize(96);
        self.transform3= transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)])#the brightness, contrast, saturation etc was changed
        self.transform4 = transforms.Compose([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))])#noise in the form of gaussian blur was introduced
        #Other forms of transforms like horizontal, vertical flips are also possible.

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        d=[]
        a=[]
        data = np.load(self.data_list[idx], allow_pickle=True)
        #d=data[0][::10,::10,:]
        img = self.transform(data[0])
        image= self.transform2(img)
        image1=self.transform3(image)
        image2=self.transform4(image)
        action = torch.Tensor(data[1])
        
        #print("g")

        return image,image1,image2,action


def get_dataloader(data_dir, batch_size, num_workers=4):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                #shuffle=shuffle
            )
    