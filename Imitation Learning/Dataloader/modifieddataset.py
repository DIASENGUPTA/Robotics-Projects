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

    def __len__(self):
        return len(self.data_list)

    def load_data(self,idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        #d=data[0][::10,::10,:]
        img = self.transform(data[0])
        image= self.transform2(img)
        action = torch.Tensor(data[1])
        return image,action


    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        obs, label = self.load_data(idx)
        if idx<3:
            obs = np.concatenate([np.zeros(obs.shape) for i in range(4)], axis=0)#if the index is less than 0 then 0 is appended in the tensor because it does not have sufficient previous frames
        else:
            # concatenating the last num_frames-1 observations with the current observation
            obs = np.concatenate([self.load_data(idx - 3 + i)[0] for i in range(4)], axis=0)#3 previous frames are loaded at every instance and appended to the current frame
        obs1= torch.Tensor(obs)
        return obs1, label
        


def get_dataloader(data_dir, batch_size, num_workers=4):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                #shuffle=shuffle
            )
    