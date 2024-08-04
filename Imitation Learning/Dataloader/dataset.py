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

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        data = np.load(self.data_list[idx], allow_pickle=True)
        img = self.transform(data[0])
        image= self.transform2(img)
        action = torch.Tensor(data[1])

        return (image, action)


def get_dataloader(data_dir, batch_size, num_workers=4):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                #shuffle=shuffle
            )
    