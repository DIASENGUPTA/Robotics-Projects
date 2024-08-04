import time
import random
import argparse

import torch
#import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from network4class import ClassificationNetwork
from dataset import get_dataloader


def train(data_folder, save_path):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    
    device = torch.device('cuda')
    nr_epochs = 100
    batch_size = 64
    nr_of_classes = 4  # needs to be changed
    start_time = time.time()

    infer_action = ClassificationNetwork()
    infer_action1=nn.DataParallel(infer_action)
    infer_action1.to(device)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-5)
    l=nn.BCELoss()# BCELoss is used for this classification

    train_loader = get_dataloader(data_folder, batch_size)
    #print(train_loader.shape)
    loss_values=[]
    for epoch in range(nr_epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []
        #print("Hi")

        for batch_idx, batch in enumerate(train_loader):
            batch_in, batch_gt = batch[0].to(device), infer_action.actions_to_classes(batch[1]).to(device)
            #print(batch_gt)
            #print("Hello")

            batch_out = infer_action1(batch_in)
            #print(batch_out)
            loss=l(batch_out, batch_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #average_loss=total_loss/(batch_idx+1)
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))
        loss_values.append(total_loss)
    torch.save(infer_action1, save_path)
    plt.plot(loss_values)
    plt.title('Loss for 4 class plot')
    plt.savefig('/projectnb/rlvn/students/ksg25/PythonAPI/examples/Loss4classNew.jpg')

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="./", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="./", type=str, help='path where to save your model in .pth format')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)