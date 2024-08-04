import time
import random
import argparse

import torch
#import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from networkaugmented import ClassificationNetwork
from augmenteddata import get_dataloader


def train(data_folder, save_path):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    
    device = torch.device('cuda')
    nr_epochs = 100
    batch_size = 64
    nr_of_classes = 7  # needs to be changed
    start_time = time.time()

    infer_action = ClassificationNetwork()
    infer_action1=nn.DataParallel(infer_action)
    infer_action1.to(device)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-2)
    #optimizer = torch.optim.SGD(infer_action.parameters(), lr=0.01)
    

    train_loader = get_dataloader(data_folder, batch_size)
    #print(train_loer.shape)
    loss_values=[]
    for epoch in range(nr_epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []
        #print("Hi")

        for batch_idx, batch in enumerate(train_loader):
            batch_in1,batch_in2,batch_in3, batch_gt = batch[0].to(device),batch[1].to(device),batch[2].to(device),infer_action.actions_to_classes(batch[3]).to(device)
            #print(batch_in.shape)
            #print("Hello")

            batch_out1 = infer_action1(batch_in1)
            batch_out2= infer_action1(batch_in2)
            batch_out3=infer_action1(batch_in3)
            b_out=torch.cat((batch_out1,batch_out2,batch_out3))#All three class outputs for formatted images are added to the tensor
            b_g=torch.cat((batch_gt,batch_gt,batch_gt))#All the corresponding labels are also added.
            #print(batch_out.shape)
            loss = cross_entropy_loss(b_out, b_g)

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
    plt.title('Augmented Data Plot')
    plt.plot(loss_values)
    plt.savefig('/projectnb/rlvn/students/ksg25/PythonAPI/examples/AugmentDatanew.jpg')


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
                    C = number of classes
    batch_out:      torch.Tensor of size (batch_size, C)
    batch_gt:       torch.Tensor of size (batch_size, C)
    return          float
    """
    probs = batch_out - torch.logsumexp(batch_out, dim=1, keepdim=True)
    
    batch_gt_probs = torch.sum(batch_gt * probs, dim=1)
    
    loss = -torch.mean(batch_gt_probs)
    
    #return loss
    #loss=nn.functional.binary_cross_entropy_with_logits(batch_out,batch_gt)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="./", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="./", type=str, help='path where to save your model in .pth format')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)