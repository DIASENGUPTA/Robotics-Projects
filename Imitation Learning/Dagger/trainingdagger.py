import time
import random
import argparse

import torch
#import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from networkdagger import ClassificationNetwork
from dataset import get_dataloader

#Tried training dagger with infer_action as expert policy
beta=0.5

def train(data_folder, save_path):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    #Normal training function
    device = torch.device('cuda')
    nr_epochs = 5
    batch_size = 64
    nr_of_classes = 7  # needs to be changed
    start_time = time.time()

    infer_action = ClassificationNetwork()
    infer_action.to(device)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-2)
    pi_1=infer_action()

    train_loader = get_dataloader(data_folder, batch_size)


    for epoch in range(nr_epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            batch_in, batch_gt = batch[0].to(device), infer_action.actions_to_classes(batch[1]).to(device)
            batch_out = infer_action(batch_in)
            loss = cross_entropy_loss(batch_out, batch_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))
    D=[]
    D_out=[]
#DAgger iterations
    for iteration in range(10):#performing 10 dagger interations
        pi_i = lambda batch_in: beta * infer_action(batch_in) + (1-beta) * pi_1(batch_in)# Defining the new policy pi_i as combination of expert and learner policy.
        #In this case Pi_1 is the learners policy

    # Collecting new data from the current policy
        for batch_idx, batch in enumerate(train_loader):
            batch_in1, batch_gt1 = batch[0].to(device), infer_action.actions_to_classes(batch[1]).to(device)#Inserting a new set of inputs
            batch_out1 = pi_i(batch_in)#Getting new set of outputs

    # Adding the new data to the old one and converting to tensors
        D+=batch_in1
        D1=torch.cuda.FloatTensor(D)
        D_out+=batch_out1
        D2=torch.cuda.FloatTensor(D_out)

        for epoch in range(nr_epochs):
            total_loss = 0

            batch_out = pi_i(D1)#Getting output according to the modified policy
            loss = cross_entropy_loss(batch_out, D2)#new loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            time_per_epoch = (time.time() - start_time) / (epoch + 1)
            time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
            print("Epoch %5d\t[DAgger %d]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, iteration, total_loss, time_left))
        pi_1=pi_i

# Save trained model
    torch.save(infer_action.state_dict(), save_path)# Saving the updated policy every time


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
                    C = number of classes
    batch_out:      torch.Tensor of size (batch_size, C)
    batch_gt:       torch.Tensor of size (batch_size, C)
    return          float
    """
    #log_softmax
    log_probs = batch_out - torch.logsumexp(batch_out, dim=1, keepdim=True)
    
    # probability of each 
    batch_gt_log_probs = torch.sum(batch_gt * log_probs, dim=1)
    
    #negative log likelihood
    loss = -torch.mean(batch_gt_log_probs)
    
    #return loss
    #loss=nn.functional.binary_cross_entropy_with_logits(batch_out,batch_gt)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="./", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="./", type=str, help='path where to save your model in .pth format')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)