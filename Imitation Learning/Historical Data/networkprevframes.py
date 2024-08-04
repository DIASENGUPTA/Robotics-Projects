import torch
import torch.nn as nn
import numpy as np


class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        #self.gpu = torch.device('cuda')
        #self.device = torch.device('cuda')
        self.conv_layers = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        conv_output_shape = self._get_conv_output([12,96,128])#The image dimension is now changed to 12*96*128 due to extra frames
        self.linear_layers = nn.Sequential(
            nn.Linear(conv_output_shape, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 7)
        )

    def _get_conv_output(self, shape):
        # Test the output shape of the convolutional layers
        x = torch.rand(1, *shape)
        x = self.conv_layers(x)
        return int(np.prod(x.size()))

    def forward(self, observation):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        x = observation  # Reshape the input from (batch_size, height, width, channel) to (batch_size, channel, height, width)
        #print(x.shape)
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
        #return nn.functional.softmax(x, dim=1)

    def actions_to_classes(self, actions):
        """
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are C different classes, then
        every action is represented by a C-dim vector which has exactly one
        non-zero entry (one-hot encoding). That index corresponds to the class
        number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size C
        """
        classes=[]
        for action in actions:
            class_action=torch.zeros(7)
            '''
            Classes defined :
            0: empty
            1: throttle
            2: break
            3: steer right
            4: steer left
            5: steer right and brake
            6: steer left and brake
            '''
            #Check the condition classes
            if action[0]==0 and action[1]==0 and action[2]==0:
                class_action[0]=1#no move
            elif action[0]>0.04 and action[2]==1:
                class_action[5]=1#steer right and break
            elif action[0]<-0.04 and action[2]==1:
                class_action[6]=1#steer left and break
            elif action[0]>0.04 and action[1]>0 and action[2]==0:
                class_action[3]=1#steer right
            elif action[0]<-0.04 and action[1]>0 and action[2]==0:
                class_action[4]=1#steer left
            elif action[1]>0 and action[2]==0:
                class_action[1]=1#throttle
            elif action[2]==1:
                class_action[2]=1#brake
            classes.append(torch.Tensor(class_action))
        classes=torch.stack(classes)
        return classes


    def scores_to_action(self, scores):
        """
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [accelaration, steering, braking].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        for c in scores:
            v=[0,0,0]
            x=torch.argmax(c)
            if x==1:
                v[1]=0.39
            elif x==2:
                v[2]=1
            elif x==3:
                v[0]=1
                v[1]=0.39
                v[2]=0
            elif x==4:
                v[0]=-1
                v[1]=0.39
                v[2]=0
            elif x==5:
                v[0]=1
                v[2]=1
            elif x==6:
                v[0]=-1
                v[2]=1
        return v
        




        


