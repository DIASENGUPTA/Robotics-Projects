import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev, BSpline
from scipy.optimize import minimize
import time
import cv2
import carla
import torch
from scipy.signal import convolve2d

import torch
import torchvision
#import cv2
import argparse
from PIL import Image
#from utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
import random

class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=68) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(self, cut_size=68, spline_smoothness=10, gradient_threshold=14, distance_maxima_gradient=3):
        self.car_position = np.array([48,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0
    
    def front2bev(self, front_view_image):
        '''
        ##### TODO #####
        This function should transform the front view image to bird-eye-view image.
        
        input:
            front_view_image)96x96x3

        output:
            bev_image 96x96x3
        '''

        def homography_ipmnorm2g(top_view_region):#Top-view Perspective
            src=np.float32([[0,0],[1,0],[0,1],[1,1]])
            H_ipmnorm2g=cv2.getPerspectiveTransform(src,np.float32(top_view_region))
            return H_ipmnorm2g

        bev=np.zeros((96,96,3))
        H,W=96,96
        top_view_region= np.array([[50,-25],[50,25],[0,25],[0,-25]])

        cam_xyz=[-5.5,0,2.5]
        cam_yaw=0
        cam_pitch=8
        cam_roll=0

        width=1920
        height=1080
        fov=64

        focal=width/(2.0*np.tan(fov*np.pi/360.0))
        K=np.identity(3) #Projection Matrix
        K[0,0]=K[1,1]=focal
        K[0,2]=width/2.0
        K[1,2]=height/2.0
        #Translation and Rotation Matrix
        H_g2cam=np.array(carla.Transform(carla.Location(*cam_xyz),carla.Rotation(yaw=cam_yaw
                            ,pitch=cam_pitch,roll=cam_roll)).get_matrix())
        H_g2cam=np.concatenate([H_g2cam[:3,0:2],np.expand_dims(H_g2cam[:3,3],1)],1)
        #Multiplication of K and [R t]
        trans_mat=np.array([[0,1,0],[0,0,-1],[1,0,0]])
        temp_mat=np.matmul(trans_mat, H_g2cam)
        H_g2in=np.matmul(K,temp_mat)
        #Multiplication of K [R t] with top view region perspective
        H_ipmnorm2g=homography_ipmnorm2g(top_view_region)
        H_ipmnorm2in=np.matmul(H_g2in,H_ipmnorm2g)
        #Top-view projection
        S_in_inv=np.array([[1/np.float64(width),0,0],[0,1/np.float64(height),0],[0,0,1]])
        M_ipm2im_norm=np.matmul(S_in_inv,H_ipmnorm2in)

        #Visualisation
        M=torch.zeros(1,3,3)
        M[0]=torch.from_numpy(M_ipm2im_norm).type(torch.FloatTensor)

        linear_points_W=torch.linspace(0,1-1/W,W)
        linear_points_H=torch.linspace(0,1-1/H,H)

        base_grid=torch.zeros(H,W,3)
        base_grid[:,:,0]=torch.ger(torch.ones(H),linear_points_W)
        base_grid[:,:,1]=torch.ger(linear_points_H,torch.ones(W))
        base_grid[:,:,2]=1

        grid=torch.matmul(base_grid.view(H*W,3),M.transpose(1,2))
        lst=grid[:,:,2:].squeeze(0).squeeze(1).numpy()>=0
        grid=torch.div(grid[:,:,0:2],grid[:,:,2:])

        x_vals=grid[0,:,0].numpy()*width
        y_vals=grid[0,:,1].numpy()*height

        indicate_x1=x_vals<width
        indicate_x2=x_vals>0

        indicate_y1=y_vals<height
        indicate_y2=y_vals>0

        indicate=(indicate_x1*indicate_x2*indicate_y1*indicate_y2*lst)*1

        img=front_view_image

        for _i in range(H):
            for _j in range(W):
                _idx= _j+_i*W

                _x=int(x_vals[_idx])
                _y=int(y_vals[_idx])
                _indic=indicate[_idx]

                if _indic==0:
                    continue
                bev[_i,_j]=img[_y,_x]

        bev_image=np.uint8(bev)
        cv2.imwrite('BEVimage.png',bev_image)
        #cv2.imshow('IMG1', np.uint8(bev_image))
        return bev_image 


    def cut_gray(self, state_image_full):
        '''
        ##### TODO #####
        This function should cut the imagen at the front end of the car (e.g. pixel row 68) 
        and translate to grey scale

        input:
            state_image_full 96x96x3

        output:
            gray_state_image 68x96x1
        '''
        #plt.imshow('i',state_image_full)
        #image=self.detect_object(state_image_full)
        img=self.front2bev(state_image_full)# BEV for each image is computed
        cropped_image = img[:68, :, :]
        # Convert to grayscale
        gray_state_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('Grayscaleimage.png',gray_state_image)
        return gray_state_image[::-1]
        


    def edge_detection(self, gray_image):
        '''
        ##### TODO #####
        In order to find edges in the gray state image, 
        this function should derive the absolute gradients of the gray state image.
        Derive the absolute gradients using numpy for each pixel. 
        To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero. 

        input:
            gray_state_image 68x96x1

        output:
            gradient_sum 68x96x1

        '''
        dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        dy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Compute the convolutions of the image with the Sobel kernels
        Ix = convolve2d(gray_image, dx, mode='same')
        Iy = convolve2d(gray_image, dy, mode='same')

        # Compute the magnitude of the gradients
        gradient_sum = np.sqrt(Ix**2 + Iy**2)

        # Set gradients below the threshold to zero
        gradient_sum[gradient_sum < self.gradient_threshold] = 0
        grad_norm = (gradient_sum / np.max(gradient_sum)) * 255

        # Convert the gradients back to uint8 datatype
        grad_norm = grad_norm.astype(np.uint8)
        print(grad_norm.shape)
        cv2.imwrite('Gradientimage.png',np.uint8(grad_norm))
        #plt.figure()
        #plt.imshow(gradient_sum) 
        #plt.show()
        return gradient_sum


    def find_maxima_gradient_rowwise(self, gradient_sum):
        '''
        ##### TODO #####
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima. 
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 68x96x1

        output:
            maxima (np.array) 2x Number_maxima

        '''
        argmaxima=[]
        for i in range(gradient_sum.shape[0]):
            row = gradient_sum[i]
            peaks, _ = find_peaks(row, distance=3)
            argmaxima.append(np.array([[peaks,i]]))

        # Convert the list of maxima to a 2D numpy array
        #argmaxima = np.array(argmaxima)
        return argmaxima


    def find_first_lane_point(self, gradient_sum):
        '''
        Find the first lane_boundaries points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered. 
        Even though there is space for improvement ;) 

        input:
            gradient_sum 68x96x1

        output: 
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        '''
        
        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # loop through the rows
        while not lanes_found:
            
            # Find peaks with min distance of at least 3 pixel 
            argmaxima = find_peaks(gradient_sum[row],distance=3)[0]
            print(argmaxima.shape)

            # if one lane_boundary is found
            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])

                if argmaxima[0] < 48:
                    lane_boundary2_startpoint = np.array([[0,  row]])
                else: 
                    lane_boundary2_startpoint = np.array([[96,  row]])

                lanes_found = True
            
            # if 2 lane_boundaries are found
            elif argmaxima.shape[0] == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1],  row]])
                lanes_found = True

            # if more than 2 lane_boundaries are found
            elif argmaxima.shape[0] > 2:
                # if more than two maxima then take the two lanes next to the car, regarding least square
                A = np.argsort((argmaxima - self.car_position[0])**2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]],  0]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]],  0]])
                lanes_found = True

            row += 1
            
            # if no lane_boundaries are found
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0,  0]])
                lane_boundary2_startpoint = np.array([[0,  0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found


    def lane_detection(self, state_image_full):
        '''
        ##### TODO #####
        This function should perform the road detection 

        args:
            state_image_full [96, 96, 3]

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        '''

        # to gray
        gray_state = self.cut_gray(state_image_full)

        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)

        # first lane_boundary points
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)

        # if no lane was found,use lane_boundaries of the preceding step
        if lane_found:
            
            ##### TODO #####
            #  in every iteration: 
            # 1- find maximum/edge with the lowest distance to the last lane boundary point 
            # 2- append maxium to lane_boundary1_points or lane_boundary2_points
            # 3- delete maximum from maxima
            # 4- stop loop if there is no maximum left 
            #    or if the distance to the next one is too big (>=100)

            # lane_boundary 1

            # lane_boundary 2

            ################
            argmaxima=[]
            for i in range(gradient_sum.shape[0]):
                if(i!=0):
                    row = gradient_sum[i]
                    peaks, _ = find_peaks(row, distance=3)
                    #Last end lane points are taken
                    last_point1 = lane_boundary1_points[-1] if lane_boundary1_points.shape[0] > 0 else None
                    last_point2 = lane_boundary2_points[-1] if lane_boundary2_points.shape[0] > 0 else None
                    #Empty distance arrays are defined
                    distance0=np.ones(1200)*np.inf
                    distance1=np.ones(1200)*np.inf
                    peak_array=[]
                    #Peak array stores all the gradients row wise 
                    #Distance 0 and 1 stores the the distance of the immediate next peaks from the last lane points
                    for j in range(peaks.shape[0]):
                        peak_array.append(np.array([[peaks[j],i]]))
                        distance0[j]=np.linalg.norm(np.array([[peaks[j],i]])-last_point1)
                        distance1[j]=np.linalg.norm(np.array([[peaks[j],i]])-last_point2)
                    #If peak array length is 2, those 2 points are taken as boundaries
                    if(len(peak_array)==2):
                        print(peak_array)
                        lane_boundary1_points = np.vstack((lane_boundary1_points, peak_array[1]))
                        lane_boundary2_points = np.vstack((lane_boundary2_points, peak_array[0]))
                    #Otherwise, the gradient with minimum distance from the last lane points is taken
                    elif(len(peak_array)>0):
                        x=np.argmin(distance1)
                        #for i in range(x+1):
                        distance0[x]=np.inf# If one peak array point is found to be closest to one lane that is removed while searching for the same to prevent the condition where both last end points are closest to the same peak in the next row
                        lane_boundary1_points = np.vstack((lane_boundary1_points, peak_array[np.argmin(distance0)]))
                        lane_boundary2_points = np.vstack((lane_boundary2_points, peak_array[np.argmin(distance1)]))
                    else:
                        continue
                
            '''
            ##### TODO #####
            # spline fitting using scipy.interpolate.splprep 
            # and the arguments self.spline_smoothness
            # 
            # if there are more lane_boundary points points than spline parameters 
            # else use perceding spline
            '''
            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:

                # Pay attention: the first lane_boundary point might occur twice
                # lane_boundary 1
                #print(lane_boundary1_points)
                lane_boundary1, u1 = splprep([lane_boundary1_points[:,0], lane_boundary1_points[:,1]], s=self.spline_smoothness)
                #lane_boundary1 = BSpline(*tck1)
            
                # lane_boundary 2
                #print(lane_boundary2_points)
                lane_boundary2, u2= splprep([lane_boundary2_points[:,0], lane_boundary2_points[:,1]], s=self.spline_smoothness)
                #lane_boundary2 = BSpline(*tck2)
                
            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
            ################

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return lane_boundary1, lane_boundary2


    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        '''
        Plot lanes and way points
        '''
        # evaluate spline for 6 different spline parameters.
        t = np.linspace(0, 1, 6)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))
        
        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])
        plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1]+96-self.cut_size, linewidth=5, color='orange')
        plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1]+96-self.cut_size, linewidth=5, color='orange')
        if len(waypoints)>0:
            #print(waypoints)
            plt.scatter(waypoints[0], waypoints[1]+96-self.cut_size, color='white')

        plt.axis('off')
        plt.xlim((-0.5,95.5))
        plt.ylim((-0.5,95.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()
