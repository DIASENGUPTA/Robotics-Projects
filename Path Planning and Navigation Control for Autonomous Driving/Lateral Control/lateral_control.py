import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
from lane_detection import LaneDetection


class LateralController:
    '''
    Lateral control using the Stanley controller

    functions:
        stanley 

    init:
        gain_constant (default=5)
        damping_constant (default=0.5)
    '''


    def __init__(self, gain_constant=0.01, damping_constant=0.5):

        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0
        #self.epsilon=0.0001


    def stanley(self, waypoints, speed):
        '''
        ##### TODO #####
        one step of the stanley controller with damping
        args:
            waypoints (np.array) [2, num_waypoints]
            speed (float)
        '''
        # derive orientation error as the angle of the first path segment to the car orientation
        print(waypoints)
        diff_x = waypoints[0][1] - waypoints[0][0]
        diff_y = waypoints[1][1] - waypoints[1][0]
        orientation_error = np.arctan2(diff_x, diff_y)
        print(orientation_error)
        
        waypoints_array = np.array(waypoints)
         
        #cross_track error    
        crosstrack_error = waypoints_array[0,1]-waypoints_array[0,0]
        print('Try',(self.gain_constant * crosstrack_error) / (1+speed))
        psi_crosstrack = np.arctan((self.gain_constant * crosstrack_error) / (1+speed))
        print(psi_crosstrack,'c')

        #Final error
        sigma =orientation_error+psi_crosstrack

        #Damping
        steering_angle=sigma-self.damping_constant*(sigma-self.previous_steering_angle)
    
        self.previous_steering_angle=steering_angle
        #steering_angle =sigma - 2*np.pi if sigma > np.pi else  sigma + 2*np.pi
        # clip to the maximum stering angle (0.4) and rescale the steering action space
        print(np.clip(steering_angle, -0.7, 0.7) / 0.7)
        return (np.clip(steering_angle, -0.7, 0.7) / 0.7) 






