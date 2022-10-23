# teleoperate the robot and perform SLAM

# basic python packages
from pickle import FALSE, TRUE
import numpy as np
import cv2 
import os, sys
import glob
import time

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import PenguinPi # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
#import shutil # python package for file operations
#import copy
from pathlib import Path

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import components for the detector
#import torch
import json
#from sklearn.cluster import KMeans
from network.scripts.detector import Detector

class Operate:
    def __init__(self, args):
        self.folder = 'lab_output/'
 
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length = 0.07) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'read_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.ckpt, use_gpu=False)
            self.network_vis = np.ones((240, 320,3))* 100
         #Initialisng paramaters and arrays to be used
        self.robot_pose = np.array([0,0,0])
        self.paths = [[[0,0],[0.5,0.5]],[[0.5,0.5],[1,1]],[[1,1],[1,0.5]]]
        self.forward = False
        self.point_idx = 1
        self.waypoints = []
        self.wp = [0,0]
        self.min_dist = 50
        self.auto_path = False
        self.taglist = []
        #Defining parameters for SLAM
        self.P = np.zeros((3,3))
        self.marker_pos = np.zeros((2,10))
        self.lmc = 1e-6
        self.path_idx = 0
        self.fruit_list = []
        self.fruit_true_pos = []
        self.aruco_true_pos = np.empty([10, 2])
        self.pred_count = 0
        #Contorl and travel parameters
        self.tick = 20
        self.turning_tick = 5
        self.boundary = 0.30
        #Defining the model
        fileK = "{}intrinsic.txt".format('./calibration/param/')
        self.camera_matrix = np.loadtxt(fileK, delimiter=',')
        #Variable to store fruit boxes and info
        self.boxes = []
        self.completed_img_dict = {}
        self.dict_idx = 0
        self.tagret_pose_dict ={}
        self.grid = cv2.imread('grid.png')
        self.raw_boxes=[]

        #Setting a condition for slam to map
        self.SLAM_DONE =FALSE
    
    # wheel control
    def control(self):       
        if args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas

    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        #Get the posititins of the fruits
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)
 
    def bounding_box_output(self, box_list):
        with open(f'lab_output/img_{self.pred_count}.txt', "w") as f: #Chane thye a back to w if it does not fix it
            json.dump(box_list, f)
            self.pred_count += 1

    # save images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        f1_ = os.path.join(self.folder, f'pred_{self.image_id}.png')
        if self.command['save_image']:
            self.command['save_image'] = False
            self.detector_output, self.network_vis, self.bounding_boxes, pred_count = self.detector.detect_single_image(self.img)
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{pred_count} fruits detected'
            image = cv2.cvtColor(self.file_output[0], cv2.COLOR_BGR2RGB)
            self.pred_fname = self.output.write_image(image,self.file_output[1])
            self.bounding_box_output(self.bounding_boxes) #save bounding box text file
            self.notification = f'Prediction is saved to pred_{self.pred_count-1}.png'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)
    
    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            """
            self.output.write_slam_map(self.ekf)
            """
            self.notification = 'Map of ARUCO markers is saved'
            import TargetPoseEst
            import mapping_eval
            self.command['output'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, position=(h_pad, v_pad))

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,(320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view,position=(h_pad, 240+2*v_pad))

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector', position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))
    # keyboard teleoperation        
    def update_keyboard(self):
        relative_speed = 1
        for event in pygame.event.get():
            ########### replace with your M1 codes ###########
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [relative_speed, 0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-relative_speed, 0]            
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, relative_speed] #[1,1] for wide arc
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -relative_speed] #[1,-1] for wide arc
            # Optional:
            # stop (if no key pressed)
            elif event.type == pygame.KEYUP and (event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]):
                self.command['motion'] = [0, 0]
            ####################################################
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
    
            # quit
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

    def drive_robot(self):
        waypoint_x = self.wp[0]
        waypoint_y = self.wp[1]
        #Updating the robots psoe
        self.robot_pose = self.ekf.get_state_vector()
        robot_x = self.robot_pose[0]
        robot_y = self.robot_pose[1]

        self.distance = np.sqrt((waypoint_x-robot_x)**2 + (waypoint_y-robot_y)**2) #calculates distance between robot and waypoint

        robot_theta = self.robot_pose[2]
        waypoint_angle = np.arctan2((waypoint_y-robot_y),(waypoint_x-robot_x))
        theta1 = robot_theta - waypoint_angle
        if waypoint_angle < 0:
            theta2 = robot_theta - waypoint_angle - 2*np.pi
        else:
            theta2 = robot_theta - waypoint_angle + 2*np.pi

        if abs(theta1) > abs(theta2):
            self.theta_error = theta2
        else:
            self.theta_error = theta1

        if self.forward == False:
            #Update turning tick speed depending on theta_error to waypoint
            self.turning_tick = int(abs(5 * self.theta_error) + 3)
            if self.theta_error > 0:
                self.command['motion'] = [0,-1]
                self.notification = 'Robot is turning right'

            if self.theta_error < 0:
                self.command['motion'] = [0,1]
                self.notification = 'Robot is turning left'

        # stop turning if less than threshold
        if not self.forward:
            if abs(self.theta_error)  < 0.05:
                self.command['motion'] = [0,0]
                self.notification = 'Robot stopped turning'
                self.forward = True #go forward now
                return

        #Driving forward
        if self.forward:
            #Update tick speed depending on distance to waypoint
            self.tick = int(10 * self.distance  + 30)

            #Checking if distance is increasing, stop driving
            if self.distance > self.min_dist + 0.1:
                self.command['motion'] = [0,0]
                self.notification = 'Robot stopped moving'
                self.forward = False
                self.min_dist = 50
                return

            # Distance is decreasing
            else:
                #Drive until goal arrived
                distance_threshold = 0.1 #0.05
                if self.distance < distance_threshold:
                    self.command['motion'] = [0,0]
                    self.notification = 'Robot arrived'
                    self.forward = False
                    self.min_dist = 50

                    #Check if last path and last waypoint reached
                    if self.point_idx == len(self.waypoints) - 1: #reached last wp of path
                        if self.path_idx == len(self.paths) - 1: #stop pathing
                            self.auto_path = False
                        else: #Increment path and reset idx
                            self.path_idx += 1
                            self.waypoints = self.paths[self.path_idx]
                            self.point_idx = 1 
                            self.wp = self.waypoints[self.point_idx]
                        self.pibot.set_velocity([0,0],time = 3)
                    else:
                        self.point_idx += 1
                        self.wp = self.waypoints[self.point_idx]
                    print(f"Moving to new waypoint {self.wp}")
                    return

                else:
                    #ReAdjust angle if theta_error increased
                    if abs(self.theta_error) > 15/57.3 and self.distance > 0.15: #0.2
                        self.command['motion'] = [0,0]
                        self.notification = 'Readjusting angle'
                        self.forward = False
                        self.min_dist = 50
                        return

                    self.min_dist = self.distance
                    self.command['motion'] = [1,0]
                    self.notification = 'Robot moving forward'        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/best.pt')
    parser.add_argument("--slam_map", default="lab_output/slam_map.txt")
    parser.add_argument("--fruit_poses", default="lab_output/targets.txt")
    args, _ = parser.parse_known_args()

    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    # delete old lab_output files
    files = glob.glob('lab_output/*')
    for f in files:
        os.remove(f)

    start = False
    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)
    while start:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        #operate.detect_target()
        operate.save_image()
        # visualise
        operate.draw(canvas)
        pygame.display.update()