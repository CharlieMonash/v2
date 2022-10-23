# teleoperate the robot and perform SLAM

# basic python packages
from pickle import FALSE, TRUE
import numpy as np
import cv2 
import os, sys
import time

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import PenguinPi # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations
import copy
from pathlib import Path

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import components for the detector
import torch
import json
from sklearn.cluster import KMeans
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
        weight_path =  'network/scripts/model/best.pt'
        #self.detector = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path, force_reload=True)    
        calibration_path = 'calibration/param/'
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



    #Target Pose Est
    def get_bounding_box(self,fruit_select):
        #Multiplying the pixel values by the appropraite scale
        xmin = fruit_select[0] *480/240
        ymin = fruit_select[1] *640/320
        xmax = fruit_select[2] *480/240
        ymax = fruit_select[3] *640/240
        fruit = fruit_select[5]

        fruit_xcent = (xmin + xmax)/2
        fruit_ycent = (ymin + ymax)/2 
        fruit_width = xmax - xmin
        fruit_height = ymax - ymin

        class_converter = {0:1,1:3,2:4,3:5,4:2}

        return (class_converter[fruit], [fruit_xcent,fruit_ycent, fruit_width, fruit_height])
        
    # estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
    def estimate_pose(self):
        focal_length = self.camera_matrix[0][0]
        # actual sizes of targets [For the simulation models]
        # You need to replace these values for the real world objects
        target_dimensions = []

        apple_dimensions = [0.075448, 0.074871, 0.071889]
        target_dimensions.append(apple_dimensions)
        lemon_dimensions = [0.060588, 0.059299, 0.053017]
        target_dimensions.append(lemon_dimensions)
        pear_dimensions = [0.0946, 0.0948, 0.135]
        target_dimensions.append(pear_dimensions)
        orange_dimensions = [0.0721, 0.0771, 0.0739]
        target_dimensions.append(orange_dimensions)
        strawberry_dimensions = [0.052, 0.0346, 0.0376]
        target_dimensions.append(strawberry_dimensions)

        target_list = ['apple', 'lemon', 'pear', 'orange', 'strawberry']

        target_pose_dict = {}
        # for each target in each detection output, estimate its pose
        for target_num in self.completed_img_dict.keys():
            box = self.completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
            robot_pose = self.completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
            true_height = target_dimensions[target_num-1][2]
            
            ######### Replace with your codes #########
            target_pose = {'y': 0.0, 'x': 0.0}
            
            cam_res = 640 # camera resolution in pixels

            A = focal_length * true_height / box[3][0] # actual depth of object
    
            x_robot = robot_pose[0][0]
            y_robot = robot_pose[1][0]
            theta_robot = robot_pose[2][0]

            x_camera = cam_res/2 - box[0][0]
            theta_camera = np.arctan(x_camera/focal_length)
            theta_total = theta_robot + theta_camera

            y_object = A * np.cos(theta_total)
            x_object = A * np.cos(theta_total)
            
            x_object_world = x_robot + x_object
            y_object_world = y_robot + y_object

            target_pose = {'y':y_object_world,'x':x_object_world}
            #print(target_pose)
            #print("")
            target_pose_dict[target_list[target_num-1]] = target_pose
            ###########################################
        return target_pose_dict

    def merge_to_mean(self,position_est, remove_outlier = False):

        # Inputs:
        # position_est : An numpy array of coordinates {position_est[estimation #][0 = x, 1 = y]}
        # remove_outlier : Boolean (Remove outliers using Standard Distribution z-scores)
        # Outputs:
        # new_mean : An numpy array of coordinates {new_mean[0 = x, 1 = y]}

        # Check if the position_est has no elements
        if len(position_est) == 0:
            return None

        # Set up working parameters
        position_est_result = []
        z_threshold = 3

        # Compute mean and standard deviations
        means = np.mean(position_est, axis = 0)
        stds = np.std(position_est, axis = 0)
        mean_x = means[0]
        std_x = stds[0]
        mean_y = means[1]
        std_y = stds[1]
        
        # Remove outliers
        if remove_outlier:
            for i in range(len(position_est)):
                coordinates = position_est[i]
                z_score_x = (coordinates[0] - mean_x)/std_x
                z_score_y = (coordinates[1] - mean_y)/std_y
                if np.abs(z_score_x) > z_threshold or np.abs(z_score_y) > z_threshold:
                    position_est_result.append(coordinates)
        else:
            position_est_result = position_est

        # Compute Mean
        new_mean = np.mean(position_est_result, axis = 0)

        return new_mean


    def sort_locations_and_merge(self,position_est, distance_threshold = 0.3, remove_outlier = False, use_Kmeans = False):

        # Inputs:
        # position_est : An numpy array of coordinates {position_est[estimation #][0 = x, 1 = y]}
        # distance_threshold : the distance assumption that two fruits of the same type will be apart for
        # remove_outlier : Boolean (Remove outliers using Standard Distribution z-scores)
        # Outputs:
        # new_mean : An numpy array of coordinates {new_mean[0 = x, 1 = y]}

        # Initialize two sets of position estimations for each fruit of the same type
        position_est1 = []
        position_est2 = []

        # Sort data
        for i in range(len(position_est)):

            if(use_Kmeans):

                kmeans = KMeans(n_clusters = 2)
                kmeans.fit(position_est)
                if(kmeans.labels_[i] == 0):
                    position_est1.append(position_est[i])
                else:
                    position_est2.append(position_est[i])

            else:

                if(i == 0): # Take the first position estimation as the reference for the first fruit
                    position_est1.append(position_est[i])
                    continue
                else:
                    coordinates = position_est[i]
                    x_distance = np.abs(coordinates[0] - position_est[0][0])
                    y_distance = np.abs(coordinates[1] - position_est[0][1])
                    distance = np.sqrt(x_distance ** 2 + y_distance ** 2)
                    if(distance < distance_threshold):
                        position_est1.append(coordinates)
                    else:
                        position_est2.append(coordinates)

        # Merge position estimations
        position1 = self.merge_to_mean(position_est1, remove_outlier)
        position2 = self.merge_to_mean(position_est2, remove_outlier)

        # return the position estimations
        positions = []
        if(position1 is not None):
            positions.append(position1)
        if(position2 is not None):
            positions.append(position2)
        return positions
            

    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    def merge_estimations(self,target_pose_dict):
        target_map = target_pose_dict
        apple_est, lemon_est, pear_est, orange_est, strawberry_est = [], [], [], [], []
        target_est = {}
        
        # combine the estimations from multiple detector outputs
        for f in target_map:
            for key in target_map[f]:
                if key.startswith('apple'):
                    apple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
                elif key.startswith('lemon'):
                    lemon_est.append(np.array(list(target_map[f][key].values()), dtype=float))
                elif key.startswith('pear'):
                    pear_est.append(np.array(list(target_map[f][key].values()), dtype=float))
                elif key.startswith('orange'):
                    orange_est.append(np.array(list(target_map[f][key].values()), dtype=float))
                elif key.startswith('strawberry'):
                    strawberry_est.append(np.array(list(target_map[f][key].values()), dtype=float))

        ######### Replace with your codes #########
        # TODO: the operation below takes the first three estimations of each target type, replace it with a better merge solution
        remove_outlier = False
        use_Kmeans = False
        if len(apple_est) > 1:
            apple_est = self.sort_locations_and_merge(apple_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)
        if len(lemon_est) > 1:
            lemon_est = self.sort_locations_and_merge(lemon_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)
        if len(pear_est) > 1:
            pear_est = self.sort_locations_and_merge(pear_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)
        if len(orange_est) > 1:
            orange_est = self.sort_locations_and_merge(orange_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)
        if len(strawberry_est) > 1:
            strawberry_est = self.sort_locations_and_merge(strawberry_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)

        for i in range(2):
            try:
                target_est['apple_'+str(i)] = {'y':apple_est[i][0], 'x':apple_est[i][1]}
            except:
                pass
            try:
                target_est['lemon_'+str(i)] = {'y':lemon_est[i][0], 'x':lemon_est[i][1]}
            except:
                pass
            try:
                target_est['pear_'+str(i)] = {'y':pear_est[i][0], 'x':pear_est[i][1]}
            except:
                pass
            try:
                target_est['orange_'+str(i)] = {'y':orange_est[i][0], 'x':orange_est[i][1]}
            except:
                pass
            try:
                target_est['strawberry_'+str(i)] = {'y':strawberry_est[i][0], 'x':strawberry_est[i][1]}
            except:
                pass
        ########################################### 
        return target_est
    
    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            #Setting the min confidence of the network
            self.detector.conf = 0.4
            result = self.detector(self.img)
            self.command['inference'] = False
            self.file_output = (np.squeeze(result.render()), self.ekf)
            #self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'
            self.notification = f'{self.network_vis.shape[0]} target type(s) detected'
            #Add the target info along with the robots pose to a dictionary
            image_data = result.pandas().xyxy[0]
            image_data_list = image_data.values.tolist()
            yolo = []
            target_lst_box = [[], [], [], [], []]
            target_lst_pose = [[], [], [], [], []]
            for fruit_data in image_data_list:
                yolo.append(self.get_bounding_box(fruit_data))

            all_vals = yolo

            for (target_num, box) in all_vals:
                pose = self.robot_pose # [x, y, theta] 
                target_lst_box[target_num-1].append(box) # bounding box of target
                target_lst_pose[target_num-1].append(np.array(pose).reshape(3,)) # robot pose

            # if there are more than one objects of the same type, combine them
            for i in range(5):
                if len(target_lst_box[i])>0:
                    box = np.stack(target_lst_box[i], axis=1)
                    pose = np.stack(target_lst_pose[i], axis=1)
                    self.completed_img_dict[self.dict_idx+1] = {'target': box, 'robot': pose}
                    #increment dictionary index
                    self.dict_idx +=1
        
    def bounding_box_output(self, box_list):
        with open(f'lab_output/img_{self.pred_count}.txt', "a") as f: #Chane thye a back to w if it does not fix it
            json.dump(box_list, f)
            self.pred_count += 1

    # save images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        f1_ = os.path.join(self.folder, f'pred_{self.image_id}.png')
        if self.command['save_image']:
            #image = self.pibot.get_image()
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(f_, image)
            #self.image_id += 1
            #self.pred_fname = self.output.write_image(image,self.ekf)
            self.command['save_image'] = False
            #self.notification = f'{f_} is saved'

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

    
    def read_fruit_data(self):
        if self.command['read_inference']:
            target_map = self.estimate_pose()
            target_est = self.merge_estimations(target_map)
            base_dir = Path('./')           
            # save target pose estimations
            with open(base_dir/'lab_output/targets.txt', 'w') as fo:
                json.dump(target_est, fo)
            self.notification = 'Fruit Locations Saved'
            self.command['read_inference'] = False
    
    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map of ARUCO markers is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            image = cv2.cvtColor(self.pibot.get_image(), cv2.COLOR_RGB2BGR)
            self.pred_fname = self.output.write_image(image,self.ekf)
            self.notification = f'Prediction is saved to {operate.pred_fname}'
            self.command['save_inference'] = False
            #Writing the bounding boxes and pose to a file

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


            """
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # AFR
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                self.command['read_inference'] = True
            """

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