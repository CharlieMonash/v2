# estimate the pose of a target object detected
from pyexpat.errors import XML_ERROR_ABORTED
import numpy as np
import json
#import os
from pathlib import Path
import ast
# import cv2
#import math
#from machinevisiontoolbox import Image
#from network.scripts.detector import Detector
#import matplotlib.pyplot as plt
import PIL

# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(base_dir, file_path, image_poses):
    # there are at most five types of targets in each image
    target_lst_box = [[], [], [], [], [],[]]
    target_lst_pose = [[], [], [], [], [],[]]
    completed_img_dict = {}

    # add the bounding box info of each target in each image
    # target labels: 1 = apple, 2 = lemon, 3 = pear, 4 = orange, 5 = strawberry, 0 = not_a_target
    text_file_path = file_path.split('.')[0]+'.txt'

     #reading json data
    with open(text_file_path,"r") as f:
        data = json.load(f)
        for fruit in data:
            xmin = fruit['xmin']*2
            ymin = fruit['ymin']*2
            xmax = fruit['xmax']*2
            ymax = fruit['ymax']*2
            x = (xmin + xmax)/2
            y = (ymin + ymax)/2
            width = xmax - xmin
            height = ymax - ymin

            box = [x, y, width, height]
            pose = image_poses[file_path] #[x, y, theta]
            target_num = fruit['class']
            target_lst_box[target_num].append(box)
            target_lst_pose[target_num].append(np.array(pose).reshape(3,)) # robot pose
    
    # if there are more than one objects of the same type, combine them
    for i in range(5):
        if len(target_lst_box[i])>0:
            box = np.stack(target_lst_box[i], axis=1)
            pose = np.stack(target_lst_pose[i], axis=1)
            completed_img_dict[i+1] = {'target': box, 'robot': pose}
    return completed_img_dict

# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
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

    target_list = ['apple', 'lemon', 'orange', 'pear', 'strawberry']

    target_pose_dict = {}
    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
        for i in range(len(completed_img_dict[target_num]['target'][0])):
            box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
            robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
            true_height = target_dimensions[target_num-1][2]
            
            ######### Replace with your codes #########
            # TODO: compute pose of the target based on bounding box info and robot's pose
            target_pose = {'y': 0.0, 'x': 0.0}
            
            cam_res = 640 # camera resolution in pixels

            A = focal_length * true_height / box[3][0] # actual depth of object
    
            x_robot = robot_pose[0][0]
            y_robot = robot_pose[1][0]
            theta_robot = robot_pose[2][0]

            x_camera = cam_res/2 - box[0][0]
            theta_camera = np.arctan(x_camera/focal_length)
            theta_total = theta_robot + theta_camera

            y_object = A * np.sin(theta_total)
            x_object = A * np.cos(theta_total)
            
            x_object_world = x_robot + x_object
            y_object_world = y_robot + y_object

            target_pose = {'y':y_object_world,'x':x_object_world}
            target_pose_dict[f'{target_list[target_num-1]}_{i}'] = target_pose
            ###########################################
        
    return target_pose_dict

# EXTRA: to changes
def mean_fruit(fruit_est):
    while len(fruit_est) > 2:
        min_dist = 9999
        #find two points close to each other
        for i, fruit1 in enumerate(fruit_est):
            for j, fruit2 in enumerate(fruit_est):
                if (fruit1[0] != fruit2[0]) or (fruit1[1] != fruit2[1]): #if not same fruit
                    distance = np.sqrt((fruit1[1]-fruit2[1])**2+(fruit1[0]-fruit2[0])**2)
                    if distance < min_dist:
                        min_dist = distance
                        min1 = i
                        min2 = j

        x_avg = (fruit_est[min1][1] + fruit_est[min2][1])/2 #averaging x
        y_avg = (fruit_est[min1][0] + fruit_est[min2][0])/2 #averaging y
        fruit_est = np.delete(fruit_est,(min1, min2), axis=0)
        fruit_est = np.vstack((fruit_est, [y_avg,x_avg]))
    return fruit_est

def merge_to_mean(position_est, remove_outlier = False):
    # Set up working parameters
    position_est = np.array(position_est)

    position_est_result = []
    z_threshold = 3

    # Compute mean and standard deviations
    means = np.mean(position_est, axis = 0)
    #print(means)
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

def sort_locations_and_merge(position_est, distance_threshold = 0.3, remove_outlier = False, use_Kmeans = False):
    position_est1 = []
    position_est2 = []
    position_est = np.array(position_est)

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
    print(position_est2)
    position1 = merge_to_mean(position_est1, remove_outlier)
    position2 = merge_to_mean(position_est2, remove_outlier)
    # return the position estimations
    positions = []
    if(position1 is not None):
        positions.append(position1)
    if(position2 is not None):
        positions.append(position2)
    return positions

def read_search_list():
    search_list_1 = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()
        for fruit in fruits:
            search_list_1.append(fruit.strip())

    return search_list_1

# merge the estimations of the targets so that there are at most 3 estimations of each target type
def merge_estimations(target_pose_dict):
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
                #Charlie - somehow doubling up on values in above part
                # NEED TO FIX

    ######### Replace with your codes #########
    # TODO: the operation below takes the first three estimations of each target type, replace it with a better merge solution
    remove_outlier = False
    use_Kmeans = False
    search_list = read_search_list()

    if 'apple' in search_list:
        apple_est = np.array([np.mean(apple_est, axis=0)])
    else:
        if len(apple_est) > 2:
            apple_est = self.sort_locations_and_merge(apple_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)
    
    if 'lemon' in search_list:
        lemon_est = np.array([np.mean(lemon_est, axis=0)])
    else:
        if len(lemon_est) > 2:
            lemon_est = sort_locations_and_merge(lemon_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)

    if 'pear' in search_list:
        pear_est = np.array([np.mean(pear_est, axis=0)])
    else:
        if len(pear_est) > 2:
            pear_est = sort_locations_and_merge(pear_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)

    if 'orange' in search_list:
        orange_est = np.array([np.mean(orange_est, axis=0)])
    else:
        if len(orange_est) > 2:
            orange_est = sort_locations_and_merge(orange_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)

    if 'strawberry' in search_list:
        strawberry_est = np.array([np.mean(strawberry_est, axis=0)])
    else:
        if len(strawberry_est) > 2:
            print(strawberry_est)
            #strawberry_est = sort_locations_and_merge(strawberry_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)
            #NEED TO FIX

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

def read_slam_map(self,fname):
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        slam_pos = np.zeros([10,2])
        i=0
        # remove unique id of targets of the same type
        for key in gt_dict["taglist"]:
            slam_pos[key-1,0] = np.round(gt_dict["map"][0][i], 1)
            slam_pos[key-1,1] = np.round(gt_dict["map"][1][i], 1)
            i+=1
            #marker_id = int(gt_dict["taglist"][key])
            #aruco_true_pos[marker_id-1][0] = x
            #aruco_true_pos[marker_id-1][1] = y
        #Charlie - doesnt work properly if doesnt have all ten markers!
        return slam_pos

if __name__ == "__main__":
    # camera_matrix = np.ones((3,3))/2
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')
    # a dictionary of all the saved detector outputs
    image_poses = {}
    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']
    # estimate pose of targets in each detector output
    target_map = {}        
    for file_path in image_poses.keys():
        completed_img_dict = get_image_info(base_dir, file_path, image_poses)
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)
    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    target_est = merge_estimations(target_map)
    # save target pose estimations
    with open(base_dir/'lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo)
    print('Estimations saved!')