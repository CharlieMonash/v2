# estimate the pose of a target object detected
from pyexpat.errors import XML_ERROR_ABORTED
import numpy as np
import json
import os
from pathlib import Path
import ast
# import cv2
import math
from machinevisiontoolbox import Image
from network.scripts.detector import Detector
import matplotlib.pyplot as plt
import PIL

#Defining Yolo place the model weight filepath into the weights_path variable
weights_path =  'network/scripts/model/best.pt'
Yolo = Detector(weights_path, use_gpu=False)
# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
def get_bounding_box(target_number, image_path):
    image = PIL.Image.open(image_path).resize((640,480), PIL.Image.NEAREST)
    target = Image(image)==target_number
    """
    blobs = target.blobs()
    [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
    width = abs(u1-u2)
    height = abs(v1-v2)
    center = np.array(blobs[0].centroid).reshape(2,)
    box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]
    """
    # plt.imshow(fruit.image)
    # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
    # plt.show()
    # assert len(blobs) == 1, "An image should contain only one object of each target type"
    #Pssing the image through the network
    _,_,yolo_results,_= Yolo.detect_single_image(image)
    #Creating the box from the yellow results
    #print(yolo_results)
    shapes = yolo_results.shape
    """
    for i in range(shapes[0])
        width = abs(yolo_results[i][1]-yolo_results[i][2])
        height = abs(yolo_results[i][3]-yolo_results[i][4])
        center = np.array([(yolo_results[i][1]+yolo_results[i][2])/2,(yolo_results[i][3]+yolo_results[i][4])/2])
        box = [center[0], center[1], int(width), int(height)]
    """
    box = np.zeros((yolo_results.shape[0],4))
    for i in range(yolo_results.shape[0]):
        width = abs(yolo_results[i][1]-yolo_results[i][2])
        height = abs(yolo_results[i][3]-yolo_results[i][4])
        centre_0 = (yolo_results[i][1]+yolo_results[i][2])/2
        centre_1 = (yolo_results[i][3]+yolo_results[i][4])/2
        box[i] = [centre_0, centre_1, int(width), int(height)]
    return box

# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(base_dir, file_path, image_poses):
    # there are at most five types of targets in each image
    target_lst_box = [[], [], [], [], [],[]]
    target_lst_pose = [[], [], [], [], [],[]]
    completed_img_dict = {}

    # add the bounding box info of each target in each image
    # target labels: 1 = apple, 2 = lemon, 3 = pear, 4 = orange, 5 = strawberry, 0 = not_a_target
    text_file_path = file_path.split('.')[0]+'.txt'
    
    img_vals = set(Image(base_dir / file_path, grey=True).image.reshape(-1))
    #for target_num in img_vals:
    box = get_bounding_box(0, base_dir/file_path) # [x,y,width,height]

    for target_num in range(box.shape[0]):
            box_1D = box[target_num]
            pose = image_poses[file_path] # [x, y, theta]
            target_lst_box[target_num].append(box_1D) # bounding box of target
            target_lst_pose[target_num].append(np.array(pose).reshape(3,)) # robot pose

     #reading json data
    """
    with open(text_file_path,"r") as f:
        data = json.load(f)
        for fruit in data:
            xmin = fruit['xmin']
            ymin = fruit['ymin']
            xmax = fruit['xmax']
            ymax = fruit['ymax']
            x = (xmin + xmax)/2
            y = (ymin + ymax)/2
            width = xmax - xmin
            height = ymax - ymin

            box = [x, y, width, height]
            pose = image_poses[file_path] #[x, y, theta]
            target_num = fruit['class']
            target_lst_box[target_num].append(box)
            target_lst_pose[target_num].append(np.array(pose).reshape(3,)) # robot pose
        """
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

    target_list = ['apple', 'lemon', 'pear', 'orange', 'strawberry']

    target_pose_dict = {}
    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
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

        y_object = A * np.cos(theta_total)
        x_object = A * np.cos(theta_total)
        
        x_object_world = x_robot + x_object
        y_object_world = y_robot + y_object

        target_pose = {'y':y_object_world,'x':x_object_world}
        target_pose_dict[target_list[target_num-1]] = target_pose
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
        #merge two points by averaging
        print("")
        print(fruit_est[min1][1])
        print(fruit_est[min2][1])
        print("")
        print(fruit_est[min1][0])
        print(fruit_est[min2][0])

        x_avg = (fruit_est[min1][1] + fruit_est[min2][1])/2 #averaging x
        y_avg = (fruit_est[min1][0] + fruit_est[min2][0])/2 #averaging y
        fruit_est = np.delete(fruit_est,(min1, min2), axis=0)
        fruit_est = np.vstack((fruit_est, [y_avg,x_avg]))
    return fruit_est


# merge the estimations of the targets so that there are at most 3 estimations of each target type
def merge_estimations(target_pose_dict):
    target_pose_dict = target_pose_dict
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
    if len(apple_est) > 2:
        apple_est = mean_fruit(apple_est)
    if len(lemon_est) > 2:
        lemon_est = mean_fruit(lemon_est)
    if len(pear_est) > 2:
        pear_est = mean_fruit(pear_est)
    if len(orange_est) > 2:
        orange_est = mean_fruit(orange_est)
    if len(strawberry_est) > 2:
        strawberry_est = mean_fruit(strawberry_est)

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