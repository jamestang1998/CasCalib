import sys
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_directory)
import util
import os
import data
import json

from datetime import datetime

import csv
import pickle

import run_calibration_ransac
import matplotlib.image as mpimg

today = datetime.now()
import numpy as np
import math
import plotting
import synchronization

import bundle_adjustment
import bundle_adjustment_multi
import torch
import knn
import ICP
import geometry
import matplotlib.pyplot as plt
from xml.dom import minidom
from scipy.spatial.transform import Rotation

import eval_functions
import torch
import spline_model

from eval_human_pose import Metrics
metrics = Metrics()
import csv
import plane_peturb
import h5py
import pandas as pd
import bundle_adjustment_plotly
import itertools
import copy
import time_align
import random

#The name of is the current date
name = str(today.strftime('%Y%m%d_%H%M%S'))

#Gets the hyperparamter from hyperparameter.json
threshold_euc, threshold_cos, angle_filter_video, confidence, termination_cond, num_points, h, iter, focal_lr, point_lr = util.hyperparameter('/local/tangytob/Summer2023/multiview_synchronization/hyperparameter_fei.json')

hyperparam_dict = {"threshold_euc": threshold_euc, "threshold_cos": threshold_cos, "angle_filter_video": angle_filter_video, "confidence": confidence, "termination_cond": termination_cond, "num_points": num_points, "h": h, "optimizer_iteration" :iter, "focal_lr" :focal_lr, "point_lr": point_lr}

#Making the directories, eval is the accuracy wit hthe ground truth, output is the calibration saved as a pickle file, plot is the plots that are created during optimization.

if os.path.isdir('./plots') == False:
    os.mkdir('./plots')

plot_scale = 1
line_amount = 50

ankles_array = []

factor = 10
start = 0
end = 0#factor
index_array = []
skip = 1

camera_list = [
'54138969',
'55011271',
'58860488',
'60457274'
]

scene_num = 0
#EXTRINSIC MATRIX DETAILS
#https://ksimek.github.io/2012/08/22/extrinsic/
#for subject in ['S1', 'S5','S6','S7','S8','S9']:
#for subject in ['S5','S6','S7','S8','S9']:
#for subject in ['S1','S5','S6','S7','S8']:

#REF    SYNC   REF          SYNC         REF        SYNC
#scale, scale, shift start, shift start, shift end, shift end
#experiments_time = [experiments_time[0]]

#experiments_time = [100, 200, 400, 600, 800, 1000]
experiments_time = [1000]
image_scale = 1.0
cam_comb = util.random_combination([0,1,2,3], 2, np.inf)

print(cam_comb)

#for subject in ['S9', 'S11']:

if os.path.isdir('./plots/single_' + name) == False:
    os.mkdir('./plots/single_' + name)

subject_array = ["S1","S5", "S6", "S7", "S8", "S9", "S11"]
#subject_array = ["S9", "S11"]
#subject_array = ["S11"]
    
for subject in subject_array:
    for scene in ['Walking']:

        if subject == "S7":
            scene = 'Walking 1'
        focal_array = []
        name_folder = '/' + str(subject) + '_' + str(scene)

        save_dir = './plots/single_' + name + name_folder

        if os.path.isdir('./plots/single_' + name + name_folder) == False:
            os.mkdir('./plots/single_' + name + name_folder)

        if os.path.isdir('./plots/single_' + name + name_folder + '/calibration_init') == False:
            os.mkdir('./plots/single_' + name + name_folder + '/calibration_init')

        if os.path.isdir('./plots/single_' + name + name_folder + '/calibration_equal') == False:
            os.mkdir('./plots/single_' + name + name_folder + '/calibration_equal')
        '''
        with open('./plots/single_' + name + name_folder + '/calibration_init' + '/calibration_init.csv','a') as file:

            writer1 = csv.writer(file)
            writer1.writerow(["subject", "camera", "focal", "normal", "ankle", "focal error"])
            file.close


        with open('./plots/single_' + name + name_folder + '/calibration_equal' + '/calibration_equal.csv','a') as file:

            writer1 = csv.writer(file)
            writer1.writerow(["subject", "camera", "focal", "normal", "ankle", "focal error"])
            file.close
        '''
        
        detection_path_array = []
        frame_dir_array = []
        single_view_array = []
        data_array = []

        #hdf5_path = '/local2/tangytob/h36m/processed/processed/' + subject + '/' + scene + '/annot.h5'
        '''
        h36m_3d = None
        h36m_2d = None

        h36m_3d_camera = None
        with h5py.File(hdf5_path, "r") as f:
            # Print time root level object names (aka keys) 
            # these can be group or dataset names 
            #print(f['action'], " HELLOOOASDASDASD")
            #print("Keys: %s" % f.keys())
            #print(f.value)
            # get first object name/key; may or may NOT be a group
            a_group_key = list(f.keys())[0]

            # get the object type for a_group_key: usutimey group or dataset
            #print(type(f[a_group_key])) 

            # If a_group_key is a group name, 
            # this gets the object names in the group and returns as a list
            hdf5_data = list(f[a_group_key])

            # If a_group_key is a dataset name, 
            # this gets the dataset values and returns as a list
            hdf5_data = list(f[a_group_key])
            # preferred methods to get dataset values:
            ds_obj = f[a_group_key]      # returns as a h5py dataset object
            ds_arr = f[a_group_key][()]  # returns as a numpy array

            h36m_3d = f['pose']['3d'][:]
            h36m_2d = f['pose']['2d'][:]

            h36m_3d_camera = f['camera'][:]
        '''
        #########################

        output_folder = subject + '_' + scene
        #PLOTS SAVED IN PLOTS NOW!!!
        
        if os.path.isdir('./plots/single_' + name + name_folder + '/' + output_folder) == False:
            os.mkdir('./plots/single_' + name + name_folder + '/' + output_folder)
        
        fl = output_folder + '-c0_avi'

        frame_array = []
        output_path = './plots/single_' + name + name_folder + '/' + output_folder + '/'
        
        scale = 1
        scale_tsai = 1

        grid_step = 10

        average_angle_array = []
        dist_error = []

        ######################################################################################
        scene_list = [output_folder]

        detections_array = []

        #######################################################################################

        x_2d = np.linspace(-5, 5, grid_step)
        y_2d = np.linspace(0, 10, grid_step)
        xv_2d, yv_2d = np.meshgrid(x_2d, y_2d)
        coords_xy_2d =np.array((xv_2d.ravel(), yv_2d.ravel())).T

        cam_intrinsics_array = []

        #############
        plane_points_3d = []

        file_num = 0

        pose_track = 1

        plane_points_time = []
        time_points_time = []

        print(len(plane_points_time))
        #stop
        #############
        cam_axis = []
        cam_position = []

        cam_axis0 = []
        cam_position0 = []

        cam_axis1 = []
        cam_position1 = []

        cam_axis2 = []
        cam_position2 = []

        cam_axis3 = []
        cam_position3 = []

        cam_axis4 = []
        cam_position4 = []


        pos0 = np.array([0,0,0])#plane_matrix_array0[:3,3]
        axis0 = np.array([[1,0,0],[0,1,0],[0,0,1]])#plane_matrix_array0[:3,:3]

        time_points_array4 = []
        time_track = []

        frame_dict_array = []
        center_array = []

        time_cam_intrinsics = []

        pre_bundle_cal_array = []
        single_view_cal_array = []

        with open('/local/tangytob/Summer2023/multiview_synchronization/camera-parameters.json', 'r') as f:
            h36m_json = json.load(f)

        gt_rotation_array = []
        gt_translation_array = []

        distortion_k_array = []
        distortion_p_array = []
        
        points_h36m_3d = []

        points_h36m_3d1 = []
        points_h36m_3d2 = []
        points_h36m_center = []

        plane_matrix_array = []
        plane_dict_array = []
        #The rotation matrix is the camera orientation, but the translation is the extrinsic translation probably
        save_dict_array = []
        points_2d_array = []
        ankle_head_2d_array = []

        pose_2d_array = []

        gt_intrinsics_array = []

        #############################
        # FOCAL INITALIZATION
        
        for k1 in list(h36m_json['extrinsics'][subject].keys()):

            with open('/local/tangytob/Summer2023/multiview_synchronization/configuration.json', 'r') as f:
                configuration = json.load(f)

            with open('/local2/tangytob/h36m_hrnet_walking/' + subject + '_Walking_' + str(k1) + '/result_Videos_' + scene + '_.json', 'r') as f:
                points_2d = json.load(f)


            datastore_cal = data.coco_mmpose_dataloader(points_2d)        
            
            frame_dir = '/local2/tangytob/CPSC533R/human3_6/frames/' + subject + '/' + scene + '.' + str(k1) + '.mp4/frame0.jpg'
            img = mpimg.imread(frame_dir)
            
            ankles, cam_matrix, normal, ankleWorld, focal_pred, focal_batch, ransac_focal, datastore_filtered = run_calibration_ransac.run_calibration_ransac(datastore_cal, '/local/tangytob/Summer2023/multiview_synchronization/hyperparameter_fei.json', img, img.shape[1], img.shape[0], str(k1) + '_', name, skip_frame = configuration['skip_frame'], max_len = configuration['max_len'], min_size = configuration['min_size'])
            
            focal_array.append(ransac_focal)
            save_dict = {"cam_matrix":cam_matrix.tolist(), "ground_normal":normal.tolist(), "ground_position":ankleWorld.tolist()}

            with open('./plots/single_' + name + name_folder + '/calibration_init' + '/calibration_init_' + str(k1) + '_.json','w') as file:
                json.dump(save_dict, file)
                file.close


            
        
        #focal_init = 1145.0#np.median(focal_array)
        focal_init = np.median(focal_array)
        #focal_init = None
        print(list(h36m_json['extrinsics'][subject].keys()), " THE KEYSSSSSS")
        #stop
        for k1 in list(h36m_json['extrinsics'][subject].keys()):
            print(k1, " THIS IS THE KEY")
            distortion_k_array.append(np.array([h36m_json['intrinsics'][k1]["distortion"][0], h36m_json['intrinsics'][k1]["distortion"][1], h36m_json['intrinsics'][k1]["distortion"][4]]))
            distortion_p_array.append(np.array([h36m_json['intrinsics'][k1]["distortion"][2], h36m_json['intrinsics'][k1]["distortion"][3]]))

            gt_rotation = h36m_json['extrinsics'][subject][k1]['R']
            gt_translation = list(np.array(h36m_json['extrinsics'][subject][k1]['t'])/1000)

            gt_intrinsics_array.append(h36m_json['intrinsics'][k1]["calibration_matrix"])

            c_rotation, c_translation = util.extrinsics_to_camera_pose(gt_rotation, gt_translation)
            
            gt_rotation_array.append(gt_rotation)
            gt_translation_array.append(np.array([c_translation[0][0], c_translation[1][0], c_translation[2][0]]))

            with open('/local/tangytob/Summer2023/camera_calibration_synchronization/configuration.json', 'r') as f:
                configuration = json.load(f)

            with open('/local2/tangytob/h36m_hrnet_walking/' + subject + '_Walking_' + str(k1) + '/result_Videos_' + scene + '_.json', 'r') as f:
                points_2d = json.load(f)

            print(len(points_2d["Info"]), " LENGTH OF THE ARRAY")
            datastore = data.coco_mmpose_dataloader(points_2d, scale_x = image_scale, scale_y = image_scale)

            print(datastore.__len__(), " THE DATASTORE")
            #print(datastore.getData(), " THE DATASTORE")

            #stop
            #datastore_cal = data.h36m_gt_dataloader(points_2d)        
            
            frame_dir = '/local2/tangytob/CPSC533R/human3_6/frames/' + subject + '/' + scene + '.' + str(k1) + '.mp4/frame0.jpg'
            img = mpimg.imread(frame_dir)

            ankles, cam_matrix, normal, ankleWorld, focal_pred, focal_batch, ransac_focal, datastore_filtered = run_calibration_ransac.run_calibration_ransac(datastore, '/local/tangytob/Summer2023/multiview_synchronization/hyperparameter_fei.json', img, img.shape[1], img.shape[0], str(k1) + '_', name, skip_frame = configuration['skip_frame'], max_len = configuration['max_len'], min_size = configuration['min_size'], f_init = focal_init)

            save_dict = {"cam_matrix":cam_matrix.tolist(), "ground_normal":normal.tolist(), "ground_position":ankleWorld.tolist()}

            with open('./plots/single_' + name + name_folder + '/calibration_equal' + '/calibration_equal_'  + str(k1) + '_.json','w') as file:
                json.dump(save_dict, file)
                file.close