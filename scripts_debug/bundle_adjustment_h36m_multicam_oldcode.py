import sys
import os
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

import torch
import knn
import ICP
import geometry
import matplotlib.pyplot as plt
from xml.dom import minidom
from scipy.spatial.transform import Rotation

import eval_functions
import torch

from eval_human_pose import Metrics
metrics = Metrics()
import csv
import pandas as pd
import time_align_inlier
import bundle_adjustment_plotly_fix

#The name of is the current date
name = str(today.strftime('%Y%m%d_%H%M%S')) +'_h36m'

#Gets the hyperparamter from hyperparameter.json
threshold_euc, threshold_cos, angle_filter_video, confidence, termination_cond, num_points, h, iter, focal_lr, point_lr = util.hyperparameter('../hyperparameters/hyperparameter_h36m_multicam.json')

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
'''
camera_list = [
'54138969',
'55011271'
]
'''
scene_num = 0
#EXTRINSIC MATRIX DETAILS
#https://ksimek.github.io/2012/08/22/extrinsic/
#for subject in ['S1', '','','','','']:
#for subject in ['','','','','']:
#for subject in ['S1','','','','']:

#REF    SYNC   REF          SYNC         REF        SYNC
#scale, scale, shift start, shift start, shift end, shift end
#experiments_time = [experiments_time[0]]

#experiments_time = [100, 200, 400, 600, 800, 1000]
experiments_time = [0, 100, 150, 200]

image_scale = 1.0
#cam_comb = util.random_combination([0,1,2,3], 2, np.inf)
#cam_comb = [(0,1), (1,3), (3,2), (2,0)]
cam_comb = [0, 1]
cam_comb = [0, 1,2,3]

print(cam_comb)
'''
time_pred = {
    'S1': {
        0: [9, 0, -2],
        50: [59, 59, 48],
        100: [109, 109, 98],
        150: [160, 159, 148],
        200: [209, 209, 198]
    },
    'S11': {
        0: [12, 9, -4],
        50: [62, 59, 46],
        100: [112, 109, 95],
        150: [162, 159, 146],
        200: [10, 10, 196]
    },
    'S5': {
        0: [19, 14, 0],
        50: [69, 64, 50],
        100: [119, 113, 100],
        150: [169, 163, 150],
        200: [220, 214, 201]
    },
    'S6': {
        0: [22, 11, 29],
        50: [72, 61, 79],
        100: [122, 111, 129],
        150: [173, 162, 180],
        200: [228, 213, 231]
    },
    'S7': {
        0: [14, 9, -1],
        50: [64, 59, 49],
        100: [114, 109, 99],
        150: [164, 162, 149],
        200: [214, 212, 199]
    },
    'S8': {
        0: [18, 19, 0],
        50: [68, 70, 50],
        100: [118, 120, 100],
        150: [163, 169, 150],
        200: [219, 220, 200]
    },
    'S9': {
        0: [12, 6, -5],
        50: [62, 56, 46],
        100: [112, 106, 95],
        150: [163, 156, 145],
        200: [221, 215, 205]
    },
    'S7': {
        0: [14, 9, -1],
        50: [64, 59, 49],
        100: [114, 109, 99],
        150: [164, 162, 149],
        200: [214, 212, 199]
    },
    'S8': {
        0: [18, 19, 0],
        50: [68, 70, 50],
        100: [118, 120, 100],
        150: [163, 169, 150],
        200: [219, 220, 200]
    },
    'S9': {
        0: [12, 6, -5],
        50: [62, 56, 46],
        100: [112, 106, 95],
        150: [163, 156, 145],
        200: [221, 215, 205]
    }
}
'''


time_pred = {'S1': {0: [11, 12, -2], 50: [62, 62, 48], 100: [112, 112, 98], 150: [162, 162, 148], 200: [212, 212, 198]}, 'S5': {0: [19, 14, 0], 50: [68, 64, 50], 100: [119, 113, 100], 150: [168, 163, 150], 200: [217, 213, 200]}, 'S6': {0: [30, 11, 31], 50: [80, 61, 81], 100: [130, 111, 133], 150: [194, 163, 183], 200: [244, 213, 233]}, 'S7': {0: [15, 9, -1], 50: [65, 59, 49], 100: [115, 109, 98], 150: [165, 161, 147], 200: [215, 211, 197]}, 'S8': {0: [17, 21, 1], 50: [67, 71, 51], 100: [113, 121, 101], 150: [168, 170, 151], 200: [218, 220, 201]}, 'S9': {0: [12, 4, -4], 50: [62, 54, 46], 100: [112, 105, 96], 150: [163, 156, 146], 200: [215, 207, 197]}, 'S11': {0: [13, 10, -5], 50: [63, 60, 46], 100: [114, 110, 95], 150: [163, 160, 146], 200: [12, 11, 196]}}

#for subject in ['', 'S11']:

if os.path.isdir('./plots/time_' + name) == False:
    os.mkdir('./plots/time_' + name)

subject_array = ["S1","S5", "S6", "S7", "S8", "S9", "S11"]

#subject_array = ["S5"]
#experiments_time = [100]
#subject_array = ["S7", "S8", "S9", "S11"]
#subject_array = [""]
#subject_array = ["", "S11"]
#subject_array = ["S11"]
#test_list = [0, 100, 200, 300]

#test_list = [300]


with open('./plots/time_' + name + '/result_bundle_sync.csv','a') as file:

    writer1 = csv.writer(file)
    writer1.writerow(["offset", "exp", "angle_diff pre bundle", "focal_error pre bundle", "results_position_diff pre bundle", "angle_diff bundle", "focal_error bundle", "results_position_diff bundle"])
    file.close

with open('./plots/time_' + name + '/result_bundle_no_sync.csv','a') as file:

    writer1 = csv.writer(file)
    writer1.writerow(["offset", "exp", "angle_diff pre bundle", "focal_error pre bundle", "results_position_diff pre bundle", "angle_diff bundle", "focal_error bundle", "results_position_diff bundle"])
    file.close

for subject in subject_array:
    for scene in ['Walking']:

        if subject == "S7":
            scene = 'Walking 1'
        focal_array = []
        
        
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
        
        scale = 1
        scale_tsai = 1

        grid_step = 10

        average_angle_array = []
        dist_error = []

        ######################################################################################

        detections_array = []

        #######################################################################################

        x_2d = np.linspace(-5, 5, grid_step)
        y_2d = np.linspace(0, 10, grid_step)
        xv_2d, yv_2d = np.meshgrid(x_2d, y_2d)
        coords_xy_2d =np.array((xv_2d.ravel(), yv_2d.ravel())).T

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

        pos0 = np.array([0,0,0])#plane_matrix_array0[:3,3]
        axis0 = np.array([[1,0,0],[0,1,0],[0,0,1]])#plane_matrix_array0[:3,:3]

        time_points_array4 = []
        time_track = []

        frame_dict_array = []
        center_array = []

        time_cam_intrinsics = []


        with open('../jsons/camera-parameters.json', 'r') as f:
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
        '''
        for k1 in list(h36m_json['extrinsics'][subject].keys()):
            continue
            ind_camera = [i for i, n in enumerate(h36m_3d_camera) if int(n) == int(k1)]

            keypoint_index = [
                        0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27 
            ]
            #points_2d = np.array(h36m_2d)[:, [3, 8, 24], :][ind_camera, :, :]
            points_2d = np.array(h36m_2d)[:, keypoint_index, :][ind_camera, :, :]
            print(points_2d.shape, " POINTS IN H36M 2d")
            points_2d_array.append(points_2d)
            #####################

            with open('/local/tangytob/Summer2023/camera_calibration_synchronization/configuration.json', 'r') as f:
                configuration = json.load(f)


            datastore_cal = data.h36m_gt_dataloader(points_2d)        
            
            frame_dir = '/local2/tangytob/h36m/processed/processed/' + subject + '/' + scene + '/imageSequence/' + k1 + '/img_000001.jpg'
            img = mpimg.imread(frame_dir)
            
            ankles, cam_matrix, normal, ankleWorld, ransac_focal, datastore_filtered = run_calibration_ransac.run_calibration_ransac(datastore_cal, '/local/tangytob/Summer2023/camera_calibration_synchronization/scripts_multicam/hyperparameter_h36m.json', img, img.shape[1], img.shape[0], str(k1) + '_', name, skip_frame = configuration['skip_frame'], max_len = configuration['max_len'], min_size = configuration['min_size'])
            
            focal_array.append(ransac_focal)
        '''    
        
        #focal_init = 1145.0#np.median(focal_array)
        #focal_init = np.median(focal_array)
        #focal_init = None
        print(list(h36m_json['extrinsics'][subject].keys()), " THE KEYSSSSSS")
        #stop
        #for k1 in list(h36m_json['extrinsics'][subject].keys()):
        for k1 in camera_list:
            #print(k1, " THIS IS THE KEY")
            #stop
            distortion_k_array.append(np.array([h36m_json['intrinsics'][k1]["distortion"][0], h36m_json['intrinsics'][k1]["distortion"][1], h36m_json['intrinsics'][k1]["distortion"][4]]))
            distortion_p_array.append(np.array([h36m_json['intrinsics'][k1]["distortion"][2], h36m_json['intrinsics'][k1]["distortion"][3]]))

            gt_rotation = h36m_json['extrinsics'][subject][k1]['R']
            gt_translation = list(np.array(h36m_json['extrinsics'][subject][k1]['t'])/1000)

            c_rotation, c_translation = util.extrinsics_to_camera_pose(gt_rotation, gt_translation)
            
            gt_intrinsics_array.append(np.array(h36m_json['intrinsics'][k1]["calibration_matrix"]))
            gt_rotation_array.append(gt_rotation)
            gt_translation_array.append(np.array([c_translation[0][0], c_translation[1][0], c_translation[2][0]]))

            with open('../jsons/configuration.json', 'r') as f:
                configuration = json.load(f)

            with open('/local2/tangytob/h36m_hrnet_walking/' + subject + '_Walking_' + str(k1) + '/result_Videos_' + scene + '_.json', 'r') as f:
                points_2d = json.load(f)

            print(len(points_2d["Info"]), " LENGTH OF THE ARRAY")
            datastore = data.coco_mmpose_dataloader(points_2d, scale_x = image_scale, scale_y = image_scale)

            print(datastore.__len__(), " THE DATASTORE")
            #print(datastore.getData(), " THE DATASTORE")
            '''
            for ps in list(datastore.getData().keys()):
                #print(ps, datastore.getData()[ps])
                for pos in datastore.getData()[ps]:
                    print(ps, pos['id'])
            '''
            #stop
            #datastore_cal = data.h36m_gt_dataloader(points_2d)        
            
            frame_dir = '/local2/tangytob/CPSC533R/human3_6/frames/' + subject + '/' + scene + '.' + str(k1) + '.mp4/frame0.jpg'
            img = mpimg.imread(frame_dir)

            with open('../calibration/h36m_calibration_20220620_160757/' + subject + '_' + k1 + '/calibration.pickle', 'rb') as f:
                pickle_file = pickle.load(f)
            #print(pickle_file)
            #print(pickle_file["ankles"].shape, " HSPAE")
            cam_matrix = np.array(pickle_file["cam_matrix"])#np.array(h36m_json['intrinsics'][k1]["calibration_matrix"])
            normal = np.array(pickle_file["normal"])#np.array(gt_rotation) @ np.array([0,0,1])
            ankleWorld = np.array(pickle_file["ankles"][:, 0])#np.array(gt_translation)
            #datastore_filtered = loaded_list['datastore_filtered']
            
            save_dict = {"cam_matrix":cam_matrix, "ground_normal":normal, "ground_position":ankleWorld}
            ##################################
            #datastore = data.h36m_gt_dataloader(points_2d) 
            #print(datastore.getData())
            #stop
            data_2d = util.get_ankles_heads_dictionary(datastore, cond_tol = confidence)
            pose_2d = util.get_ankles_heads_pose_dictionary(datastore, cond_tol = confidence)
            print(len(pose_2d), len(data_2d), " THIS IS POSE 2D")

            ankle_head_2d_array.append(data_2d)
            pose_2d_array.append(pose_2d)
            
            plane_matrix, basis_matrix = geometry.find_plane_matrix(save_dict["ground_normal"], np.linalg.inv(save_dict['cam_matrix']),save_dict['ground_position'], img.shape[1], img.shape[0])
            
            to_pickle_plane_matrix = {"plane_matrix": plane_matrix,'intrinsics': save_dict['cam_matrix']}

            plane_data_2d = geometry.camera_to_plane(data_2d, cam_matrix, plane_matrix, save_dict['ground_position'], save_dict["ground_normal"], img.shape[1], img.shape[0])

            plane_matrix_array.append(plane_matrix)
            #print(plane_matrix, " PLANE MATRIX!!!")

            plane_dict_array.append(plane_data_2d)
            ##################################
            save_dict_array.append(save_dict)

            #points_h36m_center.append(np.expand_dims(np.array(points_center), axis = 0))
        #stop
        plane_dict_array_experiment0 = {}

        #REF    SYNC   REF          SYNC         REF        SYNC
        #scale, scale, shift start, shift start, shift end, shift end
        #for k in range(len(experiments_time)):
        #print("***************************")
        #print(plane_dict_array[0])
        plane_dict_array_experiment = []

        
        #for cam in [(1,2)]:#cam_comb:
        #for cam in cam_comb:
        
        camera_name = list(h36m_json['extrinsics'][subject].keys())
        '''
        for cam in [0,1,2,3]:
            x_plot = []
            y_plot = []
            frame_plot = []
            plane_dict_array0 = plane_dict_array[cam].copy()
            
            for f_k in list(plane_dict_array0.keys()):
                #print(list(plane_dict_array0[f_k].values())[0][0], list(plane_dict_array0[f_k].values())[0][1])
                x_plot.append(list(plane_dict_array0[f_k].values())[0][0])
                y_plot.append(list(plane_dict_array0[f_k].values())[0][1])
                frame_plot.append(f_k)
            #stop
            df_array = pd.DataFrame(columns=['x', 'y','frame'])
            df_array.x = x_plot
            df_array.y = y_plot
            df_array.frame = frame_plot 

            plotting.plot_slide1(df_array, save_dir, name + '_' + str(camera_name[cam]))
        '''

    for k in range(len(experiments_time)):
        
        shift_avg_array = []
        shift_diff_array = []
        all_shift_array = []
        all_diff_array = []

        plane_dict_array0 = plane_dict_array[0].copy()
        pose_2d_array0 = pose_2d_array[0]
        
        plane_dict_array_experiment1_array = []
        pose_dict_array_experiment1_array = []

        track1 = plane_dict_array0.keys()
        set1 = list(set(track1))
        subsect, subset_start = util.get_random_windowed_subset(set1, len(set1), offset = experiments_time[k], start = 0)
        plane_dict_array_subset0 = {key: plane_dict_array0.get(key, None) for key in subsect}
        pose_dict_array_subset0 = {key: pose_2d_array0.get(key, None) for key in subsect}
        plane_dict_array_experiment0 = util.subtract_keys(plane_dict_array_subset0)
        pose_dict_array_experiment0 = util.subtract_keys(pose_dict_array_subset0)

        plane_dict_array_experiment1_array.append(plane_dict_array_experiment0)
        pose_dict_array_experiment1_array.append(pose_dict_array_experiment0)
        
        for cam2 in cam_comb[1:]:
            #if np.dot(np.array(gt_rotation_array[cam1])[2, :], np.array(gt_rotation_array[cam2])[2, :]) < -0.85:
            #    print(" FAILED ")
            #    continue
            print(" MADE IT")
            plane_dict_array1 = plane_dict_array[cam2].copy()

            print("**************!!!")

            #############################################

            track2 = plane_dict_array1.keys()
            set2 = list(set(track2))
            
            shift_avg_array = []
            diff_avg_array = []
            true_offset_array = []
            #test_size = int(len(intersection))
            subsect1, subset_start1 = util.get_random_windowed_subset(set2, len(set2), start = experiments_time[k])
            
            true_offset = abs(subset_start - subset_start1)
            plane_dict_array_subset1 = {key: plane_dict_array1.get(key, None) for key in subsect1}
            pose_dict_array_subset1 = {key: pose_2d_array[cam2].get(key, None) for key in subsect1}

            #############################################          

            plane_dict_array_experiment1 = util.subtract_keys(plane_dict_array_subset1)
            pose_dict_array_experiment1 = util.subtract_keys(pose_dict_array_subset1)

            plane_dict_array_experiment1_array.append(plane_dict_array_experiment1)
            pose_dict_array_experiment1_array.append(pose_dict_array_experiment1)

        for exp in range(len(plane_dict_array_experiment1_array) - 1):
            for mode in [True, False]:
                
                mode_string = "no_sync"
                if mode is True:
                    mode_string = "sync"
                name_folder = '/' + str(subject) + '_' + str(experiments_time[k]) + '_' + str(exp) + '_' + mode_string

                save_dir = './plots/time_' + name + name_folder

                if os.path.isdir('./plots/time_' + name + name_folder) == False:
                    os.mkdir('./plots/time_' + name + name_folder)

                if os.path.isdir('./plots/time_' + name + name_folder + '/search_time') == False:
                    os.mkdir('./plots/time_' + name + name_folder + '/search_time')

                if os.path.isdir('./plots/time_' + name + name_folder + '/search_rot') == False:
                    os.mkdir('./plots/time_' + name + name_folder + '/search_rot')

                if os.path.isdir('./plots/time_' + name + name_folder + '/ICP') == False:
                    os.mkdir('./plots/time_' + name + name_folder + '/ICP')

                output_folder = 'terrace'
                #PLOTS SAVED IN PLOTS NOW!!!
                
                if os.path.isdir('./plots/time_' + name + name_folder + '/' + output_folder) == False:
                    os.mkdir('./plots/time_' + name + name_folder + '/' + output_folder)

                frame_array = []
                output_path = './plots/time_' + name + name_folder + '/' + output_folder + '/'


                cam_axis4 = []
                cam_position4 = []
            
                print(subsect1)
                print(" please work ")            
                print(subsect)

                ################################

                #best_shift_array = time_pred[experiments_time[k]] 
                best_shift_array = (experiments_time[k]*np.ones(len(plane_dict_array_experiment1_array) - 1))
                best_scale_array = list(np.zeros(len(best_shift_array))) 
                
                if mode == False:
                    best_shift_array[exp] = 0
                else:
                    best_shift_array[exp] = time_pred[subject][experiments_time[k]][exp]
                sync_dict_array = []
                print(best_shift_array, " BEST SHIFT")

                for sh in range(len(best_shift_array)):
                    sync_dict = time_align_inlier.time_knn(1.0, best_shift_array[sh], plane_dict_array_experiment1_array[sh + 1], plane_dict_array_experiment1_array[0])
                    sync_dict_array.append(sync_dict)

                icp_rot_array, icp_init_rot_array, icp_shift_array, init_ref_center_array, init_sync_center_array, time_shift_array, time_scale_array, sync_dict_array, index_array = ICP.icp(plane_dict_array_experiment1_array[0], plane_dict_array_experiment1_array[1:], best_shift_array, best_scale_array, sync_dict_array, save_dir = './plots/time_' + name + name_folder, name = '_')
                
                pre_bundle_cal_array = []
                single_view_cal_array = []
                cam_intrinsics_array = []
                pose_2d_array_comb = []
                #pose_2d_array_experiment = [pose_dict_array_experiment0, pose_dict_array_experiment1]
                pose_2d_array_experiment = pose_2d_array

                #indices_array = index_array
                indices_array = [list(sync_dict_array[0].keys()), list(sync_dict_array[0].values())]

                ref_indices = []
                for s in range(len(sync_dict_array)):
                    ref_indices.append(list(sync_dict_array[s].keys()))

                ref_intersection = set(ref_indices[0]).intersection(*map(set, ref_indices[1:]))

                indices_array = [ref_intersection]

                for s in range(len(sync_dict_array)):
                    sync_indices = []
                    for i in ref_intersection:
                        sync_indices.append(sync_dict_array[s][i])
                    indices_array.append(sync_indices)

                for i in range(0, len(plane_matrix_array)):

                    #icp_sync_shift_matrix = None

                    init_ref_shift_matrix = None
                    init_sync_shift_matrix = None
                    icp_rot_matrix = None
                    init_rot_matrix = None
                    plane_matrix = plane_matrix_array[i]
                    
                    print(len(init_ref_center_array), i, " HELLOASSDASD")
                    if i == 0:
                        init_ref_shift_matrix = np.array([init_ref_center_array[0][0], init_ref_center_array[0][1]])
                        init_sync_shift_matrix = np.array([init_ref_center_array[0][0], init_ref_center_array[0][1]])
                        icp_rot_matrix = np.array([[1,0 ], [0,1]])
                        init_rot_matrix = np.array([[1,0 ], [0,1]])
                        
                        #icp_sync_shift_matrix = np.array([icp_shift_array[0][0], icp_shift_array[0][1]])
                    
                    else:
                    
                        init_ref_shift_matrix = init_ref_center_array[i - 1]
                        init_sync_shift_matrix = init_sync_center_array[i - 1]
                        icp_rot_matrix = icp_rot_array[i - 1]
                        init_rot_matrix = icp_init_rot_array[i - 1]
                        plane_matrix = plane_matrix_array[i]

                        #icp_sync_shift_matrix = icp_shift_array[i - 1]


                    init_rot_matrix = np.array([[init_rot_matrix[0][0],init_rot_matrix[0][1],0],[init_rot_matrix[1][0],init_rot_matrix[1][1],0],[0,0,1]])
                    icp_rot_matrix = np.array([[icp_rot_matrix[0][0],icp_rot_matrix[0][1],0],[icp_rot_matrix[1][0],icp_rot_matrix[1][1],0],[0,0,1]])
                    
                    peturb_single_view = {'cam_matrix':  save_dict_array[i]['cam_matrix'],'ground_position':  save_dict_array[i]['ground_position'],'ground_normal':  save_dict_array[i]['ground_normal']}
                    peturb_extrinsics = {'init_sync_center_array': init_ref_shift_matrix, 'icp_rot_array': icp_rot_matrix, 'icp_init_rot_array': init_rot_matrix, 'plane_matrix_array': plane_matrix}

                    ###########################
                    pre_bundle_cal_array.append(peturb_extrinsics)
                    single_view_cal_array.append(peturb_single_view)
                    T01 = pos0#np.array([0,0,0])
                    #print(T01, " TO!!!!!!@#!@#")
                    R01 = icp_rot_matrix @ init_rot_matrix @ plane_matrix[:3,:3]
                    ##########################
                    sync_axis = np.transpose(R01)
                    t01_shift = (T01 + np.array([init_ref_shift_matrix[0], init_ref_shift_matrix[1], 0]))
                    t01_rot_shift = np.linalg.norm(np.array([init_sync_shift_matrix[0], init_sync_shift_matrix[1], 0]))*(init_rot_matrix @ np.array([0,1,0]))
                    t01_rot_rot_shift = (icp_rot_matrix @ t01_rot_shift) + t01_shift
                    #print(plane_matrix[:3,3], " plane matrix")

                    print(icp_rot_matrix, " ICP !!!!!")
                    print(init_rot_matrix, " INIT !!!!!")
                    
                    sync_position = -1*(t01_rot_rot_shift - np.array([init_sync_shift_matrix[0], init_sync_shift_matrix[1], 0]) - plane_matrix[:3,3])#np.transpose(ref_axis) @ T01 - pos0
                    #sync_position = -1*(t01_rot_rot_shift - np.array([init_sync_shift_matrix[0] + icp_sync_shift_matrix[0], init_sync_shift_matrix[1] + icp_sync_shift_matrix[1], 0]) - plane_matrix[:3,3])#np.transpose(ref_axis) @ T01 - pos0

                    cam_axis4.append(np.transpose(R01))
                    cam_position4.append(sync_position)
                    #cam_axis4.append(np.transpose(icp_rot_matrix @ init_rot_matrix @ plane_matrix[:3,:3]))
                    #cam_position4.append(icp_rot_matrix @ init_rot_matrix @ (plane_matrix[:3,3] - np.array([init_ref_shift_matrix[0], init_ref_shift_matrix[1], 0])))

                    cam_intrinsics = save_dict_array[i]#single_view_array[i]

                    cam_intrinsics_array.append(cam_intrinsics['cam_matrix'])
                    
                    pose_comb = {}
                    #for ia in [1000]:#indices_array[i]:\
                    for ia in indices_array[i]:
                        #print(indices_array)
                        #stop
                        #print(ia)
                        #if ia != 11:
                        #    continue
                
                        if ia in pose_2d_array_experiment[i]:
                            pose_comb[ia] = pose_2d_array_experiment[i][ia]
                    
                    pose_2d_array_comb.append(pose_comb)
                    #pose_2d_array_comb.append(pose_2d_array[cam[i]][indices_array[i]])
                print(pose_2d_array_comb[0].keys(), " HII1")
                print("************")
                print(pose_2d_array_comb[1].keys(), " HIII2")
                print("************")
                print(pose_2d_array_comb[2].keys(), " HIII3")
                print("************")
                print(pose_2d_array_comb[3].keys(), " HIII4")

                print(sync_dict_array)       
                #sync_dict_array = [{1000:1000}, {1000:1000}, {1000:1000}]         

                #matched_points = bundle_adjustment_plotly_multi.match_3d_multiview(sync_dict_array, pre_bundle_cal_array, single_view_cal_array, pose_2d_array_comb,center_array, output_path, scale = scale, title = 'rotation perturb TRACKS', name = ['perturb'], k = 20)
                
                #print(matched_points, " MATCHED POINTSSS")
                matched_points = bundle_adjustment_plotly_fix.match_3d_plotly_input2d_farthest_point(pre_bundle_cal_array, single_view_cal_array, pose_2d_array_comb,center_array, output_path, scale = scale, title = 'rotation perturb TRACKS', name = ['perturb'], k = 200)

                distortion_k_array = []
                distortion_p_array = []

                bundle_rotation_matrix_array = cam_axis4
                bundle_position_matrix_array = cam_position4
                bundle_intrinsic_matrix_array = cam_intrinsics_array

                run_name = '_' + str(i) + '_'
                '''
                w0 = 0.1
                w1 = 1	
                w2 = 1	
                w3 = 1	
                w4 = 1
                '''
                
                w0 = 1	
                w1 = 10	
                w2 = 10#0.1	
                w3 = 0.1	
                w4 = 0.1
                

                #bundle_rotation_matrix_array, bundle_position_matrix_array, bundle_intrinsic_matrix_array = bundle_adjustment_plotly_multi.bundle_adjustment_so3_gt(matched_points, bundle_rotation_matrix_array, bundle_position_matrix_array, bundle_intrinsic_matrix_array, gt_rotation_array, gt_translation_array, gt_intrinsics_array, h, distortion_k_array, distortion_p_array, save_dir = output_path, iteration = 200, run_name = run_name, w0 = w0, w1 = w1, w2 = w2, w3 = w3, w4 = w4)
                bundle_rotation_matrix_array, bundle_position_matrix_array, bundle_intrinsic_matrix_array = bundle_adjustment_plotly_fix.bundle_adjustment_so3_gt(matched_points, bundle_rotation_matrix_array, bundle_position_matrix_array, bundle_intrinsic_matrix_array, gt_rotation_array, gt_translation_array, gt_intrinsics_array, h, distortion_k_array, distortion_p_array, save_dir = output_path, iteration = 200, run_name = run_name, w0 = w0, w1 = w1, w2 = w2, w3 = w3)
                #print("*************************")
                #print(cam_axis4)
                #print("&&&&&&&&&&&&&&&&&&&&&&&&&")
                
                cam_position_rotate, cam_axis_rotate, R1, translation_template_rotate, translation_rotate = metrics.procrustes_rotation_translation_template(torch.unsqueeze(torch.tensor(cam_position4), dim = 0).double(), cam_axis4, torch.unsqueeze(torch.tensor(gt_translation_array), dim = 0).double(), gt_rotation_array, use_reflection=False, use_scaling=True)
                bundle_cam_position_rotate, bundle_cam_axis_rotate, bundle_R1, bundle_translation_template_rotate, bundle_translation_rotate = metrics.procrustes_rotation_translation_template(torch.unsqueeze(torch.tensor(bundle_position_matrix_array), dim = 0).double(),  bundle_rotation_matrix_array, torch.unsqueeze(torch.tensor(gt_translation_array), dim = 0).double(), gt_rotation_array, use_reflection=False, use_scaling=True)

                print(" PRE BUNDLE RESULTS")
                results_focal_pred1, results_focal_tsai1, angle_diff1, focal_error1, results_position_diff1 = eval_functions.evaluate_relative(cam_intrinsics_array, cam_axis_rotate, [cam_position_rotate], gt_intrinsics_array, gt_rotation_array, [gt_translation_array], output_path, 'pre_bundle')
                print("BUNDLE RESULTS")
                results_focal_pred2, results_focal_tsai2, angle_diff2, focal_error2, results_position_diff2 = eval_functions.evaluate_relative(bundle_intrinsic_matrix_array, bundle_cam_axis_rotate, [bundle_cam_position_rotate], gt_intrinsics_array, gt_rotation_array, [gt_translation_array], output_path, 'result_bundle')
                
                plotting.plot_camera_pose([cam_axis_rotate, gt_rotation_array], [cam_position_rotate, gt_translation_array], save_dir, scale = 1, name = "pre_bundle")
                plotting.plot_camera_pose([bundle_cam_axis_rotate, gt_rotation_array], [bundle_cam_position_rotate, gt_translation_array], save_dir, scale = 1, name = "pose_bundle")

                if mode == True:
                    with open('./plots/time_' + name + '/result_bundle_sync.csv','a') as file:

                        writer1 = csv.writer(file)
                        writer1.writerow([str(experiments_time[k]), str(exp), str(angle_diff1), str(focal_error1), str(results_position_diff1), str(angle_diff2), str(focal_error2), str(results_position_diff2)])
                        file.close
                else:
                    with open('./plots/time_' + name + '/result_bundle_no_sync.csv','a') as file:

                        writer1 = csv.writer(file)
                        writer1.writerow([str(experiments_time[k]), str(exp), str(angle_diff1), str(focal_error1), str(results_position_diff1), str(angle_diff2), str(focal_error2), str(results_position_diff2)])
                        file.close  