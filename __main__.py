# Might have to add to path
# sys.path.append('CalibS') 
import sys
sys.path.append('../../CalibSingleFromP2D/dlt_calib') 

from util import *
from run_calibration_ransac import *
from eval_human_pose import *
import json
from datetime import datetime
import csv
import matplotlib.image as mpimg
import os
import time_align
import numpy as np
import geometry
import plotting 
import multiview_utils
import ICP
import bundle_adjustment
import eval_functions
import torch
from xml.dom import minidom
import math 
import plotting_multiview

today = datetime.now()

metrics = Metrics()

name = str(today.strftime('%Y%m%d_%H%M%S')) + '_EPFL_campus4_res50'

(threshold_euc, threshold_cos, angle_filter_video, 
 confidence, termination_cond, num_points, h, iter, focal_lr, point_lr) = util.hyperparameter(
     'hyperparameter.json')
hyperparam_dict = {"threshold_euc": threshold_euc, "threshold_cos": threshold_cos, 
                   "angle_filter_video": angle_filter_video, "confidence": confidence, 
                   "termination_cond": termination_cond, "num_points": num_points, "h": h, 
                   "optimizer_iteration" :iter, "focal_lr" :focal_lr, "point_lr": point_lr}

output_path = 'outputs/plots/all_' + name

if os.path.isdir('./outputs') == False:
    os.mkdir('./outputs')

if os.path.isdir('outputs/plots') == False:
    os.mkdir('outputs/plots')

if os.path.isdir('outputs/plots/all_' + name) == False:
    os.mkdir('outputs/plots/all_' + name)

if os.path.isdir('outputs/plots/all_' + name + '/bundle') == False:
    os.mkdir('outputs/plots/all_' + name + '/bundle')

with open('outputs/plots/all_' + name +  '/result_sync.csv','a') as file:
    writer1 = csv.writer(file)
    writer1.writerow(["shift gt", "shift", "subset", "camera1", "camera2"])
    file.close

with open('outputs/plots/all_' + name + '/result_average_sync.csv','a') as file:
    writer1 = csv.writer(file)
    writer1.writerow(["shift gt", "cam1", "cam2", "shift avg", "shift std"])
    file.close

with open('outputs/plots/all_' + name + '/result_average_all.csv','a') as file:
    writer1 = csv.writer(file)
    writer1.writerow(["shift gt", "shift avg", "shift std", "diff avg", "diff std"])
    file.close

with open('outputs/plots/all_' + name + '/result_bundle_sync.csv','a') as file:
    writer1 = csv.writer(file)
    writer1.writerow(["cam1", "cam2", "offset", "offset pred", "offset diff", "exp", "focal pre bundle", "focal_tsai", "angle_diff pre bundle", "error_npjpe pre bundle", "focal_error pre bundle", "results_position_diff pre bundle", "focal bundle", "focal_tsai", "angle_diff bundle", "error_npjpe bundle", "focal_error bundle", "results_position_diff bundle"])
    file.close

with open('outputs/plots/all_' + name + '/result_bundle_no_sync.csv','a') as file:
    writer1 = csv.writer(file)
    writer1.writerow(["cam1", "cam2", "offset", "offset pred", "offset diff", "exp", "focal pre bundle", "focal_tsai", "angle_diff pre bundle", "error_npjpe pre bundle", "focal_error pre bundle", "results_position_diff pre bundle", "focal bundle", "focal_tsai", "angle_diff bundle", "error_npjpe bundle", "focal_error bundle", "results_position_diff bundle"])
    file.close

terrace_array_names = ['terrace1-c0_avi', 'terrace1-c1_avi', 'terrace1-c2_avi', 'terrace1-c3_avi']

cam_comb = [(0,1), (0,2)]
print(cam_comb)

with open('configuration.json', 'r') as f:
    configuration = json.load(f)

num = 0
focal_array = []
calib_array = []

ankle_head_2d_array = []
pose_2d_array = []

plane_matrix_array = []    
plane_dict_array = []

save_dict_array = []


gt_rotation_array = []
gt_translation_array = []
gt_intrinsics_array = []

for vid in terrace_array_names:
    intrinsic_gt_path = 'example_data/gt_calibrations/single_calibration_' + str(num) + '_.json'
    
    with open(intrinsic_gt_path, 'r') as f:
        intrinsic_gt = json.load(f)
    
    cam_extrinsic_path = 'example_data/gt_calibrations/terrace-tsai-c' + str(num) + '.xml'
    print(cam_extrinsic_path, " THE EXTRISIC PATH")
    cam_extrinsic = minidom.parse(cam_extrinsic_path)

    models_extrinsics = cam_extrinsic.getElementsByTagName('Extrinsic')
    models_instrinsics = cam_extrinsic.getElementsByTagName('Intrinsic')
    models_geometry = cam_extrinsic.getElementsByTagName('Geometry')
    
    tx = float(models_extrinsics[0].attributes.items()[0][1])/1000.0
    ty = float(models_extrinsics[0].attributes.items()[1][1])/1000.0
    tz = float(models_extrinsics[0].attributes.items()[2][1])/1000.0
    rx = float(models_extrinsics[0].attributes.items()[3][1])
    ry = float(models_extrinsics[0].attributes.items()[4][1])
    rz = float(models_extrinsics[0].attributes.items()[5][1])
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(rx), -math.sin(rx) ],
                    [0,         math.sin(rx), math.cos(rx)  ]
                    ])

    R_y = np.array([[math.cos(ry),    0,      math.sin(ry)  ],
                    [0,                     1,      0                   ],
                    [-math.sin(ry),   0,      math.cos(ry)  ]
                    ])

    R_z = np.array([[math.cos(rz),    -math.sin(rz),    0],
                    [math.sin(rz),    math.cos(rz),     0],
                    [0,                     0,                      1]
                    ])
    R_rot = R_z @ R_y @ R_x
    cam_pos = (np.linalg.inv(R_rot)@(-np.array([tx,ty,tz])))
    gt_rotation_array.append(R_rot)
    gt_translation_array.append(cam_pos)

    gt_intrinsics_array.append(intrinsic_gt["intrinsics"])
    num = num + 1

for vid in terrace_array_names:
    
    with open('example_data/detections/result_' + vid.split('_')[0] + '_.json', 'r') as f:
        points_2d = json.load(f)
    
    datastore_cal = data.coco_mmpose_dataloader(points_2d, bound_lower = 100, bound = 500)  
    frame_dir = 'example_data/frames/' + vid + '/00000000.jpg'
    img = mpimg.imread(frame_dir)
    
    (ankles, cam_matrix, normal, ankleWorld, focal, focal_batch, ransac_focal, datastore_filtered) = run_calibration_ransac(
         datastore_cal, 'hyperparameter.json', img, 
         img.shape[1], img.shape[0], name, num, skip_frame = configuration['skip_frame'], 
         max_len = configuration['max_len'], min_size = configuration['min_size'], save_dir = './outputs')
    focal_array.append(cam_matrix[0][0])
    calib_array.append({'cam_matrix': cam_matrix, 'ground_normal': normal, 'ground_position': ankleWorld})
    print(ankles, cam_matrix, normal)
    
    #########################
    save_dict = {"cam_matrix":cam_matrix, "ground_normal":normal, "ground_position":ankleWorld}
    ##################################
    datastore = data.coco_mmpose_dataloader(points_2d, bound_lower = 100, bound = 500) 
    data_2d = multiview_utils.get_ankles_heads_dictionary(datastore, cond_tol = confidence)
    pose_2d = multiview_utils.get_ankles_heads_pose_dictionary(datastore, cond_tol = confidence)
    print(len(pose_2d), len(data_2d), " THIS IS POSE 2D")

    ankle_head_2d_array.append(data_2d)
    pose_2d_array.append(pose_2d)
    
    plane_matrix, basis_matrix = geometry.find_plane_matrix(save_dict["ground_normal"], np.linalg.inv(save_dict['cam_matrix']),save_dict['ground_position'], img.shape[1], img.shape[0])
    
    to_pickle_plane_matrix = {"plane_matrix": plane_matrix,'intrinsics': save_dict['cam_matrix']}
    
    plane_data_2d = geometry.camera_to_plane(data_2d, cam_matrix, plane_matrix, save_dict['ground_position'], save_dict["ground_normal"], img.shape[1], img.shape[0])

    plane_matrix_array.append(plane_matrix)
    
    plane_dict_array.append(plane_data_2d)
    ##################################
    save_dict = {"cam_matrix":cam_matrix, "ground_normal":normal, "ground_position":ankleWorld}
    save_dict_array.append(save_dict)

    num = num + 1


best_shift_array, best_scale_array, sync_dict_array = time_align.time_all(plane_dict_array[0], plane_dict_array[1:], save_dir = 'outputs/plots/all_' + name, sync = True, name = "temporal", window = 1, dilation = 1)

for k in range(len(best_shift_array)):
    with open('./outputs/plots/all_' + name + '/result_sync.csv','a') as file:

        writer1 = csv.writer(file)

        writer1.writerow([str(k), best_shift_array[k]])
        file.close

####################

icp_rot_array, icp_init_rot_array, icp_shift_array, init_ref_center_array, init_sync_center_array, time_shift_array, time_scale_array, sync_dict_array, index_array = ICP.icp(plane_dict_array[0], plane_dict_array[1:], best_shift_array, best_scale_array, sync_dict_array, save_dir = './outputs/plots/all_' + name, name = '_')

pre_bundle_cal_array = []
single_view_cal_array = []
cam_intrinsics_array = []
pose_2d_array_comb = []
pose_2d_array_experiment = pose_2d_array

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

cam_axis4 = []
cam_position4 = []
pos0 = np.array([0,0,0])

for i in range(0, len(plane_matrix_array)):

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
    
    else:
    
        init_ref_shift_matrix = init_ref_center_array[i - 1]
        init_sync_shift_matrix = init_sync_center_array[i - 1]
        icp_rot_matrix = icp_rot_array[i - 1]
        init_rot_matrix = icp_init_rot_array[i - 1]
        plane_matrix = plane_matrix_array[i]

    init_rot_matrix = np.array([[init_rot_matrix[0][0],init_rot_matrix[0][1],0],[init_rot_matrix[1][0],init_rot_matrix[1][1],0],[0,0,1]])
    icp_rot_matrix = np.array([[icp_rot_matrix[0][0],icp_rot_matrix[0][1],0],[icp_rot_matrix[1][0],icp_rot_matrix[1][1],0],[0,0,1]])
    
    peturb_single_view = {'cam_matrix':  save_dict_array[i]['cam_matrix'],'ground_position':  save_dict_array[i]['ground_position'],'ground_normal':  save_dict_array[i]['ground_normal']}
    peturb_extrinsics = {'init_sync_center_array': init_ref_shift_matrix, 'icp_rot_array': icp_rot_matrix, 'icp_init_rot_array': init_rot_matrix, 'plane_matrix_array': plane_matrix}

    ###########################
    pre_bundle_cal_array.append(peturb_extrinsics)
    single_view_cal_array.append(peturb_single_view)
    T01 = pos0#
    R01 = icp_rot_matrix @ init_rot_matrix @ plane_matrix[:3,:3]
    ##########################
    sync_axis = np.transpose(R01)
    t01_shift = (T01 + np.array([init_ref_shift_matrix[0], init_ref_shift_matrix[1], 0]))
    t01_rot_shift = np.linalg.norm(np.array([init_sync_shift_matrix[0], init_sync_shift_matrix[1], 0]))*(init_rot_matrix @ np.array([0,1,0]))
    t01_rot_rot_shift = (icp_rot_matrix @ t01_rot_shift) + t01_shift

    print(icp_rot_matrix, " ICP !!!!!")
    print(init_rot_matrix, " INIT !!!!!")
    
    sync_position = -1*(t01_rot_rot_shift - np.array([init_sync_shift_matrix[0], init_sync_shift_matrix[1], 0]) - plane_matrix[:3,3])#np.transpose(ref_axis) @ T01 - pos0

    cam_axis4.append(np.transpose(R01))
    cam_position4.append(sync_position)
    cam_intrinsics = save_dict_array[i]

    cam_intrinsics_array.append(cam_intrinsics['cam_matrix'])
    
    pose_comb = {}
    for ia in indices_array[i]:

        if ia in pose_2d_array_experiment[i]:
            pose_comb[ia] = pose_2d_array_experiment[i][ia]
    
    pose_2d_array_comb.append(pose_comb)

print(pose_2d_array_comb[0].keys(), " HII1")
print("************")
print(pose_2d_array_comb[1].keys(), " HIII2")
print("************")
print(pose_2d_array_comb[2].keys(), " HIII3")
print("************")
print(pose_2d_array_comb[3].keys(), " HIII4")

print(sync_dict_array)       

matched_points = bundle_adjustment.match_3d_plotly_input2d_farthest_point(pre_bundle_cal_array, single_view_cal_array, pose_2d_array_comb, k = 20)

distortion_k_array = []
distortion_p_array = []

bundle_rotation_matrix_array = cam_axis4
bundle_position_matrix_array = cam_position4
bundle_intrinsic_matrix_array = cam_intrinsics_array

run_name = '_' + str(i) + '_'

w0 = 1	
w1 = 10	
w2 = 10	
w3 = 0.1	
w4 = 0.1

bundle_rotation_matrix_array, bundle_position_matrix_array, bundle_intrinsic_matrix_array = bundle_adjustment.bundle_adjustment(matched_points, bundle_rotation_matrix_array, bundle_position_matrix_array, bundle_intrinsic_matrix_array, h, distortion_k_array, distortion_p_array, save_dir = output_path, iteration = 200, run_name = run_name, w0 = w0, w1 = w1, w2 = w2, w3 = w3)

cam_position_rotate, cam_axis_rotate, R1, translation_template_rotate, translation_rotate = metrics.procrustes_rotation_translation_template(torch.unsqueeze(torch.tensor(cam_position4), dim = 0).double(), cam_axis4, torch.unsqueeze(torch.tensor(gt_translation_array), dim = 0).double(), gt_rotation_array, use_reflection=False, use_scaling=True)
bundle_cam_position_rotate, bundle_cam_axis_rotate, bundle_R1, bundle_translation_template_rotate, bundle_translation_rotate = metrics.procrustes_rotation_translation_template(torch.unsqueeze(torch.tensor(bundle_position_matrix_array), dim = 0).double(),  bundle_rotation_matrix_array, torch.unsqueeze(torch.tensor(gt_translation_array), dim = 0).double(), gt_rotation_array, use_reflection=False, use_scaling=True)

print(" PRE BUNDLE RESULTS")
results_focal_pred1, results_focal_tsai1, angle_diff1, focal_error1, results_position_diff1 = eval_functions.evaluate_relative(cam_intrinsics_array, cam_axis_rotate, [cam_position_rotate], gt_intrinsics_array, gt_rotation_array, [gt_translation_array], output_path, 'pre_bundle')
print("BUNDLE RESULTS")
results_focal_pred2, results_focal_tsai2, angle_diff2, focal_error2, results_position_diff2 = eval_functions.evaluate_relative(bundle_intrinsic_matrix_array, bundle_cam_axis_rotate, [bundle_cam_position_rotate], gt_intrinsics_array, gt_rotation_array, [gt_translation_array], output_path, 'result_bundle')

plotting_multiview.plot_camera_pose([cam_axis_rotate, gt_rotation_array], [cam_position_rotate, gt_translation_array], 'outputs/plots/all_' + name + '/bundle', scale = 1, name = "pre_bundle")
plotting_multiview.plot_camera_pose([bundle_cam_axis_rotate, gt_rotation_array], [bundle_cam_position_rotate, gt_translation_array], 'outputs/plots/all_' + name + '/bundle', scale = 1, name = "pose_bundle")

with open('outputs/plots/all_' + name + '/result_bundle_sync.csv','a') as file:

    writer1 = csv.writer(file)
    writer1.writerow(["0", "0", str(angle_diff1), str(focal_error1), str(results_position_diff1), str(angle_diff2), str(focal_error2), str(results_position_diff2)])
    file.close
