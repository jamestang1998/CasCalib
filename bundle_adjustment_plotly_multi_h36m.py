
import torch
import matplotlib.pyplot as plt
import numpy as np
import util
import matplotlib.image as mpimg
import bundle_intersect 

import bone_length
from eval_human_pose import Metrics
import bundle_intersect
from random import sample
from sklearn.neighbors import NearestNeighbors
from xml.dom import minidom
import math

import plotly.express as px

import plotly.express as px
import plotly.graph_objects as go
import traceback

metrics = Metrics()
import pytorch3d
from pytorch3d.renderer import PerspectiveCameras

from torch.optim import Adam
from torch.optim import SGD

from pytorch3d.transforms.so3 import (
    so3_exponential_map,
    so3_relative_angle,
)
#wandb.init(project="bundle-adjustment")
# TODO
# scale/camera position constraints?
# TUNE your learning rate
# LOOK for numerical instability by checking the learning rate plots and seeing where it gets worse

#10-13-2022
#Measure scale i.e. average depth of the person keypoints , compare to 3d gt
#scale invariant camera location i.e. camera positions normalized (relative camera positions shouldnt change)

#10- 20 -2022
#try computing ankle from the hip instead of from ankle? (ankle moves too much)
# !!!!!!!!!!!!! compare relative rotations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#10 - 29 - 2022
# why dont you just compute the match in here instead of this index passing???????
# FIX THE SYNC MATCHING!!!!!
class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, match1, match2):
            'Initialization'
            self.match1 = match1
            self.match2 = match2

    def __len__(self):
            'Denotes the total number of samples'
            return len(self.match1)

    def __getitem__(self, index):
            'Generates one sample of data'
            X = self.match1[index].double()
            y = self.match2[index].double()

            return X, y

def back_project(point_2d, R, T, f, p):

    #point_2d = point_2d_array

    #point_2d[:, 2] = -1*point_2d[:, 2]
    
    #print(f, torch.tensor([0]).double(), p, " F P TENSOR !!!!!!!!!!!!!!!!")
    #print(f[0], torch.tensor([0]).double(), p[[0]], p[[1]], " F P TENSOR")
    #print(f[[0]].shape,f[[1]].shape, torch.tensor([0]).shape, p[0].shape, " F P TENSOR")
    #print(f.shape, torch.tensor([0]).shape, p.shape, " F P TENSOR !!")
    #f -> torch.tensor([f1, f2])]
    '''
    k = torch.zeros((3,3)).double()
    
    k[0][0] = k[0][0] + f[0]
    k[1][1] = k[1][1] + f[1]

    k[0][2] = k[0][2] + p[0]
    k[1][2] = k[1][2] + p[1]
    k[2][2] = k[2][2] + 1.0
    '''
    
    
    row1 = torch.stack([f[[0]], torch.tensor([0]).double(), p[[0]]])
    row2 = torch.stack([torch.tensor([0]).double(), f[[1]], p[[1]]])
    row3 = torch.stack([torch.tensor([0]).double(), torch.tensor([0]).double(), torch.tensor([1.0]).double()])

    k = torch.squeeze(torch.stack([row1, row2, row3]))

    print(k, " HELLO k")
    #stop
    #print(f)
    #print(k)
    #R1 = torch.transpose(torch.squeeze(R), 0, 1)
    #print(R1.shape, T.shape, " R T")
    #T1 = -R1 @ T
    #R1 = torch.transpose(torch.squeeze(R), 0, 1)
    R1 = torch.transpose(torch.squeeze(R), 0, 1)
    #print(R1.shape, T.shape, " R T")
    T1 = -R1 @ T
    #print("*****************************************")
    #print(R, T, " R AND T")
    #print(R1, T1, " R1 AND T1")
    #print(point_2d, " points")
    #print(R1.shape, T1.shape, k.shape, point_2d.shape, " SHAPESSSS")
    #ray = torch.inverse(R1) @ (torch.inverse(k) @ torch.transpose(point_2d, 0, 1) - T1)
    #ray = torch.inverse(R1) @ (torch.inverse(k) @ torch.transpose(point_2d, 0, 1) - torch.unsqueeze(T1,dim = 1))
    #ray = (torch.inverse(k) @ torch.transpose(point_2d, 0, 1) - torch.unsqueeze(T1,dim = 1))
    #ray = torch.inverse(R1) @ ((torch.inverse(k) @ torch.transpose(point_2d, 0, 1)) - torch.unsqueeze(T1,dim = 1))
    ray = torch.inverse(R1) @ ((torch.inverse(k) @ torch.transpose(point_2d, 0, 1)) - torch.unsqueeze(T1,dim = 1))

    #ray = torch.inverse(R1) @ (torch.transpose(point_2d, 0, 1) - torch.unsqueeze(T1,dim = 1))
    #ray = torch.inverse(R1) @ (torch.transpose(point_2d, 0, 1) - torch.unsqueeze(T1,dim = 1))
    return torch.transpose(ray, 0, 1)


def bundle_adjustment_so3_gt(matched_points, camera_rotation, camera_translation, single_view_array, camera_rotation_gt, camera_translation_gt, single_view_array_gt, h = 1.6, distortion_k_array = [], distortion_p_array = [], iteration = 100, save_dir = '',run_name = '', focal_lr = 0.1, rot_lr = 0.1 , translation_lr = 0.1, w0 = 1.0, w1 = 1.0, w2 = 1.0, w3 = 1.0, w4 = 1.0, plot_true = True):
    
    fig1, ax1 = plt.subplots(1, 1)

    ##########################################
    camera_array_R = []
    camera_array_T = []
    camera_array_f = []
    camera_array_p = []

    camera_array_tan = []
    camera_array_rad = []

    for i in range(len(camera_rotation)):

        R=torch.unsqueeze(torch.from_numpy(camera_rotation[i]), dim = 0).double()
        T=torch.unsqueeze(torch.from_numpy(camera_translation[i]), dim = 0).double()
        focal_length=torch.tensor([[single_view_array[i][0][0], single_view_array[i][0][0]]]).double()
        principal_point=torch.tensor([[single_view_array[i][0][2], single_view_array[i][1][2]]]).double()

        camera_array_R.append(R)
        camera_array_T.append(T)
        camera_array_f.append(focal_length)
        camera_array_p.append(principal_point)
        
        if len(distortion_k_array) > 0:
            camera_array_tan.append(distortion_k_array[i])
            camera_array_rad.append(distortion_p_array[i])
    
    R=torch.cat(camera_array_R, dim = 0).double()#.detach().clone().requires_grad_(True)
    T=torch.cat(camera_array_T, dim = 0).double()#.detach().clone().requires_grad_(True)
    focal_length=torch.cat(camera_array_f, dim = 0).double()#.detach().clone().requires_grad_(True)
    principal_point=torch.cat(camera_array_p, dim = 0).double()#.detach().clone().requires_grad_(True)

    R_axis_angle = pytorch3d.transforms.matrix_to_axis_angle(R)
    #print(R_axis_angle)
    #print(R_absolute)
    #print(R.shape, T.shape, focal_length.shape, principal_point.shape, " SHAOPESSS")
    camera_array = PerspectiveCameras(
        R=R_axis_angle,#R,
        T=T,
        focal_length=focal_length, 
        principal_point=principal_point
        )

    R_matrix = camera_array.R
    T_matrix = camera_array.T
    f_matrix = camera_array.focal_length
    p_matrix = camera_array.principal_point
    
    error_array = []
    params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 1}
    
    print(R_matrix)
    print("*************")
    #stop
    R_matrix.requires_grad = True
    T_matrix.requires_grad = True
    f_matrix.requires_grad = True

    optimizer = Adam([
        {'params': R_matrix, 'lr':rot_lr},
        {'params': T_matrix, 'lr':translation_lr},
        {'params': f_matrix, 'lr':focal_lr}
    ])
    
    accum_iter = 0
    itr = 0
    
    avg_error_array = []
    print("Hello i am in the loop now")
    
    joint_r = [0, 2, 4, 10, 12, 14]
    joint_l = [1, 3, 5, 11, 13, 15]

    height_r = [0, 2, 4, 6, 7]
    height_l = [1, 3, 5, 6, 7]
    
    '''
    keypoint_array = ['nose',           #0
                      'left_eye',       #1
                      'right_eye',      #2
                      'left_ear',       #3
                      'right_ear',      #4
                      'left_shoulder',  #5
                      'right_shoulder', #6
                      'left_elbow',     #7
                      'right_elbow',    #8
                      'left_wrist',     #9
                      'right_wrist',    #10
                      'left_hip',       #11
                      'right_hip',      #12
                      'left_knee',      #13
                      'right_knee',     #14
                      'left_ankle',     #15
                      'right_ankle']    #16
    
    l_ankle = 15
    r_ankle = 16
    '''
    '''
    keypoint_array = ['nose',           #0
                      'left_eye',       #1
                      'right_eye',      #2
                      'left_ear',       #3
                      'right_ear',      #4
                      'left_shoulder',  #5
                      'right_shoulder', #6
                      'left_hip',       #7
                      'right_hip',      #8
                      'left_knee',      #9
                      'right_knee',     #10
                      'left_ankle',     #11
                      'right_ankle']    #12
    '''

    keypoint_dict =  {
                      'left_shoulder':0,
                      'right_shoulder':1,
                      'left_hip':2,
                      'right_hip':3,
                      'left_knee':4,
                      'right_knee':5,
                      'left_ankle':6,
                      'right_ankle':7}
    
    keypoint_array = [
                      'left_shoulder',  #0
                      'right_shoulder', #1
                      'left_hip',       #2
                      'right_hip',      #3
                      'left_knee',      #4
                      'right_knee',     #5
                      'left_ankle',     #6
                      'right_ankle']    #7
    
    l_ankle = 6
    r_ankle = 7
    #key_point_select = [0,1,2,3,4,5,6,11,12,13,14,15,16]
    key_point_select = [5,6,11,12,13,14,15,16]
    #key_point_arms = [7,8,9,10]
    
    joint_r = [2, 5, 7]
    joint_l = [1, 4, 6]

    #height_r = [0, 2, 4, 6, 7]
    #height_l = [1, 3, 5, 6, 7]
    height_r = [2, 5, 7]
    height_l = [1, 4, 6]

    joint_array =  [
                    [1,0],  #Right Shoulder - Left Shoulder #0                              
                    [0,2], #Left Shoulder - Left Hip        #1
                    [1,3], #Right Shoulder - Right Hip      #2
                    [3,2],#Right Hip - Left Hip             #3
                    [2,4],#Left Hip - Left Knee             #4
                    [3,5],#Right Hip - Right Knee          #5
                    [4,6],#Left Knee - Left Ankle          #6
                    [5,7]]#Right Knee - Right Ankle       #7
    
    

    '''
    joint_array =  [[0,1],  #Nose - Right Eye               #0
                    [0,2],  #Nose - Left Eye                #1
                    [1,2],  #Right Eye - Left Eye           #2
                    [1,3],  #Right Eye - Right Ear          #3
                    [2,4],  #Left Eye - Left Ear            #4
                    [3,5],  #Right Ear - Right Shoulder     #5
                    [4,6],  #Left Ear - Left Shoulder       #6
                    [5,6],  #Right Shoulder - Left Shoulder #7 
                             
                    [5,7], #Left Shoulder - Left Hip       #8
                    [6,8], #Right Shoulder - Right Hip     #9
                    [8,7],#Right Hip - Left Hip           #10
                    
                    [7,9],#Left Hip - Left Knee           #11
                    [8,10],#Right Hip - Right Knee         #12
                    [9,11],#Left Knee - Left Ankle         #13
                    [10,12]]#Right Knee - Right Ankle       #14
    '''

    '''
    joint_r = [2, 5, 7]
    joint_l = [1, 4, 6]

    #height_r = [0, 2, 4, 6, 7]
    #height_l = [1, 3, 5, 6, 7]
    height_r = [2, 5, 7]
    height_l = [1, 4, 6]
    '''
    '''
    joint_array =  [[0,1],  #Nose - Right Eye               #0
                    [0,2],  #Nose - Left Eye                #1
                    [1,2],  #Right Eye - Left Eye           #2
                    [1,3],  #Right Eye - Right Ear          #3
                    [2,4],  #Left Eye - Left Ear            #4
                    [3,5],  #Right Ear - Right Shoulder     #5
                    [4,6],  #Left Ear - Left Shoulder       #6
                    [5,6],  #Right Shoulder - Left Shoulder #7              
                    [6,8],  #Left Shoulder - Left Elbow     #8
                    [5,7],  #Right Shoulder - Right Elbow   #9
                    [7,9],  #Right Elbow - Right Wrist      #10
                    [8,10], #Left Elbow - Left Wrist        #11
                    [6,12], #Left Shoulder - Left Hip       #12
                    [5,11], #Right Shoulder - Right Hip     #13
                    [11,12],#Right Hip - Left Hip           #14
                    [12,14],#Left Hip - Left Knee           #15
                    [11,13],#Right Hip - Right Knee         #16
                    [14,16],#Left Knee - Left Ankle         #17
                    [13,15]]#Right Knee - Right Ankle       #18
    '''
    '''
    joint_array_all =  [[0,1],  #Nose - Right Eye               #0
                        [0,2],  #Nose - Left Eye                #1
                        [1,2],  #Right Eye - Left Eye           #2
                        [1,3],  #Right Eye - Right Ear          #3
                        [2,4],  #Left Eye - Left Ear            #4
                        [3,5],  #Right Ear - Right Shoulder     #5
                        [4,6],  #Left Ear - Left Shoulder       #6
                        [5,6],  #Right Shoulder - Left Shoulder #7              
                        [6,8],  #Left Shoulder - Left Elbow     #8
                        [5,7],  #Right Shoulder - Right Elbow   #9
                        [7,9],  #Right Elbow - Right Wrist      #10
                        [8,10], #Left Elbow - Left Wrist        #11
                        [6,12], #Left Shoulder - Left Hip       #12
                        [5,11], #Right Shoulder - Right Hip     #13
                        [11,12],#Right Hip - Left Hip           #14
                        [12,14],#Left Hip - Left Knee           #15
                        [11,13],#Right Hip - Right Knee         #16
                        [14,16],#Left Knee - Left Ankle         #17
                        [13,15]]#Right Knee - Right Ankle       #18
    '''
    f_param_array = []
    r_matrix_array = []
    t_matrix_array = []
    print(len(matched_points[0][0]), " LEN")
    comb_array = util.random_combination(list(range(len(matched_points[0][0]))), 2, np.inf)

    '''
    comb_array = []
    for i in range(1, len(matched_points[0][0])):
        comb_array.append((0, i))
    '''
    comb_stack = []
    for ca in comb_array:
        
        pair0 = []
        pair1 = []
        for md in range(len(matched_points[0])):
            print("**************************")
            match_dict = matched_points[0][md]
            print(match_dict, " MATCH DICT")

            person1 = []
            person2 = []

            cam1 = match_dict[ca[0]]['cam']#list(match_dict.keys())[0] 
            cam2 = match_dict[ca[1]]['cam']#list(match_dict.keys())[1] 
            print(cam1, cam2)

            points_cam0 = torch.tensor(list(match_dict[ca[0]]['pose']))[key_point_select, :].double()
            points_cam1 = torch.tensor(list(match_dict[ca[1]]['pose']))[key_point_select, :].double()

            #points_cam0 = torch.tensor(list(match_dict[ca[0]]['pose'].values())[:-1]).double()
            #points_cam1 = torch.tensor(list(match_dict[ca[1]]['pose'].values())[:-1]).double()
            
            pair0.append(points_cam0)
            pair1.append(points_cam1)
            '''
            fig11, ax11 = plt.subplots(1, 1)
            ax11.scatter(np.array(list(match_dict[ca[0]]['pose'].values()))[:,0], np.array(list(match_dict[ca[0]]['pose'].values())[:-1])[:,1], c = 'r')
            
            joint1 = np.array(list(match_dict[ca[0]]['pose']))[:,0]
            joint2 = np.array(list(match_dict[ca[0]]['pose']))[:,1]
            
            #for ap in joint_array:
            #    ax11.plot([joint1[ap[0]], joint1[ap[1]]],[joint2[ap[0]], joint2[ap[1]]], c = 'r')
            
            plt.gca().set_aspect('equal', adjustable='box')
            ax11.invert_yaxis()

            plt.show()
            '''
            #ax1.scatter(points_cam2.clone().reshape(-1, 3)[:, 0], points_cam2.clone().reshape(-1, 3)[:, 1], c = 'b')

        pair_stack0 = torch.stack(pair0)
        pair_stack1 = torch.stack(pair1)

        comb_stack.append({ca[0]: pair_stack0, ca[1]: pair_stack1})
        print(pair_stack0.shape, " THE STACKED SHApE")

    for i in range(iteration):
        optimizer.zero_grad()
        error_array = []
        data = []

        pred_3d_array = []
        shoulder_array = []
        hip_array = []
        
        loss_array = []
        for comb_dict in comb_stack:

            #data = []
            itr = 0
            
            person1 = []
            person2 = []

            #print(comb_dict)

            cam1 = list(comb_dict.keys())[0]
            cam2 = list(comb_dict.keys())[1]

            points_cam1 = comb_dict[cam1].reshape((-1,3))#torch.tensor(list(match_dict[ca[0]]['pose'].values())[:-1]).double()
            points_cam2 = comb_dict[cam2].reshape((-1,3))#torch.tensor(list(match_dict[ca[1]]['pose'].values())[:-1]).double()
            
            print(points_cam1, " points cam1")
            print(points_cam2, " points cam2")
            #stop
            #print(cam1, cam2, " THE CAMMERAAAAAA")

            #training_set = Dataset(match_dict[ca[0]]['pose'].values(), match_dict[ca[1]]['pose'].values())
            #training_generator = torch.utils.data.DataLoader(training_set, **params)

            end1_array = []
            end2_array = []
            
            R_mat1 = so3_exponential_map(R_matrix)[cam1]
            R_mat2 = so3_exponential_map(R_matrix)[cam2]
            #stop         
            
            den = 0

            #points_cam1 = comb_dict[cam1].reshape((-1,3))#torch.tensor(list(match_dict[ca[0]]['pose'].values())[:-1]).double()
            #points_cam2 = comb_dict[cam2].reshape((-1,3))#torch.tensor(list(match_dict[ca[1]]['pose'].values())[:-1]).double()

            #stop
            #for points_cam1, points_cam2 in training_generator:
                
            #print(points_cam1.shape, " POINTS CAM")
            #points_cam1 = points_cam1[:,:, key_point_select, :]
            #points_cam2 = points_cam2[:,:, key_point_select, :]

            ax1.scatter(points_cam1.clone().reshape(-1, 3)[:, 0], points_cam1.clone().reshape(-1, 3)[:, 1], c = 'r')
            ax1.scatter(points_cam2.clone().reshape(-1, 3)[:, 0], points_cam2.clone().reshape(-1, 3)[:, 1], c = 'b')
            

            row1 = torch.stack([f_matrix[cam1][[0]], torch.tensor([0]).double(), p_matrix[cam1][[0]]]).double()
            row2 = torch.stack([torch.tensor([0]).double(), f_matrix[cam1][[1]], p_matrix[cam1][[1]]]).double()
            row3 = torch.stack([torch.tensor([0]).double(), torch.tensor([0]).double(), torch.tensor([1.0]).double()])

            k = torch.squeeze(torch.stack([row1, row2, row3]))

            plane = torch.inverse(k) @ torch.transpose(points_cam1.clone().reshape(-1, 3), 0, 1)

            #print(plane, " pLane")

            plane = torch.vstack([plane, torch.ones((1, plane.shape[1]))])

            end1 =  torch.transpose(torch.transpose(R_mat1, 0, 1) @ plane[:3, :], 0, 1)[:, :3] + torch.unsqueeze(T_matrix[cam1], dim = 0)
            
            #print(torch.transpose(torch.transpose(R_mat1, 0, 1) @ plane[:3, :], 0, 1)[:, :3], " END11111 after r mat")
            
            #print(plane, " PLANE1")
            #print(end1, " END11111")
            #stop
            #print(camera_array_tan)
            if len(distortion_k_array) > 0:
                u1 = torch.transpose(points_cam1.clone().reshape(-1, 3), 0, 1)[0, :]
                v1 = torch.transpose(points_cam1.clone().reshape(-1, 3), 0, 1)[1, :]
                end1 = util.distortion_apply(u1, v1, k, R_mat1, T_matrix[cam1], camera_array_tan[cam1], camera_array_rad[cam1])
            #print(plane, " THE PLANE")
            #stop
            ####################################################
            #print(end1.shape, " END11111111111111111")
            row1 = torch.stack([f_matrix[cam2][[0]], torch.tensor([0]).double(), p_matrix[cam2][[0]]]).double()
            row2 = torch.stack([torch.tensor([0]).double(), f_matrix[cam2][[1]], p_matrix[cam2][[1]]]).double()
            row3 = torch.stack([torch.tensor([0]).double(), torch.tensor([0]).double(), torch.tensor([1.0]).double()])

            k = torch.squeeze(torch.stack([row1, row2, row3]))

            plane = torch.inverse(k) @ torch.transpose(points_cam2.clone().reshape(-1, 3), 0, 1)
            #print(plane, " THE PLANE !!!!!!!!!!!!!!!!!!!!!!!!")
            #print(plane.shape, " pLane")

            plane = torch.vstack([plane, torch.ones((1, plane.shape[1]))])
            
            end2 = torch.transpose(torch.transpose(R_mat2, 0, 1) @ plane[:3, :], 0, 1)[:, :3] + torch.unsqueeze(T_matrix[cam2], dim = 0)
            #print(end2.shape, " END22222")
            if len(distortion_k_array) > 0:
                u2 = torch.transpose(points_cam2.clone().reshape(-1, 3), 0, 1)[0, :]
                v2 = torch.transpose(points_cam2.clone().reshape(-1, 3), 0, 1)[1, :]
                end2 = util.distortion_apply(u2, v2, k, R_mat2, T_matrix[cam2], camera_array_tan[cam2], camera_array_rad[cam2])

            end1_array.append(end1)
            end2_array.append(end2)

            start1 = T_matrix[cam1].repeat(end1.shape[0], 1) 
            start2 = T_matrix[cam2].repeat(end2.shape[0], 1) 

            p1,p2,d = bundle_intersect.closestDistanceBetweenLines(start1,end1,start2,end2)

            person1.append(p1)
            person2.append(p2)
            #print(p1.reshape(-1, len(keypoint_array) ,3).shape, " P11111111111111")

            hip_ratio_error, shoulder_ratio_error = bone_length.bone_length_constraint(p1.reshape(-1, len(keypoint_array) ,3), keypoint_dict)
            hip_ratio_error1, shoulder_ratio_error1 = bone_length.bone_length_constraint(p2.reshape(-1, len(keypoint_array) ,3), keypoint_dict)
            error = p1 - p2
            #print(torch.norm(error, dim=-1).shape, " torch error")

            #ankle keypoints should be on ground plane
            #p1_min = torch.min(p1.reshape(-1, len(keypoint_array) ,3)[:,r_ankle:,2], 1)[0]
            #p2_min = torch.min(p2.reshape(-1, len(keypoint_array) ,3)[:,l_ankle:,2], 1)[0]
            #print(p1.shape, p2.shape, " P1 P2 SHAPESSSSSSSSSSSSSSSSs")
            #print(p1.reshape(-1, len(keypoint_array) ,3).shape, p2.reshape(-1, len(keypoint_array) ,3).shape, " reshapesss")
            p1_min_r = p1.reshape(-1, len(keypoint_array) ,3)[:,r_ankle,2]
            p1_min_l = p1.reshape(-1, len(keypoint_array) ,3)[:,l_ankle,2]

            p2_min_r = p2.reshape(-1, len(keypoint_array) ,3)[:,r_ankle,2]
            p2_min_l = p2.reshape(-1, len(keypoint_array) ,3)[:,l_ankle,2]

            p1_min_full = torch.minimum(p1_min_r, p1_min_l)
            p2_min_full = torch.minimum(p2_min_r, p2_min_l)
            
            indices1 = torch.nonzero(torch.abs(p1_min_full) < 0.1).squeeze()
            indices2 = torch.nonzero(torch.abs(p2_min_full) < 0.1).squeeze()

            print(p1_min_full, " p1_min_full")
            print(p2_min_full, " p2_min_full")
            p1_min = p1_min_full[indices1]
            p2_min = p2_min_full[indices2]
            print("*************************")
            print(p1_min, "p1_min")
            print(p2_min, " p2_min")
            #p1_min = torch.min(p1.reshape(-1, 8 ,3)[:,6:,2], 1)[0]
            #p2_min = torch.min(p2.reshape(-1, 8 ,3)[:,7:,2], 1)[0]
            
            #print(p_min.shape, p1_min.shape, p2_min.shape)

            left_joints1 =  p1.reshape(-1, len(keypoint_array) ,3)[:, np.array(joint_array)[joint_l, 0],:]
            left_joints2 =  p1.reshape(-1, len(keypoint_array) ,3)[:, np.array(joint_array)[joint_l, 1],:]
            right_joints1 = p2.reshape(-1, len(keypoint_array) ,3)[:, np.array(joint_array)[joint_r, 0],:]
            right_joints2 = p2.reshape(-1, len(keypoint_array) ,3)[:, np.array(joint_array)[joint_r, 1],:]

            ##########################

            height_l1 =  p1.reshape(-1, len(keypoint_array), 3)[:, np.array(joint_array)[height_l, 0],:]
            height_l2 =  p1.reshape(-1, len(keypoint_array), 3)[:, np.array(joint_array)[height_l, 1],:]
            height_r1 = p2.reshape(-1, len(keypoint_array), 3)[:, np.array(joint_array)[height_r, 0],:]
            height_r2 = p2.reshape(-1, len(keypoint_array), 3)[:, np.array(joint_array)[height_r, 1],:]

            #print(left_joints1.shape, " left_joints1left_joints1left_joints1")

            left_joints_norm = torch.norm(left_joints1 - left_joints2, dim = 2).reshape(-1)
            right_joints_norm = torch.norm(right_joints1 - right_joints2, dim = 2).reshape(-1)

            #################################

            left_height_norm = torch.sum(torch.norm(height_l1 - height_l2, dim = 2), dim = 1)
            right_height_norm = torch.sum(torch.norm(height_r1 - height_r2, dim = 2), dim = 1)

            #print(left_joints_norm.shape, " left_joints_normleft_joints_normleft_joints_norm")
            ######################################
            #print("**************************************")
            
            shoulder1 = p1.reshape(-1, len(keypoint_array) ,3)[:, joint_array[0][0], :] - p1.reshape(-1, len(keypoint_array) ,3)[:, joint_array[0][1], :]
            shoulder2 = p2.reshape(-1, len(keypoint_array) ,3)[:, joint_array[0][0], :] - p2.reshape(-1, len(keypoint_array) ,3)[:, joint_array[0][1], :]
            #print(shoulder1.shape, " SHOUDELR 1111")


            hip1 = p1.reshape(-1, len(keypoint_array) ,3)[:, joint_array[3][0], :] - p1.reshape(-1, len(keypoint_array) ,3)[:, joint_array[3][1], :]
            hip2 = p2.reshape(-1, len(keypoint_array) ,3)[:, joint_array[3][0], :] - p2.reshape(-1, len(keypoint_array) ,3)[:, joint_array[3][1], :]

            #shoulder1.retain_grad()
            #shoulder1.retain_grad()

            shoulder_hip1 = torch.mean(torch.absolute(torch.norm(shoulder1, dim = 1) - torch.norm(hip1, dim = 1)))
            shoulder_hip2 = torch.mean(torch.absolute(torch.norm(shoulder2, dim = 1) - torch.norm(hip2, dim = 1)))

            hip1.retain_grad()
            hip2.retain_grad()

            shoulder_array.append(shoulder1)
            shoulder_array.append(shoulder2)

            hip_array.append(hip1)
            hip_array.append(hip2)
            #print(shoulder_hip1, shoulder_hip2, " shoulder hip  1111")
            #shoulder_hip1 + shoulder_hip2 + 

            pred_3d_array.append(p1)
            pred_3d_array.append(p2)

            #print(torch.mean(torch.norm(error, dim=-1)).detach().item(), torch.mean(torch.absolute(p_min)).detach().item(), torch.mean(torch.absolute(left_joints_norm - right_joints_norm)).detach().item(), torch.mean(torch.absolute(left_height_norm - h)).detach().item(), torch.mean(torch.absolute(right_height_norm - h)).detach().item(), torch.mean(hip_ratio_error).detach().item(), torch.mean(shoulder_ratio_error).detach().item(), " LOSS TERMS")
            #loss = torch.mean(torch.norm(error, dim=-1)) + w0*torch.mean(torch.absolute(p_min)) + w1*torch.mean(torch.absolute(left_joints_norm - right_joints_norm)) + w2*(torch.mean(torch.absolute(left_height_norm - h)) + torch.mean(torch.absolute(right_height_norm - h)))# + w3*(torch.mean(hip_ratio_error) + torch.mean(shoulder_ratio_error))
            
            print(torch.mean(torch.norm(error, dim=-1)).detach().item(), torch.mean(torch.absolute(p1_min)).detach().item(), torch.mean(torch.absolute(p2_min)).detach().item(), torch.mean(torch.absolute(left_joints_norm - right_joints_norm)).detach().item(), torch.mean(torch.absolute(left_height_norm - h)).detach().item(), torch.mean(torch.absolute(right_height_norm - h)).detach().item(), " LOSS TERMS")
            
            loss = torch.mean(torch.norm(error, dim=-1)) + w0*torch.mean(torch.absolute(p1_min)) +  w0*torch.mean(torch.absolute(p2_min)) + w1*torch.mean(torch.absolute(left_joints_norm - right_joints_norm)) + w2*(torch.mean(torch.absolute(left_height_norm - h)) + torch.mean(torch.absolute(right_height_norm - h)))# + w3*(torch.mean(hip_ratio_error) + torch.mean(shoulder_ratio_error)) + w4*(torch.mean(hip_ratio_error1) + torch.mean(shoulder_ratio_error1))
            
            print(loss, i, " LOSS")
            #print(loss)
            #stop
            #loss = w0*torch.mean(torch.norm(error, dim=-1)) + w1*torch.mean(torch.absolute(p_min)) + w3*torch.mean(torch.absolute(left_height_norm - h)) + w3*torch.mean(torch.absolute(right_height_norm - h)) + w2*torch.mean(hip_ratio_error) + w2*torch.mean(shoulder_ratio_error)
            #loss = w0*torch.mean(torch.norm(error, dim=-1)) + w1*torch.mean(torch.absolute(p1_min)) + w1*torch.mean(torch.absolute(p2_min)) + w2*torch.mean(torch.absolute(left_joints_norm - right_joints_norm)) + w3*torch.mean(torch.absolute(left_height_norm - h)) + w3*torch.mean(torch.absolute(right_height_norm - h))
            #loss = torch.mean(torch.norm(error, dim=-1)) + torch.mean(torch.absolute(p1_min)) + torch.mean(torch.absolute(p2_min)) + torch.mean(torch.absolute(left_height_norm - h)) + torch.mean(torch.absolute(right_height_norm - h))
            #print(loss, i, " LOSS")

            loss.retain_grad() 
            loss_array.append(loss)
            
            T_matrix.retain_grad()
            R_matrix.retain_grad()
            f_matrix.retain_grad()

            #loss.backward(retain_graph=True)           
            #loss.backward(retain_graph=True)

            #print(loss, " THE LOSS")
            #print(R_matrix.grad, "R matrix GRADSSSS")
            #print(f_matrix.grad, "f matrix GRADSSSS")
            error_array.append(loss.detach().numpy().item())
            den = den + 1
            itr = itr + 1

        #print(torch.stack(pred_3d_array).shape, " STACKKKKKKKK")
        
        mean_pose = torch.mean(torch.stack(pred_3d_array), dim = 0)

        shoulder_error_array = [] 
        hip_error_array = []
        #print(shoulder_array)
        #print(len(shoulder_array), " len shoulder")
        #stop
        
        shoulder_combs = util.random_combination(list(range(len(shoulder_array))), 2, np.inf)
        for ca in shoulder_combs:
            shoulder_error_array.append(torch.absolute(torch.norm(shoulder_array[ca[0]] - shoulder_array[ca[1]], dim = 1)))
            hip_error_array.append(torch.norm(hip_array[ca[0]] - hip_array[ca[1]], dim = 1))

        shoulder_error = torch.mean(torch.stack(shoulder_error_array).reshape(-1))
        hip_error = torch.mean(torch.stack(hip_error_array).reshape(-1))
        
        dist_error = []
        #print(pred_3d_array)
        for pr in pred_3d_array:
            #print(pr.shape, mean_pose.shape, " EMAN POSE")
            dist_error.append(torch.norm(pr - mean_pose, dim = 1))

        dist_mean_error_array = torch.stack(dist_error).reshape(-1)
        dist_mean_error = torch.mean(dist_mean_error_array)

        #################
        shoulder_array.append(shoulder1)
        shoulder_array.append(shoulder2)

        hip_array.append(hip1)
        hip_array.append(hip2)
        
        #################
        
        loss_error = torch.mean(torch.stack(loss_array))
        
        #loss.backward(retain_graph=True)
        #print(loss_error, dist_mean_error, shoulder_error, hip_error)
        loss = loss_error# + w4*dist_mean_error #+ 10*(shoulder_error + hip_error)
        print(loss, " LOSSSSS")
        loss.backward(retain_graph=True)
        error_array.append(loss.detach().numpy().item())
        '''
        print(dist_mean_error, " dist_mean_error")
        print(loss_error, " loss error")
        print(loss_array, "loss error array")
        '''
        #print(mean_pose, " MEAN POSE")
        #comb_array = util.random_combination(list(range(0, len(shoulder_array))), 2, np.inf)
        
        #shoulder_error = 0 
        #hip_error = 0
        #for ca in comb_array:
        #    shoulder_error =  shoulder_error + torch.absolute(shoulder_array[ca[0]] - shoulder_array[ca[1]])
        #    hip_error = hip_error + torch.absolute(hip_array[ca[0]] - hip_array[ca[1]])

        #shoulder_error = shoulder_error/len(shoulder_array)
        #hip_error = hip_error/len(shoulder_array)
        
        #loss = loss_array/den# + 0.1*shoulder_error/len(shoulder_array) + 0.1*hip_error/len(shoulder_array)
        #print(loss.detach().numpy().item(), itr, " Error in bundle adjustment")
        #loss.backward(retain_graph=True)
        f_array = []
        r_array = []
        t_array = []



        for p in range(R_matrix.shape[0]):
            R_new = so3_exponential_map(R_matrix)[p].detach().numpy()
            T_new = T_matrix[p].detach().numpy()
            #T_new = T_matrix[p].numpy()
            f_new = f_matrix[p].detach().numpy()
            p_new = p_matrix[p].detach().numpy()
            
            intrinsic_matrix = np.array([[f_new[0], 0, p_new[0]], [0, f_new[1], p_new[1]], [0 , 0, 1]])
            r_array.append(R_new)
            t_array.append(T_new)
            f_array.append(intrinsic_matrix)

        f_param_array.append(f_array)
        r_matrix_array.append(r_array)
        t_matrix_array.append(t_array)

        optimizer.step()

        if i == 0 or i == iteration - 1:
            #################################################################
            r_array = []
            for r in range(R_matrix.shape[0]):
                R_mat1 = so3_exponential_map(R_matrix)[r].detach().double()
                r_array.append(R_mat1)
            #print("AAAAAAAAAAAAAAAAAAAAAAAAAAA")
            #print(torch.unsqueeze(T_matrix.detach(), dim = 0).double(), r_array, torch.unsqueeze(torch.tensor(camera_translation_gt), dim = 0).double(), camera_rotation_gt, " SAAAAD")
            #cam_position, cam_axis_rotate, R1, translation_template_rotate, translation_rotate = metrics.procrustes_rotation_translation_template(torch.unsqueeze(T_matrix.detach(), dim = 0).double(), r_array, torch.unsqueeze(torch.tensor(camera_translation_gt), dim = 0).double(), camera_rotation_gt, use_reflection=False, use_scaling=True)
            #################################################################
            
            cam_position = T_matrix.detach().double()
            cam_axis_rotate = r_array
            for cam1 in range(cam_position.shape[0]):
                print(cam_axis_rotate[cam1], " CAM AXIS")
                print(cam_position, " THE POSITION")
                print(cam_position[cam1], " CAM POSITION")
                #stop
                #R_mat1 = torch.tensor(cam_axis_rotate[cam1]).double()#so3_exponential_map(R_matrix)[cam1].detach()

                #T_mat1 = torch.tensor(cam_position[cam1]).double()#T_matrix[cam1].detach()
                
                R_mat1 = so3_exponential_map(R_matrix)[cam1].detach()

                T_mat1 = T_matrix[cam1].detach()
                scale = 1.0

                trace2 = go.Scatter3d(
                    x=[T_mat1[0].numpy(),T_mat1[0].numpy() + scale*R_mat1[0][0].numpy()],
                    y=[T_mat1[1].numpy(),T_mat1[1].numpy() + scale*R_mat1[0][1].numpy()],
                    z=[T_mat1[2].numpy(),T_mat1[2].numpy() + scale*R_mat1[0][2].numpy()],
                    mode='lines',
                    name= " axis1 cam " + str(cam1),
                    marker=dict(
                        color='red',
                    )
                )

                trace3 = go.Scatter3d(
                    x=[T_mat1[0].numpy(),T_mat1[0].numpy() + scale*R_mat1[1][0].numpy()],
                    y=[T_mat1[1].numpy(),T_mat1[1].numpy() + scale*R_mat1[1][1].numpy()],
                    z=[T_mat1[2].numpy(),T_mat1[2].numpy() + scale*R_mat1[1][2].numpy()],
                    mode='lines',
                    name= " axis2 cam " + str(cam1),
                    marker=dict(
                        color='green',
                    )
                )

                trace4 = go.Scatter3d(
                    x=[T_mat1[0].numpy(),T_mat1[0].detach().numpy() + scale*R_mat1[2][0].numpy()],
                    y=[T_mat1[1].numpy(),T_mat1[1].detach().numpy() + scale*R_mat1[2][1].numpy()],
                    z=[T_mat1[2].numpy(),T_mat1[2].detach().numpy() + scale*R_mat1[2][2].numpy()],
                    mode='lines',
                    name=  " axis3 cam " + str(cam1),
                    marker=dict(
                        color='blue',
                    )
                )

                trace5 = go.Scatter3d(
                    x=[T_mat1[0].numpy()],
                    y=[T_mat1[1].numpy()],
                    z=[T_mat1[2].numpy()],
                    mode='markers+text',
                    name =  " " ,
                    marker=dict(
                        color='black',
                        size = 0.1,
                    ),
                    text = str(cam1),
                    textposition='top right',
                    textfont=dict(color='black')
                )

                data.append(trace2)
                data.append(trace3)
                data.append(trace4)
                data.append(trace5)
            
            #for match_dict in matched_points:
            for comb_dict in comb_stack: 

                person1 = []
                person2 = []

                cam1 = list(comb_dict.keys())[0]
                cam2 = list(comb_dict.keys())[1]
                #cam1 = list(match_dict.keys())[0] 
                #cam2 = list(match_dict.keys())[1] 

                #points_cam1 = comb_dict[cam1].reshape((-1,3))#torch.tensor(list(match_dict[ca[0]]['pose'].values())[:-1]).double()
                #points_cam2 = comb_dict[cam2].reshape((-1,3))#torch.tensor(list(match_dict[ca[1]]['pose'].values())[:-1]).double()

                points_cam1 = comb_dict[cam1].reshape((-1,3))#torch.tensor(list(match_dict[ca[0]]['pose'].values())[:-1]).double()
                points_cam2 = comb_dict[cam2].reshape((-1,3))#torch.tensor(list(match_dict[ca[1]]['pose'].values())[:-1]).double()

                #training_set = Dataset(match_dict[cam1], match_dict[cam2])
                #training_generator = torch.utils.data.DataLoader(training_set, **params)

                end1_array = []
                end2_array = []
                
                R_mat1 = torch.tensor(cam_axis_rotate[cam1])
                R_mat2 = torch.tensor(cam_axis_rotate[cam2])

                T_mat1 = torch.tensor(cam_position[cam1])
                T_mat2 = torch.tensor(cam_position[cam2])
                #stop 
        
                #for points_cam1, points_cam2 in training_generator:
                    
                #points_cam1 = points_cam1[:,:, key_point_select, :]
                #points_cam2 = points_cam2[:,:, key_point_select, :]

                ax1.scatter(points_cam1.clone().reshape(-1, 3)[:, 0], points_cam1.clone().reshape(-1, 3)[:, 1], c = 'r')
                ax1.scatter(points_cam2.clone().reshape(-1, 3)[:, 0], points_cam2.clone().reshape(-1, 3)[:, 1], c = 'b')
                

                row1 = torch.stack([f_matrix[cam1][[0]], torch.tensor([0]).double(), p_matrix[cam1][[0]]]).double()
                row2 = torch.stack([torch.tensor([0]).double(), f_matrix[cam1][[1]], p_matrix[cam1][[1]]]).double()
                row3 = torch.stack([torch.tensor([0]).double(), torch.tensor([0]).double(), torch.tensor([1.0]).double()])

                k = torch.squeeze(torch.stack([row1, row2, row3]))

                plane = torch.inverse(k) @ torch.transpose(points_cam1.clone().reshape(-1, 3), 0, 1)

                plane = torch.vstack([plane, torch.ones((1, plane.shape[1]))])

                vec1 = torch.transpose(torch.transpose(R_mat1, 0, 1) @ plane[:3, :], 0, 1)[:, :3]
                norms1 = torch.norm(vec1, dim=1)
                #print(norms1.shape, " NORMS 1")
                
                # Normalize the tensor by dividing each vector by its norm
                vec1_normal = torch.div(vec1, norms1.view(-1, 1))
                end1 = vec1_normal + torch.unsqueeze(T_mat1, dim = 0)
                #end1 =  torch.transpose(torch.transpose(R_mat1, 0, 1) @ plane[:3, :], 0, 1)[:, :3] + torch.unsqueeze(T_mat1, dim = 0)
                
                if len(distortion_k_array) > 0:
                    u1 = torch.transpose(points_cam1.clone().reshape(-1, 3), 0, 1)[0, :]
                    v1 = torch.transpose(points_cam1.clone().reshape(-1, 3), 0, 1)[1, :]
                    end1 = util.distortion_apply(u1, v1, k, R_mat1,T_mat1, camera_array_tan[cam1], camera_array_rad[cam1])

                row1 = torch.stack([f_matrix[cam2][[0]], torch.tensor([0]).double(), p_matrix[cam2][[0]]]).double()
                row2 = torch.stack([torch.tensor([0]).double(), f_matrix[cam2][[1]], p_matrix[cam2][[1]]]).double()
                row3 = torch.stack([torch.tensor([0]).double(), torch.tensor([0]).double(), torch.tensor([1.0]).double()])

                k = torch.squeeze(torch.stack([row1, row2, row3]))

                plane = torch.inverse(k) @ torch.transpose(points_cam2.clone().reshape(-1, 3), 0, 1)

                plane = torch.vstack([plane, torch.ones((1, plane.shape[1]))])
                
                vec2 = torch.transpose(torch.transpose(R_mat2, 0, 1) @ plane[:3, :], 0, 1)[:, :3]
                norms2 = torch.norm(vec2, dim=1)
                #print(norms1.shape, " NORMS 1")
                
                # Normalize the tensor by dividing each vector by its norm
                vec2_normal = torch.div(vec2, norms1.view(-1, 1))
                end2 = vec2_normal + torch.unsqueeze(T_mat2, dim = 0)
                #end2 = torch.transpose(torch.transpose(R_mat2, 0, 1) @ plane[:3, :], 0, 1)[:, :3] + torch.unsqueeze(T_mat2, dim = 0)
                if len(distortion_k_array) > 0:
                    u2 = torch.transpose(points_cam2.clone().reshape(-1, 3), 0, 1)[0, :]
                    v2 = torch.transpose(points_cam2.clone().reshape(-1, 3), 0, 1)[1, :]
                    end2 = util.distortion_apply(u2, v2, k, R_mat2, T_mat2, camera_array_tan[cam2], camera_array_rad[cam2])

                end1_array.append(end1)
                end2_array.append(end2)

                start1 = T_mat1.repeat(end1.shape[0], 1) 
                start2 = T_mat2.repeat(end2.shape[0], 1) 
                p1,p2,d = bundle_intersect.closestDistanceBetweenLines(start1,end1,start2,end2)

                person1.append(p1)
                person2.append(p2)

                p1_min = torch.min(p1.reshape(-1, len(keypoint_array) ,3)[:,:,2], 1)[0]
                p2_min = torch.min(p2.reshape(-1, len(keypoint_array) ,3)[:,:,2], 1)[0]
    

                ################
                person1_stack = torch.cat(person1, dim = 0).reshape(-1, len(keypoint_array), 3)
                person2_stack = torch.cat(person2, dim = 0).reshape(-1, len(keypoint_array), 3)

                for ps in [0]:
                #for ps in range(20):

                    for j in range(8):

                        trace_dash = go.Scatter3d(
                                x=[T_mat1[0].numpy(), person1_stack[ps ,j, 0].detach().numpy()],
                                y=[T_mat1[1].numpy(), person1_stack[ps, j, 1].detach().numpy()],
                                z=[T_mat1[2].numpy(), person1_stack[ps, j, 2].detach().numpy()],
                                mode='lines',
                                line={'dash': 'dash'},
                                name= 'view line ' + str(ps) + ' ' + str(cam1),
                                marker=dict(
                                        color='red',
                                    )
                            )
                        data.append(trace_dash)
                            
                        trace_dash = go.Scatter3d(
                                x=[T_mat2[0].numpy(), person2_stack[ps ,j, 0].detach().numpy()],
                                y=[T_mat2[1].numpy(), person2_stack[ps, j, 1].detach().numpy()],
                                z=[T_mat2[2].numpy(), person2_stack[ps, j, 2].detach().numpy()],
                                mode='lines',
                                line={'dash': 'dash'},
                                name= 'view line ' + str(ps) + ' ' + str(cam2),
                                marker=dict(
                                        color='blue',
                                    )
                            )
                        data.append(trace_dash)
                    
                    trace1 = go.Scatter3d(
                        x=[person1_stack[ps, 0, 0].detach().numpy()],
                        y=[person1_stack[ps, 0, 1].detach().numpy()],
                        z=[person1_stack[ps, 0, 2].detach().numpy()],
                        mode='markers+text',
                        #hovertext='HEAD1 ' + str(ps) + ' ' + str(cam1) + ' ' + str(cam2),
                        #hoverinfo="text",
                        name= 'Nose1 ' + str(cam1) + ' ' + str(cam2),
                        marker=dict(
                                color='red',
                                size=1,
                            ),
                        text = 'Nose1 ' + str(ps) + ' ' + str(cam1) + ' ' + str(cam2),
                        textposition='top right',
                        textfont=dict(color='black')
                    )

                    data.append(trace1)

                    trace2 = go.Scatter3d(
                        x=[person2_stack[ps, 0, 0].detach().numpy()],
                        y=[person2_stack[ps, 0, 1].detach().numpy()],
                        z=[person2_stack[ps, 0, 2].detach().numpy()],
                        mode='markers+text',
                        #hovertext='HEAD2 ' + str(ps) + ' ' + str(cam1) + ' ' + str(cam2),
                        #hoverinfo="text",
                        name= 'Nose2 ' + str(cam1) + ' ' + str(cam2),
                        marker=dict(
                                color='blue',
                                size=1,
                            ),
                        text = 'Nose2 ' + str(ps) + ' ' + str(cam1) + ' ' + str(cam2),
                        textposition='top right',
                        textfont=dict(color='black')
                    )
                    data.append(trace2)
                
                    ###############################
                    init1 = end1.reshape(-1, len(keypoint_array) ,3)
                    init2 = end2.reshape(-1, len(keypoint_array) ,3)
                    for j in joint_array:

                        trace1 = go.Scatter3d(
                            x=[person1_stack[ps ,j[0], 0].detach().numpy(), person1_stack[ps, j[1], 0].detach().numpy()],
                            y=[person1_stack[ps, j[0], 1].detach().numpy(), person1_stack[ps, j[1], 1].detach().numpy()],
                            z=[person1_stack[ps, j[0], 2].detach().numpy(), person1_stack[ps, j[1], 2].detach().numpy()],
                            mode='lines',
                            name= 'PL1 ' + str(ps) + ' ' + str(cam1) + ' ' + str(cam2),
                            marker=dict(
                                    color='red',
                                )
                        )

                        data.append(trace1)

                        trace2 = go.Scatter3d(
                            x=[person2_stack[ps, j[0], 0].detach().numpy(), person2_stack[ps, j[1], 0].detach().numpy()],
                            y=[person2_stack[ps, j[0], 1].detach().numpy(), person2_stack[ps, j[1], 1].detach().numpy()],
                            z=[person2_stack[ps, j[0], 2].detach().numpy(), person2_stack[ps, j[1], 2].detach().numpy()],
                            mode='lines',
                            name= 'PL2 ' + str(ps) + ' ' + str(cam1) + ' ' + str(cam2),
                            marker=dict(
                                    color='blue',
                                )
                        )
                        data.append(trace2)

                        ###############################################

                        trace1 = go.Scatter3d(
                            x=[init1[ps ,j[0], 0].detach().numpy(), init1[ps, j[1], 0].detach().numpy()],
                            y=[init1[ps, j[0], 1].detach().numpy(), init1[ps, j[1], 1].detach().numpy()],
                            z=[init1[ps, j[0], 2].detach().numpy(), init1[ps, j[1], 2].detach().numpy()],
                            mode='lines',
                            name= 'START ' + str(ps) + ' ' + str(cam1) + ' ' + str(cam2),
                            marker=dict(
                                    color='red',
                                )
                        )

                        data.append(trace1)

                        trace2 = go.Scatter3d(
                            x=[init2[ps, j[0], 0].detach().numpy(), init2[ps, j[1], 0].detach().numpy()],
                            y=[init2[ps, j[0], 1].detach().numpy(), init2[ps, j[1], 1].detach().numpy()],
                            z=[init2[ps, j[0], 2].detach().numpy(), init2[ps, j[1], 2].detach().numpy()],
                            mode='lines',
                            name= 'START ' + str(ps) + ' ' + str(cam1) + ' ' + str(cam2),
                            marker=dict(
                                    color='blue',
                                )
                        )
                        data.append(trace2)

                #break
                #if itr%100 == 0:
                #    break
                #itr = itr + 1
            
            scale = 1.0
            
            for comb_dict in comb_stack: 

                person1 = []
                person2 = []

                cam1 = list(comb_dict.keys())[0]
                cam2 = list(comb_dict.keys())[1]
                #cam1 = list(match_dict.keys())[0] 
                #cam2 = list(match_dict.keys())[1] 

                #points_cam1 = comb_dict[cam1].reshape((-1,3))#torch.tensor(list(match_dict[ca[0]]['pose'].values())[:-1]).double()
                #points_cam2 = comb_dict[cam2].reshape((-1,3))#torch.tensor(list(match_dict[ca[1]]['pose'].values())[:-1]).double()

                points_cam1 = comb_dict[cam1].reshape((-1,3))#torch.tensor(list(match_dict[ca[0]]['pose'].values())[:-1]).double()
                points_cam2 = comb_dict[cam2].reshape((-1,3))#torch.tensor(list(match_dict[ca[1]]['pose'].values())[:-1]).double()
                    
                person1 = []
                person2 = []

                #cam1 = list(match_dict.keys())[0] 
                #cam2 = list(match_dict.keys())[1] 

                #training_set = Dataset(match_dict[cam1], match_dict[cam2])
                #training_generator = torch.utils.data.DataLoader(training_set, **params)

                end1_array = []
                end2_array = []
                #translation_rotate
                print(camera_rotation_gt[cam1], " CAM 111",cam1)
                print(camera_rotation_gt[cam2], " CAM @@@@",cam2)
                '''
                R_mat1 = torch.tensor(camera_rotation_gt[cam1]).double()#so3_exponential_map(R_matrix)[cam1].detach()
                R_mat2 = torch.tensor(camera_rotation_gt[cam2]).double()#so3_exponential_map(R_matrix)[cam2].detach()

                T_mat1 = torch.tensor(camera_translation_gt[cam1]).double()#T_matrix[cam1].detach()
                T_mat2 = torch.tensor(camera_translation_gt[cam2]).double()#T_matrix[cam2].detach()
                '''

                R_mat1 = so3_exponential_map(R_matrix)[cam1].detach()
                R_mat2 = so3_exponential_map(R_matrix)[cam2].detach()

                T_mat1 = T_matrix[cam1].detach()
                T_mat2 = T_matrix[cam2].detach()
                k1 = torch.tensor(single_view_array_gt[cam1]).double()
                k2 = torch.tensor(single_view_array_gt[cam2]).double()

                print(camera_translation_gt[cam1], " t 111",cam1)
                print(camera_translation_gt[cam2], " t @@@@",cam2)
                #stop
                scale = 1.0
                #for points_cam1, points_cam2 in training_generator:
                    
                #points_cam1 = points_cam1[:,:, key_point_select, :]
                #points_cam2 = points_cam2[:,:, key_point_select, :]
            
                ax1.scatter(points_cam1.clone().reshape(-1, 3)[:, 0], points_cam1.clone().reshape(-1, 3)[:, 1], c = 'r')
                ax1.scatter(points_cam2.clone().reshape(-1, 3)[:, 0], points_cam2.clone().reshape(-1, 3)[:, 1], c = 'b')
                

                plane = torch.inverse(k1) @ torch.transpose(points_cam1.clone().reshape(-1, 3), 0, 1)

                plane = torch.vstack([plane, torch.ones((1, plane.shape[1]))])

                #####################
                # Calculate the norm of each vector
                vec1 = torch.transpose(torch.transpose(R_mat1, 0, 1) @ plane[:3, :], 0, 1)[:, :3]
                norms1 = torch.norm(vec1, dim=1)
                #print(norms1.shape, " NORMS 1")
                
                # Normalize the tensor by dividing each vector by its norm
                vec1_normal = torch.div(vec1, norms1.view(-1, 1))
                #print(torch.norm(vec1_normal, dim=1), " vec1_normal")
                
                #####################
                end1 =  vec1_normal + torch.unsqueeze(T_mat1, dim = 0)
                '''
                if len(distortion_k_array) > 0:
                    u1 = torch.transpose(points_cam1.clone().reshape(-1, 3), 0, 1)[0, :]
                    v1 = torch.transpose(points_cam1.clone().reshape(-1, 3), 0, 1)[1, :]
                    end1 = util.distortion_apply(u1, v1, k1, R_mat1,T_mat1, camera_array_tan[cam1], camera_array_rad[cam1])
                '''
                plane = torch.inverse(k2) @ torch.transpose(points_cam2.clone().reshape(-1, 3), 0, 1)

                plane = torch.vstack([plane, torch.ones((1, plane.shape[1]))])
                
                #####################
                # Calculate the norm of each vector
                vec2 = torch.transpose(torch.transpose(R_mat2, 0, 1) @ plane[:3, :], 0, 1)[:, :3]
                norms2 = torch.norm(vec2, dim=1)

                # Normalize the tensor by dividing each vector by its norm
                vec2_normal = torch.div(vec2, norms2.view(-1, 1))
                #print(torch.norm(vec2_normal, dim=1), " vec1_normal")
                #####################
                end2 = vec2_normal + torch.unsqueeze(T_mat2, dim = 0)
                
                #stop
                '''
                if len(distortion_k_array) > 0:
                    u2 = torch.transpose(points_cam2.clone().reshape(-1, 3), 0, 1)[0, :]
                    v2 = torch.transpose(points_cam2.clone().reshape(-1, 3), 0, 1)[1, :]
                    end2 = util.distortion_apply(u2, v2, k2, R_mat2, T_mat2, camera_array_tan[cam2], camera_array_rad[cam2])
                '''
                end1_array.append(end1)
                end2_array.append(end2)

                start1 = T_mat1.repeat(end1.shape[0], 1) 
                start2 = T_mat2.repeat(end2.shape[0], 1) 
                p1,p2,d = bundle_intersect.closestDistanceBetweenLines(start1,end1,start2,end2)

                person1.append(p1)
                person2.append(p2)

                p1_min = torch.min(p1.reshape(-1, len(keypoint_array) ,3)[:,:,2], 1)[0]
                p2_min = torch.min(p2.reshape(-1, len(keypoint_array) ,3)[:,:,2], 1)[0]

                ################
                person1_stack = torch.cat(person1, dim = 0).reshape(-1, len(keypoint_array), 3)
                person2_stack = torch.cat(person2, dim = 0).reshape(-1, len(keypoint_array), 3)
                ######################################
                    
        
        #################################################################
        #################################################################
        if plot_true == True:
            if i == 0 or i == iteration - 1:
                fig = go.Figure(data=data)

                fig.update_layout(scene = dict( aspectmode='data'))

                fig.update_layout(title_text='reconstruction')
                #fig.show()
                fig.write_html(save_dir + '/reconstruction_' + run_name + str(i) + '_.html')

                fig.data = []

                del fig
                
        print(" GOT TO OPTIMIZER STEP")
        average_error = np.mean(error_array)
        avg_error_array.append(average_error)
        print(average_error, " average_error")

        fig1, ax1 = plt.subplots(1, 1)
        ax1.set_yscale('log')
        ax1.plot(avg_error_array)
        fig1.savefig(save_dir + '/bundle_iteration' + run_name + '.png')
        plt.close('all')
        #print(so3_exponential_map(R_matrix)[0], " R_MATRIX1")
        #print(so3_exponential_map(R_matrix)[1], " R_MATRIX2")
        #print(torch.inverse(so3_exponential_map(R_matrix)[0]), " R_MATRIX1 inv")
        #print(torch.inverse(so3_exponential_map(R_matrix)[1]), " R_MATRIX2 inv")

        #optimizer.step()

    min_ind = np.argmin(avg_error_array)
    intrinsic_matrix_array = f_param_array[min_ind]
    rotation_matrix_array = r_matrix_array[min_ind]
    position_matrix_array = t_matrix_array[min_ind]
    '''
    rotation_matrix_array = []
    position_matrix_array = []
    intrinsic_matrix_array = []
    for p in range(R_matrix.shape[0]):
        R_new = so3_exponential_map(R_matrix)[p].detach().numpy()
        T_new = T_matrix[p].detach().numpy()
        f_new = f_matrix[p].detach().numpy()
        p_new = p_matrix[p].detach().numpy()
        
        intrinsic_matrix = np.array([[f_new[0], 0, p_new[0]], [0, f_new[1], p_new[1]], [0 , 0, 1]])
        rotation_matrix_array.append(R_new)
        position_matrix_array.append(T_new)
        intrinsic_matrix_array.append(intrinsic_matrix)
    '''
    #print("**&&A&DASDSDASD")
    #print(rotation_matrix_array, " rotation_matrix_array")
    #print(position_matrix_array, " position_matrix_array")
    return rotation_matrix_array, position_matrix_array, intrinsic_matrix_array

def match_3d_multiview(sync_dict_array, peturb_extrinsics_array, peturb_single_view_array, all_points_array,center_array, run_name, k = 5, scale = 1000, title = 'plot', name = ['pred'], frame_init = 0 ,frame_end = 500):
    
    #draw a square

    cam_amount = len(all_points_array)

    point_array = []

    ankle_array = []

    distance_hungarian_array = []

    #indices_array = util.random_combination(list(range(len(all_points_array))), 2, np.inf)
    indices_array = []

    for i in range(1, len(all_points_array)):
        indices_array.append((0, i))
    indices = []
    for ind in indices_array:
        i1 = ind[0]
        i2 = ind[1]

        #######################################
        init_ref_shift_matrix = peturb_extrinsics_array[i1]['init_sync_center_array']
        icp_rot_matrix_ref = peturb_extrinsics_array[i1]['icp_rot_array']
        init_rot_matrix_ref = peturb_extrinsics_array[i1]['icp_init_rot_array']
        plane_matrix_array_ref = peturb_extrinsics_array[i1]['plane_matrix_array'] 

        cam_axis_ref = np.transpose(icp_rot_matrix_ref @ init_rot_matrix_ref @ plane_matrix_array_ref[:3,:3])

        init_sync_shift_matrix = peturb_extrinsics_array[i2]['init_sync_center_array']
        icp_rot_matrix_sync = peturb_extrinsics_array[i2]['icp_rot_array']
        init_rot_matrix_sync = peturb_extrinsics_array[i2]['icp_init_rot_array']
        plane_matrix_array_sync = peturb_extrinsics_array[i2]['plane_matrix_array']

        cam_axis_sync = np.transpose(icp_rot_matrix_sync @ init_rot_matrix_sync @ plane_matrix_array_sync[:3,:3])
        indices.append(ind)
    
    hungarian_inds_array = []
    for ind in indices:

        i1 = ind[0]
        i2 = ind[1]
        index_dict = {i1: [], i2: []}
        ankle_dict = {i1: [], i2: []}

        distance_hungarian = []

        view_frames1 = list(sync_dict_array[i2 - 1].keys())
        view_frames2 = list(sync_dict_array[i2 - 1].values())
        #print(view_frames1, " frame1")
        #print(view_frames2, " frame2")

        hungarian_inds = []    
        #for fr in range(len(list(all_points_array[i1].keys()))):  
        for fr in range(len(view_frames1)):  

            fr1 = view_frames1[fr]
            fr2 = view_frames2[fr]
    
            if fr1 not in all_points_array[i1].keys():
                continue
            
            if fr2 not in all_points_array[i2].keys():
                continue
            print("HELLO DID WE GET HERE????")
            #######################################
            init_ref_shift_matrix = peturb_extrinsics_array[i1]['init_sync_center_array']
            icp_rot_matrix_ref = peturb_extrinsics_array[i1]['icp_rot_array']
            init_rot_matrix_ref = peturb_extrinsics_array[i1]['icp_init_rot_array']
            plane_matrix_array_ref = peturb_extrinsics_array[i1]['plane_matrix_array'] 

            init_sync_shift_matrix = peturb_extrinsics_array[i2]['init_sync_center_array']
            icp_rot_matrix_sync = peturb_extrinsics_array[i2]['icp_rot_array']
            init_rot_matrix_sync = peturb_extrinsics_array[i2]['icp_init_rot_array']
            plane_matrix_array_sync = peturb_extrinsics_array[i2]['plane_matrix_array']

            #######################################
            points_array_ref = []
            poses_array_ref = []
            for c in all_points_array[i1][fr1].keys():
                
                plane1 = util.plane_ray_intersection_np([np.array(list(all_points_array[i1][fr1][c]))[0]], [np.array(list(all_points_array[i1][fr1][c]))[1]], np.linalg.inv(peturb_single_view_array[i1]['cam_matrix']),  peturb_single_view_array[i1]["ground_normal"], peturb_single_view_array[i1]['ground_position'])
                plane1 = np.transpose(np.r_[plane1, np.ones((1, plane1.shape[1]))])
                transformed_ref = np.transpose(icp_rot_matrix_ref @ init_rot_matrix_ref @ np.transpose(np.transpose(plane_matrix_array_ref @ np.transpose(np.array(plane1))) - np.array([[init_ref_shift_matrix[0], init_ref_shift_matrix[1], 0, 0]]))[:3, :])
                
                points_array_ref.append(transformed_ref)
                #CHECK THAT THE POSES ARE MATCHING!!!
                poses_array_ref.append(np.array(list(all_points_array[i1][fr1][c]))[4])

            points_array_sync = []
            poses_array_sync = []
            for c in all_points_array[i2][fr2].keys():
                
                plane1 = util.plane_ray_intersection_np([np.array(list(all_points_array[i2][fr2][c]))[0]], [np.array(list(all_points_array[i2][fr2][c]))[1]], np.linalg.inv(peturb_single_view_array[i2]['cam_matrix']),  peturb_single_view_array[i2]["ground_normal"], peturb_single_view_array[i2]['ground_position'])
                plane1 = np.transpose(np.r_[plane1, np.ones((1, plane1.shape[1]))])
                transformed_sync = np.transpose(icp_rot_matrix_sync @ init_rot_matrix_sync @ np.transpose(np.transpose(plane_matrix_array_sync @ np.transpose(np.array(plane1))) - np.array([[init_sync_shift_matrix[0], init_sync_shift_matrix[1], 0, 0]]))[:3, :])
                
                points_array_sync.append(transformed_sync)
                poses_array_sync.append(np.array(list(all_points_array[i2][fr2][c]))[4])

            points_ref = np.concatenate(points_array_ref)
            points_sync = np.concatenate(points_array_sync)

            row_ind, col_ind = util.hungarian_assignment_one2one(points_ref, points_sync)
            
            hungarian_dist = np.linalg.norm(points_ref[row_ind] - points_sync[col_ind], axis = 1)

            hungarian_inds.append([i1, i2, fr1, fr2, row_ind, col_ind, hungarian_dist])
            '''
            print(row_ind, " row ind")
            print(col_ind, " col ind")
            for r_ind in range(len(row_ind)):
            
                sync_ind = col_ind[r_ind]
                ref_ind = row_ind[r_ind]
                index_dict[i1].append(torch.tensor(list(poses_array_ref[ref_ind].values())[:-1]))
                index_dict[i2].append(torch.tensor(list(poses_array_sync[sync_ind].values())[:-1]))

                ankle_dict[i1].append(torch.from_numpy(points_ref[ref_ind]))
                ankle_dict[i2].append(torch.from_numpy(points_sync[sync_ind]))

                distance_hungarian.append(np.linalg.norm(points_ref[ref_ind] - points_sync[sync_ind]))
            '''

        hungarian_inds_array.append(hungarian_inds)
        '''
        distance_hungarian_array.append(distance_hungarian)
        point_array.append(index_dict)
        ankle_array.append(ankle_dict)
        '''
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #print(hungarian_inds_array[0][:5])
    #print("*************************")
    #print(hungarian_inds_array[1][:5])

    multiview_dict = {}

    for hd in range(len(hungarian_inds_array)):
        for hd1 in range(len(hungarian_inds_array[hd])):

            match = hungarian_inds_array[hd][hd1]

            if match[2] not in multiview_dict:
                multiview_dict[match[2]] = [{"name0": match[0], "name1": match[1], "frame": match[3], "set0": match[4], "set1": match[5], "dist": match[6]}]
            else:
                multiview_dict[match[2]].append({"name0": match[0], "name1": match[1], "frame": match[3], "set0": match[4], "set1": match[5], "dist": match[6]})
    
    multiview_dict_all = util.match_pairs_graph_per_frame(multiview_dict)
    #print(multiview_dict_all)
    #stop
    dict_array = []
    dist_array = []

    for mda in multiview_dict_all:
        
        #print(mda, " THE MDA")\
        #print(all_points_array)
        #print("***ASD")
        #print(mda.keys())

        match_poses = []

        hungarian_average = []
        
        conf_array_average = []
        if len(list(mda.keys())) < cam_amount:
            continue
        for cn in list(mda.keys()):
            ind_match = mda[cn][1]
            frame = mda[cn][0]

            hungarian_dist = mda[cn][2]
            #print(mda, " mda")
            #print(mda[cn], " mda[cn] ")
            #print(all_points_array[cn].keys())
            ind_key = list(all_points_array[cn][frame].keys())
            
            #print(cn, " THIS IS THE CN")
            #print(all_points_array[cn])
            #print("********************************")
            #print(all_points_array[cn][frame])
            #print("********************************")
            #print(ind_key, " ind kety")
            pose = all_points_array[cn][frame][ind_key[ind_match]]

            pose_array = np.array(list(pose[4].values())[:-1])
            '''
            print(pose)
            conf_array = []
            for ps in pose_array:
                print(ps, " THE PS")

                conf_array.append(ps[2])
            
            if np.mean(conf_array) < 0.75:
                continue 
            '''
            #print(pose)
            #stop
            pose_array[:, 2] = 1.0
            #print(pose_array.shape, " THE SHAEEEE")
            #print(pose_array)
            #stop
            pose_dict = {'cam': cn, 'frame': frame, 'pose': pose_array.tolist()}
            
            conf_array = []

            #for kp in list(pose[4].keys()):
            #    conf_array.append(pose[4][kp][2])
            
            #conf_array_average.append(np.mean(conf_array))
            match_poses.append(pose_dict)
            #print(pose)
            hungarian_average.append(hungarian_dist)

        dist_array.append(np.mean(hungarian_average))
        #dist_array.append(np.mean(conf_array_average))
        dict_array.append(match_poses)
        #all_points_array[]
        #all_points_array[]
        '''
        for r_ind in range(len(row_ind)):
        
            sync_ind = col_ind[r_ind]
            ref_ind = row_ind[r_ind]
            
            index_dict[i1].append(torch.tensor(list(poses_array_ref[ref_ind].values())[:-1]))
            index_dict[i2].append(torch.tensor(list(poses_array_sync[sync_ind].values())[:-1]))

            ankle_dict[i1].append(torch.from_numpy(points_ref[ref_ind]))
            ankle_dict[i2].append(torch.from_numpy(points_sync[sync_ind]))

            distance_hungarian.append(np.linalg.norm(points_ref[ref_ind] - points_sync[sync_ind]))
        '''
    #print(all_points_array)
    #print(dict_array)

    #print(hungarian_inds_array)
    #stop
    #print(point_array, " point array")
    
    #selected_indices = [500, 550, 600]#[np.argsort(dist_array)[:k]]
    selected_indices = [np.argsort(dist_array)[:k]]
    
    #print(np.array(dist_array)[selected_indices])
    
    sample_point_array = np.array(dict_array)[selected_indices]

    '''
    for pa in range(len(point_array)):

        dists = distance_hungarian_array[pa]
        selected_indices = [np.argsort(dists)[:k]]
        #print(np.array(dists)[selected_indices])
        print(selected_indices)
        key1 = list(point_array[pa].keys())[0]
        key2 = list(point_array[pa].keys())[1]

        pose1 = torch.stack(point_array[pa][key1], dim = 0)[selected_indices, :]
        pose2 = torch.stack(point_array[pa][key2], dim = 0)[selected_indices, :]

        pose1[:,:,:,2] = 1.0 
        pose2[:,:,:,2] = 1.0
        #index_dict = {key1: torch.stack(point_array[pa][key1], dim = 0)[selected_indices, :], key2: torch.stack(point_array[pa][key2], dim = 0)[selected_indices, :]}
        index_dict = {key1: pose1, key2: pose2}

        sample_point_array.append(index_dict)
    '''
    '''
    sample_point_array = []
    for pa in range(len(point_array)):

        dists = distance_hungarian_array[pa]
        selected_indices = [np.argsort(dists)[:k]]
        #print(np.array(dists)[selected_indices])
        print(selected_indices)
        key1 = list(point_array[pa].keys())[0]
        key2 = list(point_array[pa].keys())[1]

        pose1 = torch.stack(point_array[pa][key1], dim = 0)[selected_indices, :]
        pose2 = torch.stack(point_array[pa][key2], dim = 0)[selected_indices, :]

        pose1[:,:,:,2] = 1.0 
        pose2[:,:,:,2] = 1.0
        #index_dict = {key1: torch.stack(point_array[pa][key1], dim = 0)[selected_indices, :], key2: torch.stack(point_array[pa][key2], dim = 0)[selected_indices, :]}
        index_dict = {key1: pose1, key2: pose2}

        sample_point_array.append(index_dict)
    print(sample_point_array)
    '''
    return sample_point_array#point_array


def match_3d_farthest_point(sync_dict_array, peturb_extrinsics_array, peturb_single_view_array, all_points_array,center_array, run_name, k = 5, scale = 1000, title = 'plot', name = ['pred'], frame_init = 0 ,frame_end = 500):
    
    #draw a square

    cam_amount = len(all_points_array)

    point_array = []

    ankle_array = []

    distance_hungarian_array = []

    #indices_array = util.random_combination(list(range(len(all_points_array))), 2, np.inf)
    indices_array = []

    for i in range(1, len(all_points_array)):
        indices_array.append((0, i))
    indices = []
    for ind in indices_array:
        i1 = ind[0]
        i2 = ind[1]

        #######################################
        init_ref_shift_matrix = peturb_extrinsics_array[i1]['init_sync_center_array']
        icp_rot_matrix_ref = peturb_extrinsics_array[i1]['icp_rot_array']
        init_rot_matrix_ref = peturb_extrinsics_array[i1]['icp_init_rot_array']
        plane_matrix_array_ref = peturb_extrinsics_array[i1]['plane_matrix_array'] 

        cam_axis_ref = np.transpose(icp_rot_matrix_ref @ init_rot_matrix_ref @ plane_matrix_array_ref[:3,:3])

        init_sync_shift_matrix = peturb_extrinsics_array[i2]['init_sync_center_array']
        icp_rot_matrix_sync = peturb_extrinsics_array[i2]['icp_rot_array']
        init_rot_matrix_sync = peturb_extrinsics_array[i2]['icp_init_rot_array']
        plane_matrix_array_sync = peturb_extrinsics_array[i2]['plane_matrix_array']

        cam_axis_sync = np.transpose(icp_rot_matrix_sync @ init_rot_matrix_sync @ plane_matrix_array_sync[:3,:3])
        indices.append(ind)
    
    hungarian_inds_array = []
    
    view_frames = list(sync_dict_array[0].keys())
    points_array_set = []
    for fr1 in view_frames:
        init_ref_shift_matrix = peturb_extrinsics_array[0]['init_sync_center_array']
        icp_rot_matrix_ref = peturb_extrinsics_array[0]['icp_rot_array']
        init_rot_matrix_ref = peturb_extrinsics_array[0]['icp_init_rot_array']
        plane_matrix_array_ref = peturb_extrinsics_array[0]['plane_matrix_array'] 

        #######################################
        points_array_ref = []
        if fr1 not in all_points_array[i1]:
            continue
        for c in all_points_array[i1][fr1].keys():
            
            plane1 = util.plane_ray_intersection_np([np.array(list(all_points_array[i1][fr1][c]))[0]], [np.array(list(all_points_array[i1][fr1][c]))[1]], np.linalg.inv(peturb_single_view_array[i1]['cam_matrix']),  peturb_single_view_array[i1]["ground_normal"], peturb_single_view_array[i1]['ground_position'])
            plane1 = np.transpose(np.r_[plane1, np.ones((1, plane1.shape[1]))])
            transformed_ref = np.transpose(icp_rot_matrix_ref @ init_rot_matrix_ref @ np.transpose(np.transpose(plane_matrix_array_ref @ np.transpose(np.array(plane1))) - np.array([[init_ref_shift_matrix[0], init_ref_shift_matrix[1], 0, 0]]))[:3, :])
            
            points_array_ref.append(transformed_ref)

        point_mean = np.mean(points_array_ref, axis = 0)

        points_array_set.append(point_mean.tolist())
    #print(points_array_set)
    #print(torch.tensor(points_array_set).shape, " POINT ARRAY SET")
    #selected_points, selected_indices_sample = pytorch3d.ops.sample_farthest_points(torch.unsqueeze(torch.squeeze(torch.tensor(points_array_set)), dim = 0), K = 3*k)
    selected_points, selected_indices_sample = pytorch3d.ops.sample_farthest_points(torch.unsqueeze(torch.squeeze(torch.tensor(points_array_set)), dim = 0), K = k)
    #print(selected_indices_sample.shape)
    selected_indices_sample = torch.squeeze(selected_indices_sample).tolist()
    #stop
    for ind in indices:

        i1 = ind[0]
        i2 = ind[1]
        index_dict = {i1: [], i2: []}
        ankle_dict = {i1: [], i2: []}

        distance_hungarian = []

        view_frames1 = list(sync_dict_array[i2 - 1].keys())
        view_frames2 = list(sync_dict_array[i2 - 1].values())
        #print(view_frames1, " frame1")
        #print(view_frames2, " frame2")

        hungarian_inds = []    
        #for fr in range(len(list(all_points_array[i1].keys()))):  
        #for fr in range(len(view_frames1)):  
        for fr1 in selected_indices_sample:
            #fr1 = view_frames1[fr]

            if fr1 not in sync_dict_array[i2 - 1]:
                continue
            fr2 = sync_dict_array[i2 - 1][fr1]
            #fr2 = view_frames2[fr]
    
            if fr1 not in all_points_array[i1].keys():
                continue
            
            if fr2 not in all_points_array[i2].keys():
                continue
            #print("HELLO DID WE GET HERE????")
            #######################################
            init_ref_shift_matrix = peturb_extrinsics_array[i1]['init_sync_center_array']
            icp_rot_matrix_ref = peturb_extrinsics_array[i1]['icp_rot_array']
            init_rot_matrix_ref = peturb_extrinsics_array[i1]['icp_init_rot_array']
            plane_matrix_array_ref = peturb_extrinsics_array[i1]['plane_matrix_array'] 

            init_sync_shift_matrix = peturb_extrinsics_array[i2]['init_sync_center_array']
            icp_rot_matrix_sync = peturb_extrinsics_array[i2]['icp_rot_array']
            init_rot_matrix_sync = peturb_extrinsics_array[i2]['icp_init_rot_array']
            plane_matrix_array_sync = peturb_extrinsics_array[i2]['plane_matrix_array']

            #######################################
            points_array_ref = []
            poses_array_ref = []
            for c in all_points_array[i1][fr1].keys():
                
                plane1 = util.plane_ray_intersection_np([np.array(list(all_points_array[i1][fr1][c]))[0]], [np.array(list(all_points_array[i1][fr1][c]))[1]], np.linalg.inv(peturb_single_view_array[i1]['cam_matrix']),  peturb_single_view_array[i1]["ground_normal"], peturb_single_view_array[i1]['ground_position'])
                plane1 = np.transpose(np.r_[plane1, np.ones((1, plane1.shape[1]))])
                transformed_ref = np.transpose(icp_rot_matrix_ref @ init_rot_matrix_ref @ np.transpose(np.transpose(plane_matrix_array_ref @ np.transpose(np.array(plane1))) - np.array([[init_ref_shift_matrix[0], init_ref_shift_matrix[1], 0, 0]]))[:3, :])
                
                points_array_ref.append(transformed_ref)
                #CHECK THAT THE POSES ARE MATCHING!!!
                poses_array_ref.append(np.array(list(all_points_array[i1][fr1][c]))[4])

            points_array_sync = []
            poses_array_sync = []
            for c in all_points_array[i2][fr2].keys():
                
                plane1 = util.plane_ray_intersection_np([np.array(list(all_points_array[i2][fr2][c]))[0]], [np.array(list(all_points_array[i2][fr2][c]))[1]], np.linalg.inv(peturb_single_view_array[i2]['cam_matrix']),  peturb_single_view_array[i2]["ground_normal"], peturb_single_view_array[i2]['ground_position'])
                plane1 = np.transpose(np.r_[plane1, np.ones((1, plane1.shape[1]))])
                transformed_sync = np.transpose(icp_rot_matrix_sync @ init_rot_matrix_sync @ np.transpose(np.transpose(plane_matrix_array_sync @ np.transpose(np.array(plane1))) - np.array([[init_sync_shift_matrix[0], init_sync_shift_matrix[1], 0, 0]]))[:3, :])
                
                points_array_sync.append(transformed_sync)
                poses_array_sync.append(np.array(list(all_points_array[i2][fr2][c]))[4])

            points_ref = np.concatenate(points_array_ref)
            points_sync = np.concatenate(points_array_sync)

            row_ind, col_ind = util.hungarian_assignment_one2one(points_ref, points_sync)
            
            hungarian_dist = np.linalg.norm(points_ref[row_ind] - points_sync[col_ind], axis = 1)

            hungarian_inds.append([i1, i2, fr1, fr2, row_ind, col_ind, hungarian_dist])
            '''
            print(row_ind, " row ind")
            print(col_ind, " col ind")
            for r_ind in range(len(row_ind)):
            
                sync_ind = col_ind[r_ind]
                ref_ind = row_ind[r_ind]
                index_dict[i1].append(torch.tensor(list(poses_array_ref[ref_ind].values())[:-1]))
                index_dict[i2].append(torch.tensor(list(poses_array_sync[sync_ind].values())[:-1]))

                ankle_dict[i1].append(torch.from_numpy(points_ref[ref_ind]))
                ankle_dict[i2].append(torch.from_numpy(points_sync[sync_ind]))

                distance_hungarian.append(np.linalg.norm(points_ref[ref_ind] - points_sync[sync_ind]))
            '''

        hungarian_inds_array.append(hungarian_inds)
        '''
        distance_hungarian_array.append(distance_hungarian)
        point_array.append(index_dict)
        ankle_array.append(ankle_dict)
        '''
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #print(hungarian_inds_array[0][:5])
    #print("*************************")
    #print(hungarian_inds_array[1][:5])

    multiview_dict = {}

    for hd in range(len(hungarian_inds_array)):
        for hd1 in range(len(hungarian_inds_array[hd])):

            match = hungarian_inds_array[hd][hd1]

            if match[2] not in multiview_dict:
                multiview_dict[match[2]] = [{"name0": match[0], "name1": match[1], "frame": match[3], "set0": match[4], "set1": match[5], "dist": match[6]}]
            else:
                multiview_dict[match[2]].append({"name0": match[0], "name1": match[1], "frame": match[3], "set0": match[4], "set1": match[5], "dist": match[6]})
    
    multiview_dict_all = util.match_pairs_graph_per_frame(multiview_dict)
    #print(multiview_dict_all)
    #stop
    dict_array = []
    dist_array = []

    for mda in multiview_dict_all:
        
        #print(mda, " THE MDA")\
        #print(all_points_array)
        #print("***ASD")
        #print(mda.keys())

        match_poses = []

        hungarian_average = []
        
        conf_array_average = []
        if len(list(mda.keys())) < cam_amount:
            continue
        for cn in list(mda.keys()):
            ind_match = mda[cn][1]
            frame = mda[cn][0]

            hungarian_dist = mda[cn][2]
            #print(mda, " mda")
            #print(mda[cn], " mda[cn] ")
            #print(all_points_array[cn].keys())
            ind_key = list(all_points_array[cn][frame].keys())
            
            #print(cn, " THIS IS THE CN")
            #print(all_points_array[cn])
            #print("********************************")
            #print(all_points_array[cn][frame])
            #print("********************************")
            #print(ind_key, " ind kety")
            pose = all_points_array[cn][frame][ind_key[ind_match]]

            pose_array = np.array(list(pose[4].values())[:-1])
            '''
            print(pose)
            conf_array = []
            for ps in pose_array:
                print(ps, " THE PS")

                conf_array.append(ps[2])
            
            if np.mean(conf_array) < 0.75:
                continue 
            '''
            #print(pose)
            #stop
            pose_array[:, 2] = 1.0
            #print(pose_array.shape, " THE SHAEEEE")
            #print(pose_array)
            #stop
            pose_dict = {'cam': cn, 'frame': frame, 'pose': pose_array.tolist()}
            
            conf_array = []

            #for kp in list(pose[4].keys()):
            #    conf_array.append(pose[4][kp][2])
            
            #conf_array_average.append(np.mean(conf_array))
            match_poses.append(pose_dict)
            #print(pose)
            hungarian_average.append(hungarian_dist)

        dist_array.append(np.mean(hungarian_average))
        #dist_array.append(np.mean(conf_array_average))
        dict_array.append(match_poses)
        #all_points_array[]
        #all_points_array[]
        '''
        for r_ind in range(len(row_ind)):
        
            sync_ind = col_ind[r_ind]
            ref_ind = row_ind[r_ind]
            
            index_dict[i1].append(torch.tensor(list(poses_array_ref[ref_ind].values())[:-1]))
            index_dict[i2].append(torch.tensor(list(poses_array_sync[sync_ind].values())[:-1]))

            ankle_dict[i1].append(torch.from_numpy(points_ref[ref_ind]))
            ankle_dict[i2].append(torch.from_numpy(points_sync[sync_ind]))

            distance_hungarian.append(np.linalg.norm(points_ref[ref_ind] - points_sync[sync_ind]))
        '''
    #print(all_points_array)
    #print(dict_array)

    #print(hungarian_inds_array)
    #stop
    #print(point_array, " point array")
    
    #selected_indices = [500, 550, 600]#[np.argsort(dist_array)[:k]]

    #selected_indices = [np.argsort(dist_array)[:k]]
    selected_indices = [np.argsort(dist_array)]
    #selected_indices = [np.argsort(dist_array)]
    
    #print(np.array(dist_array)[selected_indices])
    
    sample_point_array = np.array(dict_array)[selected_indices]

    '''
    for pa in range(len(point_array)):

        dists = distance_hungarian_array[pa]
        selected_indices = [np.argsort(dists)[:k]]
        #print(np.array(dists)[selected_indices])
        print(selected_indices)
        key1 = list(point_array[pa].keys())[0]
        key2 = list(point_array[pa].keys())[1]

        pose1 = torch.stack(point_array[pa][key1], dim = 0)[selected_indices, :]
        pose2 = torch.stack(point_array[pa][key2], dim = 0)[selected_indices, :]

        pose1[:,:,:,2] = 1.0 
        pose2[:,:,:,2] = 1.0
        #index_dict = {key1: torch.stack(point_array[pa][key1], dim = 0)[selected_indices, :], key2: torch.stack(point_array[pa][key2], dim = 0)[selected_indices, :]}
        index_dict = {key1: pose1, key2: pose2}

        sample_point_array.append(index_dict)
    '''
    '''
    sample_point_array = []
    for pa in range(len(point_array)):

        dists = distance_hungarian_array[pa]
        selected_indices = [np.argsort(dists)[:k]]
        #print(np.array(dists)[selected_indices])
        print(selected_indices)
        key1 = list(point_array[pa].keys())[0]
        key2 = list(point_array[pa].keys())[1]

        pose1 = torch.stack(point_array[pa][key1], dim = 0)[selected_indices, :]
        pose2 = torch.stack(point_array[pa][key2], dim = 0)[selected_indices, :]

        pose1[:,:,:,2] = 1.0 
        pose2[:,:,:,2] = 1.0
        #index_dict = {key1: torch.stack(point_array[pa][key1], dim = 0)[selected_indices, :], key2: torch.stack(point_array[pa][key2], dim = 0)[selected_indices, :]}
        index_dict = {key1: pose1, key2: pose2}

        sample_point_array.append(index_dict)
    print(sample_point_array)
    '''
    return sample_point_array#point_array