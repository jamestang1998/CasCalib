import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random
import matplotlib.image as mpimg
import util
import os
import geometry
import ICP
import torch
import filter
import math

def icp(a, b, init_pose=(0,0,0), no_iterations = 13):
    '''
    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.
    '''

    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)

    #Initialise with the initial pose estimation
    Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])

    #print(src.shape, Tr[0:2].shape, " SRC TR")
    #src = cv2.transform(np.squeeze(src), Tr[0:2])
    src = (Tr @ np.concatenate((np.squeeze(src), np.ones((1, src.shape[2]))), axis = 0))[:2, :]

    #print(src.shape, " srcccccc")

    for i in range(no_iterations):
        #Find the nearest neighbours between the current source and the
        #destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(np.transpose(dst[0]))
        distances, indices = nbrs.kneighbors(np.transpose(src))

        #Compute the transformation between the current source
        #and destination cloudpoint
        #T = cv2.estimateAffine2D(src, dst[0, indices.T])
        T, inliers = cv2.estimateAffine2D(np.transpose(np.squeeze(src)), np.transpose(np.squeeze(dst))[indices,:])
        #Transform the previous source and update the
        #current source cloudpoint

        if T is None:
            continue

        if len(list(T)) == 0:
            continue
        '''
        print(src.shape, np.array(T).shape, " HELASOSDSADASSAD")
        print(src, " SRC")
        print(T, " T")
        '''
        #src = cv2.transform(np.expand_dims(src, axis = 1), np.expand_dims(T, axis = 1))
        '''
        print(np.array(np.concatenate((T, np.array([[0,0,1]])), axis = 0)).shape, " SHAPE")
        print(np.squeeze(src).shape, " srrrrrrrrc")
        print(np.ones((1, src.shape[1])), " srcccccccccccccccccccccccccccccccccccccccccccccc")
        print(np.concatenate((np.squeeze(src), np.ones((1, src.shape[1]))), axis = 0).shape, " SJAPEASDASDAS")
        '''
        #print(src.shape, dst.shape, " THIS IS SRC !!!!!")
        #print(np.array(T).shape)
        src = (np.array(np.concatenate((T, np.array([[0,0,1]])), axis = 0)) @ np.concatenate((np.squeeze(src), np.ones((1, src.shape[1]))), axis = 0))[:2, :] 
        #Save the transformation from the actual source cloudpoint
        #to the destination
        Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
    return Tr[0:2]

def icp_ransac(data_ref, data_sync, sync_time_dict, init_angle, init_ref_center, init_sync_center, reflect_true, n_iter = 10, ransac_iter = 10, num_points = 30, thresh = 2.0, save_dir = None, name = None):
    
    if os.path.isdir(save_dir + '/ICP/' + name) == False:
        os.mkdir(save_dir + '/ICP/' + name)

    frame_sync = list(sync_time_dict.values())
    frame_ref = list(sync_time_dict.keys())

    reflect_coef = 1

    if reflect_true:
        reflect_coef = -1

    R_init = np.array([[np.cos(init_angle), -np.sin(init_angle)],
                  [np.sin(init_angle),  reflect_coef*np.cos(init_angle)]])

    best_init_sync_shift = init_sync_center
    #best_
    #top_inlier = 0
    best_rot = np.array([[1,0], [0,1]])
    best_shift = np.array([0,0])

    ransac_rot = None
    ransac_shift = None

    ref_coords = None
    sync_coords = None

    top_inlier = 0
    sync_untransformed_array = None
    
    for itr in range(n_iter):
        #top_inlier = 0
        for i in range(ransac_iter):
            frames = list(random.sample(frame_ref, num_points))

            X_fix = []
            X_mov = []

            for fr in frames:
                
                fix = np.array(list(data_ref[fr].values()))[:, :2]
                mov = np.array(list(data_sync[sync_time_dict[fr]].values()))


                ref_normalize = np.transpose(np.stack([(fix[:, 0] - init_ref_center[0]), (fix[:, 1] - init_ref_center[1])]))
                sync_normalize = np.transpose(np.stack([(mov[:, 0] - init_sync_center[0]), (mov[:, 1] - init_sync_center[1])]))
                
                mov_transformed = np.transpose(best_rot @ (R_init @ np.transpose(np.array(sync_normalize)[:, :2]))) + best_shift

                #print(mov_transformed.shape, fix.shape, " mOVFIX")
                '''
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(mov_transformed)
                distances, indices = nbrs.kneighbors(ref_normalize)
                '''
                row_ind, col_ind = util.hungarian_assignment_one2one(ref_normalize, mov_transformed)

                #print(np.array(mov_transformed).shape, " MOV TrANSFORMEDDDD")

                X_mov.append(np.array(mov_transformed)[col_ind, :])
                X_fix.append(np.array(ref_normalize)[row_ind, :])

            X_fix = np.squeeze(np.concatenate(X_fix, axis = 0))[:,:2]
            X_mov = np.squeeze(np.concatenate(X_mov, axis = 0))[:,:2]

            #print(X_fix.shape, X_mov.shape, " HELAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")

            X_mov = np.transpose(best_rot @ (R_init @ (np.transpose(X_mov - init_sync_center)))) + best_shift
            
            #T, inliers = cv2.estimateAffinePartial2D(X_fix, X_mov)
            R, t = geometry.rigid_transform_3D(np.transpose(X_mov), np.transpose(X_fix))
        
            if R is None:
                continue

            if len(list(R)) == 0:
                continue

            #print(R, " T")
            #print(np.array(t).shape, " THE SHAEPEEEE")    
            ref_array = []
            sync_array = []
            inlier_sync = []

            sync_untransformed = []

            for j in range(len(frame_ref)):
                #print(j, " helaeokasd") 
                f_fix = list(data_ref[frame_ref[j]].values())
                f_mov = list(data_sync[sync_time_dict[frame_ref[j]]].values())
                
                sync_untransformed.append(np.transpose((R_init @ np.transpose(np.array(f_mov)[:, :2] - best_init_sync_shift))) + init_ref_center)
                frame_fix = np.array(f_fix)[:, :2]
                #print(frame_fix.shape, " SHAPEEEEEEEE")
                frame_mov = np.transpose((R @ best_rot @ R_init @ np.transpose(np.array(f_mov)[:, :2] - best_init_sync_shift))) + t + best_shift
                error = util.chamfer_distance(frame_fix, frame_mov, metric='l2', direction='bi')
                
                ref_array.append(frame_fix)
                sync_array.append(frame_mov)
                if error < thresh:
                    inlier_sync.append(frame_ref[j])
            
            sync_untransformed_array = np.concatenate(sync_untransformed)
            ref_array = np.concatenate(ref_array)
            sync_array = np.concatenate(sync_array)
            #print(ref_array.shape, " ref")
            #print(sync_array.shape, " sync")
            #print(itr, i , " ITERATION AND RANSAC ")
            if len(inlier_sync) > top_inlier:
                top_inlier = len(inlier_sync)
                #best_rot = T[:,0:2] 
                #best_shift = T[:,2]

                ransac_rot = R @ best_rot
                ransac_shift = t + best_shift

                ref_coords = ref_array 
                sync_coords = sync_array
        print(ransac_rot.shape, best_rot.shape, " best rot ")
        best_rot = ransac_rot @ best_rot
        best_shift = ransac_shift + best_shift

        if  save_dir is not None:
            fig, ax1 = plt.subplots(1, 1)
            ax1.set_yscale("linear")
            #print(np.array(ref_coords).shape, np.array(sync_coords).shape,  " ref COORD SHAPE AND SYC COORD SHAPE")
            ax1.scatter(np.array(ref_coords)[::10, 0], np.array(ref_coords)[::10, 1], c = 'b')
            ax1.scatter(np.array(sync_coords)[::10, 0], np.array(sync_coords)[::10, 1], c = 'r')
            ax1.set_title(str(top_inlier))

            if name is not None:
                fig.savefig(save_dir + '/ICP/' + name + '/best_' + str(itr) + '_' + str(top_inlier) + '_' + '.png')
            else:
                fig.savefig(save_dir + '/ICP/' + name + '/' + 'best_' + str(itr) + '_' + str(top_inlier) + '_' + str(itr) + '_' + '.png')
            
            if itr == 0:
                fig, ax2 = plt.subplots(1, 1)
                ax2.set_yscale("linear")
                #print(np.array(ref_coords).shape, np.array(sync_coords).shape,  " ref COORD SHAPE AND SYC COORD SHAPE")
                ax2.scatter(np.array(ref_coords)[::10, 0], np.array(ref_coords)[::10, 1], c = 'b')
                ax2.scatter(np.array(sync_untransformed_array)[::10, 0], np.array(sync_untransformed_array)[::10, 1], c = 'r')
                ax2.set_title(str(top_inlier))

                if name is not None:
                    fig.savefig(save_dir + '/ICP/' + name + '/init_' + str(itr) + '_' + str(top_inlier) + '_' + '.png')
                else:
                    fig.savefig(save_dir + '/ICP/' + name + '/' + 'init_' + str(itr) + '_' + str(top_inlier) + '_' + str(itr) + '_' + '.png')
            plt.close('all')
    
    return best_rot, best_shift, R_init

def icp_no_ransac(data_ref, data_sync, sync_time_dict, init_angle, init_ref_center, init_sync_center, reflect_true, n_iter = 10, ransac_iter = 10, num_points = 30, thresh = 2.0, save_dir = None, name = None):
    
    if os.path.isdir(save_dir + '/ICP/' + name) == False:
        os.mkdir(save_dir + '/ICP/' + name)

    frame_sync = list(sync_time_dict.values())
    frame_ref = list(sync_time_dict.keys())

    reflect_coef = 1

    if reflect_true:
        reflect_coef = -1
    
    R_init = np.array([[np.cos(init_angle), -np.sin(init_angle)],
                  [np.sin(init_angle),  reflect_coef*np.cos(init_angle)]])

    best_init_sync_shift = init_sync_center
    #best_
    #top_inlier = 0
    best_rot = np.array([[1,0], [0,1]])
    best_shift = init_ref_center

    ransac_rot = None
    ransac_shift = None

    ref_coords = None
    sync_coords = None

    top_inlier = 0
    sync_untransformed_array = []
    
    for itr in range(n_iter):

        X_fix = []
        X_mov = []

        for fr in frame_ref:
            
            fix = np.array(list(data_ref[fr].values()))[:, :2]
            mov = np.array(list(data_sync[sync_time_dict[fr]].values()))
            
            mov_transformed = np.transpose(best_rot @ (R_init @ np.transpose(np.array(mov)[:, :2] - init_sync_center))) + best_shift

            #print(mov_transformed.shape, fix.shape, " mOVFIX")
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(mov_transformed)
            distances, indices = nbrs.kneighbors(fix)

            X_mov.append(np.array(mov)[indices, :])
            X_fix.append(fix)

            if itr == 0:
                sync_untransformed_array.append(mov)

        X_fix = np.squeeze(np.concatenate(X_fix, axis = 0))[:,:2]
        X_mov = np.squeeze(np.concatenate(X_mov, axis = 0))[:,:2]

        if itr == 0:
            sync_untransformed_array = np.squeeze(np.concatenate(sync_untransformed_array, axis = 0))[:,:2]

        #print(X_fix.shape, X_mov.shape, " HELAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")

        X_mov = np.transpose(best_rot @ (R_init @ (np.transpose(X_mov - init_sync_center)))) + best_shift
        
        #T, inliers = cv2.estimateAffinePartial2D(X_fix, X_mov)
        R, t = geometry.rigid_transform_3D(np.transpose(X_mov), np.transpose(X_fix))
        print(best_rot, " best_rot")
        print(R, " r transform")
        print(np.arccos(R[0][0]), " PRED ANGLE")
        print(np.arccos(best_rot[0][0]), " init PRED ANGLE")
        #ref_coords = X_fix
        #sync_coords = X_mov
        if R is None:
            print(" R IS NONE")
            continue

        if len(list(R)) == 0:
            print(" R length is 0")
            continue

        best_rot = R @ best_rot
        best_shift = t + best_shift

        if  save_dir is not None:
            fig, ax1 = plt.subplots(1, 1)
            ax1.set_yscale("linear")
            #print(np.array(ref_coords).shape, np.array(sync_coords).shape,  " ref COORD SHAPE AND SYC COORD SHAPE")
            ax1.scatter(np.array(X_fix)[::10, 0], np.array(X_fix)[::10, 1], c = 'b')
            ax1.scatter(np.array(X_mov)[::10, 0], np.array(X_mov)[::10, 1], c = 'r')
            #ax1.scatter(np.array(X_fix)[100, 0], np.array(X_fix)[100, 1], c = 'b')
            #ax1.scatter(np.array(X_mov)[100, 0], np.array(X_mov)[100, 1], c = 'r')
            ax1.set_title(str(top_inlier))
            ax1.axis('square')

            if name is not None:
                fig.savefig(save_dir + '/ICP/' + name + '/best_' + str(itr) + '_' + str(top_inlier) + '_' + '.png')
            else:
                fig.savefig(save_dir + '/ICP/' + name + '/' + 'best_' + str(itr) + '_' + str(top_inlier) + '_' + str(itr) + '_' + '.png')
            
            if itr == 0:
                fig, ax2 = plt.subplots(1, 1)
                ax2.set_yscale("linear")
                #print(np.array(ref_coords).shape, np.array(sync_coords).shape,  " ref COORD SHAPE AND SYC COORD SHAPE")
                ax2.scatter(np.array(X_fix)[::10, 0], np.array(X_fix)[::10, 1], c = 'b')
                ax2.scatter(np.array(X_mov)[::10, 0], np.array(X_mov)[::10, 1], c = 'r')
                #ax2.scatter(np.array(X_fix)[100, 0], np.array(X_fix)[100, 1], c = 'b')
                #ax2.scatter(np.array(X_mov)[100, 0], np.array(X_mov)[100, 1], c = 'r')
                ax2.set_title(str(top_inlier))
                ax2.axis('square')

                if name is not None:
                    fig.savefig(save_dir + '/ICP/' + name + '/init_' + str(itr) + '_' + str(top_inlier) + '_' + '.png')
                else:
                    fig.savefig(save_dir + '/ICP/' + name + '/' + 'init_' + str(itr) + '_' + str(top_inlier) + '_' + str(itr) + '_' + '.png')
            plt.close('all')
    
    return best_rot, best_shift, R_init
#this function is just to test the matching function from icp normalize!!!!
def match_no_normalize(fix, mov):

    ref_normalize = np.transpose(np.stack([(fix[:, 0]), (fix[:, 1])]))
    sync_normalize = np.transpose(np.stack([(mov[:, 0]), (mov[:, 1])]))

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(sync_normalize)
    distances, indices = nbrs.kneighbors(ref_normalize)

    #X_mov.append(np.array(sync_normalize)[indices, :])
    #X_fix.append(ref_normalize)
    
    #X MOV indices
    return indices

def match(fix, mov, ref_coords, sync_coords):

    #################################################
    ref_center = np.mean(ref_coords, axis = 0)
    sync_center = np.mean(sync_coords, axis = 0)

    normalize_ref = np.array(ref_coords) - ref_center
    normalize_rot = np.array(sync_coords) - sync_center#ref_center

    normalize_ref_x_min = min(normalize_ref[:, 0])  
    normalize_ref_x_max = max(normalize_ref[:, 0])

    normalize_ref_y_min = min(normalize_ref[:, 1])
    normalize_ref_y_max = max(normalize_ref[:, 1])

    ###################
    
    normalize_rot_x_min = min(normalize_rot[:, 0])
    normalize_rot_x_max = max(normalize_rot[:, 0])

    normalize_rot_y_min = min(normalize_rot[:, 1])
    normalize_rot_y_max = max(normalize_rot[:, 1])

    ref_x_size = max([abs(normalize_ref_x_max), abs(normalize_ref_x_min)])
    ref_y_size = max([abs(normalize_ref_y_max), abs(normalize_ref_y_min)])

    rot_x_size = max([abs(normalize_rot_x_max), abs(normalize_rot_x_min)])
    rot_y_size = max([abs(normalize_rot_y_max), abs(normalize_rot_y_min)])

    X_fix = []
    X_mov = []

    ref_normalize = np.transpose(np.stack([(fix[:, 0] - ref_center[0])/ref_x_size, (fix[:, 1] - ref_center[1])/ref_y_size]))
    sync_normalize = np.transpose(np.stack([(mov[:, 0] - sync_center[0])/rot_x_size, (mov[:, 1] - sync_center[1])/rot_y_size]))

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(sync_normalize)
    distances, indices = nbrs.kneighbors(ref_normalize)

    #X_mov.append(np.array(sync_normalize)[indices, :])
    #X_fix.append(ref_normalize)
    
    #X MOV indices
    return indices

#Check that you dont match things that are outside the shifted overlap in the previous step
def icp_normalize(data_ref, data_sync, sync_time_dict, init_angle, init_ref_center, init_sync_center, reflect_true, n_iter = 10, save_dir = None, name = None, best_scale = 1.0):
    
    if os.path.isdir(save_dir + '/ICP/' + name) == False:
        os.mkdir(save_dir + '/ICP/' + name)
    
    ####################################
    ###############################################
    dilation = 15
    window = 1
    thresh_mag = 0.0
    data_interp_ref = interpolate(data_ref, 2*dilation)
    data_interp_sync = interpolate(data_sync, 2*dilation)

    ref_vel_array = []
    ref_vel_dict = {} 

    #print(data_interp_ref.keys())
    ref_dict = {}
    for rk in list(sync_time_dict.keys()):
        ref_dict[rk] = []

    sync_dict = {}
    for sk in list(sync_time_dict.values()):
        sync_dict[sk] = []

    for track in data_interp_ref.keys():
        
        ref_time = data_interp_ref[track]['frame']
        ref_points = data_interp_ref[track]['points']
        
        if len(ref_points) < 2*dilation + 1:
            continue
        #indices = np.where(np.equal(ref_time, data_interp_ref[track]['frame_actual']))[0]
        B_set = set(data_interp_ref[track]['frame_actual'])
        indices = [i for i, x in enumerate(ref_time) if x in B_set]
        #print(ref_points, " ref points")
        #print(np.array(ref_points).shape, " reffff")
        ref_vel = torch.squeeze(util.central_diff(torch.unsqueeze(torch.transpose(torch.tensor(ref_points), 0 , 1), dim = 1).double(), time = 1, dilation = dilation))
        
        ref_vel_array.append(ref_vel[:, indices])

        #if 879 in ref_time:
        #    print(track)
        #    print(" HELOOO")
        #    stop

        for ind in range(len(indices)):
            #print("*****************************")
            #print(ref_time, " ref time")
            #print(ref_dict.keys())
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            #if ref_time[indices[ind]] == 879:
            #    stop

            if ref_time[indices[ind]] not in list(sync_time_dict.keys()):
                continue 

            ref_dict[ref_time[indices[ind]]].append({track: ref_vel[:, ind]})

        #print(ref_vel.shape, " r ef vel")
        '''
        for st in range(len(ref_time)):
            
            if ref_time[st] in ref_vel_dict:
                ref_vel_dict[ref_time[st]].append(ref_vel[:, st].numpy())

            else:
                ref_vel_dict[ref_time[st]] = [ref_vel[:, st].numpy()]
        '''

    sync_vel_dict = {} 

    ###################################
    sync_vel_array = []
    for track in data_interp_sync.keys():
        
        sync_frame = data_interp_sync[track]['frame']
        sync_points = data_interp_sync[track]['points']
        sync_time = sync_frame

        if len(sync_points) < 2*dilation + 1:
            continue
        #if 879 in data_interp_sync[track]['frame_actual']:
        #    print("HELELASDASDASD")
        #    stop
        B_set = set(data_interp_sync[track]['frame_actual'])
        indices = [i for i, x in enumerate(sync_time) if x in B_set]
        
        sync_vel = torch.squeeze(util.central_diff(torch.unsqueeze(torch.transpose(torch.tensor(sync_points), 0 , 1), dim = 1).double(), time = best_scale, dilation = dilation))
        sync_vel_array.append(sync_vel[:, indices])

        for ind in range(len(indices)):
            if sync_time[indices[ind]] not in list(sync_dict.keys()):
                continue 

            sync_dict[sync_time[indices[ind]]].append({track: sync_vel[:, ind]})
    
    for k in list(ref_dict.keys()):
        myKeys = ref_dict[k]
        if len(myKeys) == 0:
            ref_dict.pop(k)
            continue
        myKeys_0 = list(myKeys[0].keys())
        myKeys_0.sort()
        ref_dict[k] = {i: ref_dict[k][0][i][:2].numpy() for i in myKeys_0}
    
    for k in list(sync_dict.keys()):
    
        myKeys = sync_dict[k]
        if len(myKeys) == 0:
            sync_dict.pop(k)
            continue
        myKeys_0 = list(myKeys[0].keys())
        myKeys_0.sort()
        sync_dict[k] = {i: sync_dict[k][0][i][:2].numpy() for i in myKeys_0}
    
    #############################################################
    ####################################

    frame_sync = list(sync_time_dict.values())
    frame_ref = list(sync_time_dict.keys())

    reflect_coef = 1

    if reflect_true:
        reflect_coef = -1
    
    R_init = np.array([[np.cos(init_angle), -np.sin(init_angle)],
                  [np.sin(init_angle),  reflect_coef*np.cos(init_angle)]])

    best_init_sync_shift = init_sync_center
    #best_
    #top_inlier = 0
    best_rot = np.array([[1,0], [0,1]])
    best_shift = [0,0]#init_ref_center

    ransac_rot = None
    ransac_shift = None

    ref_coords = None
    sync_coords = None

    top_inlier = 0
    sync_untransformed_array = []
    
    #################################################
    ref_coords = []
    for fr in list(sync_time_dict.keys()):

        for tr in list(data_ref[fr].keys()):
            
            ref_coords.append(data_ref[fr][tr][0:2])
    
    sync_coords = []
    
    for item in sync_time_dict.items():
        for tr in list(data_sync[item[1]].keys()):
            
            sync_coords.append(data_sync[item[1]][tr][0:2])
    #################################################
    ref_center = np.mean(ref_coords, axis = 0)
    sync_center = np.mean(sync_coords, axis = 0)

    normalize_ref = np.array(ref_coords) - ref_center
    normalize_rot = np.array(sync_coords) - sync_center#ref_center

    normalize_ref_x_min = min(normalize_ref[:, 0])  
    normalize_ref_x_max = max(normalize_ref[:, 0])

    normalize_ref_y_min = min(normalize_ref[:, 1])
    normalize_ref_y_max = max(normalize_ref[:, 1])

    ###################
    
    normalize_rot_x_min = min(normalize_rot[:, 0])
    normalize_rot_x_max = max(normalize_rot[:, 0])

    normalize_rot_y_min = min(normalize_rot[:, 1])
    normalize_rot_y_max = max(normalize_rot[:, 1])

    ref_x_size = 1.0#max([abs(normalize_ref_x_max), abs(normalize_ref_x_min)]) 
    ref_y_size = 1.0#max([abs(normalize_ref_y_max), abs(normalize_ref_y_min)])

    rot_x_size = 1.0#max([abs(normalize_rot_x_max), abs(normalize_rot_x_min)])
    rot_y_size = 1.0#max([abs(normalize_rot_y_max), abs(normalize_rot_y_min)])
    
    index_array = []
    for itr in range(n_iter):

        X_fix = []
        X_mov = []

        X_fix_vel = []
        X_mov_vel = []

        index_dict = {}
        #print(ref_dict.keys(), " ref!")
        #print(sync_dict.keys(), " sync!")
        for fr in frame_ref:
            
            #print(data_ref, " datar refffff")
            
            if fr not in list(ref_dict.keys()) or sync_time_dict[fr] not in list(sync_dict.keys()):
                print(fr, " FAILURE !!!")
                continue
            #print(fr, " hellooooo ")
            '''
            print("*****************************************")
            print(fr, " frame")
            print(data_ref[fr]) 
            print("##########################################")
            print(ref_dict[fr])
            print("......................................")
            '''

            fix = np.array(list(data_ref[fr].values()))[:, :2]
            mov = np.array(list(data_sync[sync_time_dict[fr]].values()))
            #print(ref_dict[fr], " ref dict")
            fix_vel = np.array(list(ref_dict[fr].values()))[:, :2]
            mov_vel = np.array(list(sync_dict[sync_time_dict[fr]].values()))

            X_fix_vel.append(fix_vel)
            #X_mov_vel.append(mov_vel)
            X_mov_vel.append(np.transpose(best_rot @ (R_init @ np.transpose(np.array(mov_vel)[:, :2]))) + best_shift)
            
            #mov_transformed = np.transpose(best_rot @ (R_init @ np.transpose(np.array(mov)[:, :2] - init_sync_center))) + best_shift

            ref_normalize = np.transpose(np.stack([(fix[:, 0] - ref_center[0])/ref_x_size, (fix[:, 1] - ref_center[1])/ref_y_size]))
            sync_normalize = np.transpose(np.stack([(mov[:, 0] - sync_center[0])/rot_x_size, (mov[:, 1] - sync_center[1])/rot_y_size]))
            
            mov_transformed = np.transpose(best_rot @ (R_init @ np.transpose(np.array(sync_normalize)[:, :2]))) + best_shift

            #print(mov_transformed.shape, fix.shape, " mOVFIX")
            '''
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(mov_transformed)
            distances, indices = nbrs.kneighbors(ref_normalize)
            '''
            row_ind, col_ind = util.hungarian_assignment_one2one(ref_normalize, mov_transformed)

            #print(np.array(mov_transformed).shape, " MOV TrANSFORMEDDDD")
            
            X_mov.append(np.array(mov_transformed)[col_ind, :])
            X_fix.append(np.array(ref_normalize)[row_ind, :])
            '''
            X_mov.append(np.array(mov_transformed)[indices, :])
            X_fix.append(ref_normalize)
            '''
            #this index dict isnt being used anywhere else ...
            #index_dict[sync_time_dict[fr]] = np.squeeze(indices)
            #print(np.squeeze(indices))
            if itr == 0:
                sync_untransformed_array.append(ref_normalize)
        
        print(np.concatenate(X_fix, axis = 0).shape, " SHAPE OF X FIX")
        index_array.append(index_dict)
        
        if np.concatenate(X_fix, axis = 0).shape[0] > 1:
            X_fix = np.squeeze(np.concatenate(X_fix, axis = 0))[:,:2]
        else:
            X_fix = np.concatenate(X_fix, axis = 0)[:,:2]
        
        if np.concatenate(X_mov, axis = 0).shape[0] > 1:
            X_mov = np.squeeze(np.concatenate(X_mov, axis = 0))[:,:2]
        else:
            X_mov = np.concatenate(X_mov, axis = 0)[:,:2]
        #X_fix_vel = np.squeeze(np.concatenate(X_fix_vel, axis = 0))[:,:2]
        #X_mov_vel = np.squeeze(np.concatenate(X_mov_vel, axis = 0))[:,:2]

        if itr == 0:
            if np.concatenate(sync_untransformed_array, axis = 0).shape[0] > 1:
                sync_untransformed_array = np.squeeze(np.concatenate(sync_untransformed_array, axis = 0))[:,:2]
            else:
                sync_untransformed_array = np.concatenate(sync_untransformed_array, axis = 0)[:,:2]
        #print(X_fix.shape, X_mov.shape, " HELAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")

        #X_mov = np.transpose(best_rot @ (R_init @ (np.transpose(X_mov - init_sync_center)))) + best_shift
        
        #T, inliers = cv2.estimateAffinePartial2D(X_fix, X_mov)
        R, t = geometry.rigid_transform_3D(np.transpose(X_mov), np.transpose(X_fix))
        #R, t = geometry.rigid_transform_3D(np.transpose(X_mov_vel), np.transpose(X_fix_vel))

        #print(best_rot, " best_rot")
        #print(R, " r transform")
        #print(np.arccos(R[0][0]), " PRED ANGLE")
        #print(np.arccos(best_rot[0][0]), " init PRED ANGLE")
        #ref_coords = X_fix
        #sync_coords = X_mov
        if math.isnan(R[0][0]):
            print(" ANGLE IS NAN")
            continue

        if R is None:
            print(" R IS NONE")
            continue

        if len(list(R)) == 0:
            print(" R length is 0")
            continue

        best_rot = R @ best_rot
        best_shift = t + best_shift

        if  save_dir is not None:
            fig, ax1 = plt.subplots(1, 1)
            ax1.set_yscale("linear")
            #print(np.array(ref_coords).shape, np.array(sync_coords).shape,  " ref COORD SHAPE AND SYC COORD SHAPE")
            ax1.scatter(np.array(X_fix)[::1, 0], np.array(X_fix)[::1, 1], c = 'b')
            ax1.scatter(np.array(X_mov)[::1, 0], np.array(X_mov)[::1, 1], c = 'r')
            #ax1.scatter(np.array(X_fix)[100, 0], np.array(X_fix)[100, 1], c = 'b')
            #ax1.scatter(np.array(X_mov)[100, 0], np.array(X_mov)[100, 1], c = 'r')
            ax1.set_title(str(top_inlier))
            ax1.axis('square')

            if name is not None:
                fig.savefig(save_dir + '/ICP/' + name + '/best_' + str(itr) + '_' + str(top_inlier) + '_' + '.png')
            else:
                fig.savefig(save_dir + '/ICP/' + name + '/' + 'best_' + str(itr) + '_' + str(top_inlier) + '_' + str(itr) + '_' + '.png')
            
            if itr == 0:
                fig, ax2 = plt.subplots(1, 1)
                ax2.set_yscale("linear")
                #print(np.array(ref_coords).shape, np.array(sync_coords).shape,  " ref COORD SHAPE AND SYC COORD SHAPE")
                ax2.scatter(np.array(X_fix)[::1, 0], np.array(X_fix)[::1, 1], c = 'b')
                ax2.scatter(np.array(X_mov)[::1, 0], np.array(X_mov)[::1, 1], c = 'r')
                #ax2.scatter(np.array(X_fix)[100, 0], np.array(X_fix)[100, 1], c = 'b')
                #ax2.scatter(np.array(X_mov)[100, 0], np.array(X_mov)[100, 1], c = 'r')
                ax2.set_title(str(top_inlier))
                ax2.axis('square')

                if name is not None:
                    fig.savefig(save_dir + '/ICP/' + name + '/init_' + str(itr) + '_' + str(top_inlier) + '_' + '.png')
                else:
                    fig.savefig(save_dir + '/ICP/' + name + '/' + 'init_' + str(itr) + '_' + str(top_inlier) + '_' + str(itr) + '_' + '.png')
            plt.close('all')
    
    return best_rot, [rot_x_size*best_shift[0], rot_y_size*best_shift[1]], R_init, index_array[-1]

#remove vel
def icp_no_vel(data_ref, data_sync, sync_time_dict, init_angle, init_ref_center, init_sync_center, reflect_true, n_iter = 10, save_dir = None, name = None, best_scale = 1.0):
    
    if os.path.isdir(save_dir + '/ICP/' + name) == False:
        os.mkdir(save_dir + '/ICP/' + name)
    
    ####################################
    ###############################################
    dilation = 15
    '''
    window = 1
    thresh_mag = 0.0
    data_interp_ref = interpolate(data_ref, 2*dilation)
    data_interp_sync = interpolate(data_sync, 2*dilation)

    ref_vel_array = []
    ref_vel_dict = {} 
    '''
    #print(data_interp_ref.keys())
    '''
    ref_dict = {}
    for rk in list(sync_time_dict.keys()):
        ref_dict[rk] = []

    sync_dict = {}
    for sk in list(sync_time_dict.values()):
        sync_dict[sk] = []
    
    for k in list(ref_dict.keys()):
        myKeys = ref_dict[k]
        if len(myKeys) == 0:
            ref_dict.pop(k)
            continue
        myKeys_0 = list(myKeys[0].keys())
        myKeys_0.sort()
        ref_dict[k] = {i: ref_dict[k][0][i][:2].numpy() for i in myKeys_0}
    
    for k in list(sync_dict.keys()):
    
        myKeys = sync_dict[k]
        if len(myKeys) == 0:
            sync_dict.pop(k)
            continue
        myKeys_0 = list(myKeys[0].keys())
        myKeys_0.sort()
        sync_dict[k] = {i: sync_dict[k][0][i][:2].numpy() for i in myKeys_0}
    '''
    #############################################################
    ####################################

    frame_sync = list(sync_time_dict.values())
    frame_ref = list(sync_time_dict.keys())

    reflect_coef = 1

    if reflect_true:
        reflect_coef = -1
    
    R_init = np.array([[np.cos(init_angle), -np.sin(init_angle)],
                  [np.sin(init_angle),  reflect_coef*np.cos(init_angle)]])

    best_init_sync_shift = init_sync_center
    #best_
    #top_inlier = 0
    best_rot = np.array([[1,0], [0,1]])
    best_shift = [0,0]#init_ref_center

    ransac_rot = None
    ransac_shift = None

    ref_coords = None
    sync_coords = None

    top_inlier = 0
    sync_untransformed_array = []
    
    #################################################
    ref_coords = []
    for fr in list(sync_time_dict.keys()):

        for tr in list(data_ref[fr].keys()):
            
            ref_coords.append(data_ref[fr][tr][0:2])
    
    sync_coords = []
    
    for item in sync_time_dict.items():
        for tr in list(data_sync[item[1]].keys()):
            
            sync_coords.append(data_sync[item[1]][tr][0:2])
    #################################################
    ref_center = np.mean(ref_coords, axis = 0)
    sync_center = np.mean(sync_coords, axis = 0)

    normalize_ref = np.array(ref_coords) - ref_center
    normalize_rot = np.array(sync_coords) - sync_center#ref_center

    normalize_ref_x_min = min(normalize_ref[:, 0])  
    normalize_ref_x_max = max(normalize_ref[:, 0])

    normalize_ref_y_min = min(normalize_ref[:, 1])
    normalize_ref_y_max = max(normalize_ref[:, 1])

    ###################
    
    normalize_rot_x_min = min(normalize_rot[:, 0])
    normalize_rot_x_max = max(normalize_rot[:, 0])

    normalize_rot_y_min = min(normalize_rot[:, 1])
    normalize_rot_y_max = max(normalize_rot[:, 1])

    ref_x_size = 1.0#max([abs(normalize_ref_x_max), abs(normalize_ref_x_min)]) 
    ref_y_size = 1.0#max([abs(normalize_ref_y_max), abs(normalize_ref_y_min)])

    rot_x_size = 1.0#max([abs(normalize_rot_x_max), abs(normalize_rot_x_min)])
    rot_y_size = 1.0#max([abs(normalize_rot_y_max), abs(normalize_rot_y_min)])
    
    index_array = []
    for itr in range(n_iter):

        X_fix = []
        X_mov = []

        X_fix_vel = []
        X_mov_vel = []

        index_dict = {}
        #print(ref_dict.keys(), " ref!")
        #print(sync_dict.keys(), " sync!")
        for fr in frame_ref:
            
            #print(data_ref, " datar refffff")
            
            if fr not in list(data_ref.keys()) or sync_time_dict[fr] not in list(data_sync.keys()):
                print(fr, " FAILURE !!!")
                continue
            #print(fr, " hellooooo ")
            '''
            print("*****************************************")
            print(fr, " frame")
            print(data_ref[fr]) 
            print("##########################################")
            print(ref_dict[fr])
            print("......................................")
            '''

            fix = np.array(list(data_ref[fr].values()))[:, :2]
            mov = np.array(list(data_sync[sync_time_dict[fr]].values()))
            #print(ref_dict[fr], " ref dict")
            #fix_vel = np.array(list(ref_dict[fr].values()))[:, :2]
            #mov_vel = np.array(list(sync_dict[sync_time_dict[fr]].values()))

            #X_fix_vel.append(fix_vel)
            #X_mov_vel.append(mov_vel)
            X_mov_vel.append(np.transpose(best_rot @ (R_init @ np.transpose(np.array(mov)[:, :2]))) + best_shift)
            
            #mov_transformed = np.transpose(best_rot @ (R_init @ np.transpose(np.array(mov)[:, :2] - init_sync_center))) + best_shift

            ref_normalize = np.transpose(np.stack([(fix[:, 0] - ref_center[0])/ref_x_size, (fix[:, 1] - ref_center[1])/ref_y_size]))
            sync_normalize = np.transpose(np.stack([(mov[:, 0] - sync_center[0])/rot_x_size, (mov[:, 1] - sync_center[1])/rot_y_size]))
            
            mov_transformed = np.transpose(best_rot @ (R_init @ np.transpose(np.array(sync_normalize)[:, :2]))) + best_shift

            #print(mov_transformed.shape, fix.shape, " mOVFIX")
            '''
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(mov_transformed)
            distances, indices = nbrs.kneighbors(ref_normalize)
            '''
            row_ind, col_ind = util.hungarian_assignment_one2one(ref_normalize, mov_transformed)

            #print(np.array(mov_transformed).shape, " MOV TrANSFORMEDDDD")
            
            X_mov.append(np.array(mov_transformed)[col_ind, :])
            X_fix.append(np.array(ref_normalize)[row_ind, :])
            '''
            X_mov.append(np.array(mov_transformed)[indices, :])
            X_fix.append(ref_normalize)
            '''
            #this index dict isnt being used anywhere else ...
            #index_dict[sync_time_dict[fr]] = np.squeeze(indices)
            #print(np.squeeze(indices))
            if itr == 0:
                sync_untransformed_array.append(ref_normalize)
        
        print(np.concatenate(X_fix, axis = 0).shape, " SHAPE OF X FIX")
        index_array.append(index_dict)
        
        if np.concatenate(X_fix, axis = 0).shape[0] > 1:
            X_fix = np.squeeze(np.concatenate(X_fix, axis = 0))[:,:2]
        else:
            X_fix = np.concatenate(X_fix, axis = 0)[:,:2]
        
        if np.concatenate(X_mov, axis = 0).shape[0] > 1:
            X_mov = np.squeeze(np.concatenate(X_mov, axis = 0))[:,:2]
        else:
            X_mov = np.concatenate(X_mov, axis = 0)[:,:2]
        #X_fix_vel = np.squeeze(np.concatenate(X_fix_vel, axis = 0))[:,:2]
        #X_mov_vel = np.squeeze(np.concatenate(X_mov_vel, axis = 0))[:,:2]

        if itr == 0:
            if np.concatenate(sync_untransformed_array, axis = 0).shape[0] > 1:
                sync_untransformed_array = np.squeeze(np.concatenate(sync_untransformed_array, axis = 0))[:,:2]
            else:
                sync_untransformed_array = np.concatenate(sync_untransformed_array, axis = 0)[:,:2]
        #print(X_fix.shape, X_mov.shape, " HELAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")

        #X_mov = np.transpose(best_rot @ (R_init @ (np.transpose(X_mov - init_sync_center)))) + best_shift
        
        #T, inliers = cv2.estimateAffinePartial2D(X_fix, X_mov)
        R, t = geometry.rigid_transform_3D(np.transpose(X_mov), np.transpose(X_fix))
        #R, t = geometry.rigid_transform_3D(np.transpose(X_mov_vel), np.transpose(X_fix_vel))

        #print(best_rot, " best_rot")
        #print(R, " r transform")
        #print(np.arccos(R[0][0]), " PRED ANGLE")
        #print(np.arccos(best_rot[0][0]), " init PRED ANGLE")
        #ref_coords = X_fix
        #sync_coords = X_mov
        if math.isnan(R[0][0]):
            print(" ANGLE IS NAN")
            continue

        if R is None:
            print(" R IS NONE")
            continue

        if len(list(R)) == 0:
            print(" R length is 0")
            continue

        best_rot = R @ best_rot
        best_shift = t + best_shift

        if  save_dir is not None:
            fig, ax1 = plt.subplots(1, 1)
            ax1.set_yscale("linear")
            #print(np.array(ref_coords).shape, np.array(sync_coords).shape,  " ref COORD SHAPE AND SYC COORD SHAPE")
            ax1.scatter(np.array(X_fix)[::1, 0], np.array(X_fix)[::1, 1], c = 'b')
            ax1.scatter(np.array(X_mov)[::1, 0], np.array(X_mov)[::1, 1], c = 'r')
            #ax1.scatter(np.array(X_fix)[100, 0], np.array(X_fix)[100, 1], c = 'b')
            #ax1.scatter(np.array(X_mov)[100, 0], np.array(X_mov)[100, 1], c = 'r')
            ax1.set_title(str(top_inlier))
            ax1.axis('square')

            if name is not None:
                fig.savefig(save_dir + '/ICP/' + name + '/best_' + str(itr) + '_' + str(top_inlier) + '_' + '.png')
            else:
                fig.savefig(save_dir + '/ICP/' + name + '/' + 'best_' + str(itr) + '_' + str(top_inlier) + '_' + str(itr) + '_' + '.png')
            
            if itr == 0:
                fig, ax2 = plt.subplots(1, 1)
                ax2.set_yscale("linear")
                #print(np.array(ref_coords).shape, np.array(sync_coords).shape,  " ref COORD SHAPE AND SYC COORD SHAPE")
                ax2.scatter(np.array(X_fix)[::1, 0], np.array(X_fix)[::1, 1], c = 'b')
                ax2.scatter(np.array(X_mov)[::1, 0], np.array(X_mov)[::1, 1], c = 'r')
                #ax2.scatter(np.array(X_fix)[100, 0], np.array(X_fix)[100, 1], c = 'b')
                #ax2.scatter(np.array(X_mov)[100, 0], np.array(X_mov)[100, 1], c = 'r')
                ax2.set_title(str(top_inlier))
                ax2.axis('square')

                if name is not None:
                    fig.savefig(save_dir + '/ICP/' + name + '/init_' + str(itr) + '_' + str(top_inlier) + '_' + '.png')
                else:
                    fig.savefig(save_dir + '/ICP/' + name + '/' + 'init_' + str(itr) + '_' + str(top_inlier) + '_' + str(itr) + '_' + '.png')
            plt.close('all')
    
    return best_rot, [rot_x_size*best_shift[0], rot_y_size*best_shift[1]], R_init, index_array[-1]

def interpolate(data, min_size = 2):
    
    track_dict = {}
    for fr in list(data.keys()):
        for tr in list(data[fr].keys()):

            if tr in track_dict: 
                track_dict[tr].append({'frame': fr, 'coord': data[fr][tr][0:2]})
            else:
                track_dict[tr] = [{'frame': fr, 'coord': data[fr][tr][0:2]}]

    points_dict = {}
    for t in track_dict.keys():
        
        points_array = []
        frame_array = []
        frame_actual_array = []
        points_actual_array = []
        if len(track_dict[t]) < min_size:
            #print("is it here?")
            continue
            
        for i in range(len(track_dict[t])):

            if i == len(track_dict[t]) - 1:
                points_array.append(track_dict[t][i]['coord'])
                frame_array.append(track_dict[t][i]['frame'])

                frame_actual_array.append(track_dict[t][i]['frame'])
                points_actual_array.append(track_dict[t][i]['coord'])

            elif track_dict[t][i + 1]['frame'] - track_dict[t][i]['frame'] == 1:

                points_array.append(track_dict[t][i]['coord'])
                #points_array.append(track_dict[t][i + 1]['coord'])

                frame_array.append(track_dict[t][i]['frame'])
                #frame_array.append(track_dict[t][i + 1]['frame'])
                frame_actual_array.append(track_dict[t][i]['frame'])
                points_actual_array.append(track_dict[t][i]['coord'])

                #print(track_dict[t][i + 1]['frame'], track_dict[t][i]['frame'], " frame!!!!")
            else:
                #print(track_dict[t][i + 1]['frame'], track_dict[t][i]['frame'], " frame!!!!")
                grid = np.arange(track_dict[t][i]['frame'], track_dict[t][i + 1]['frame'])
                #print(grid, " frame grid")

                points = np.transpose(np.array([track_dict[t][i]['coord'], track_dict[t][i + 1]['coord']]))
                time = [track_dict[t][i]['frame'], track_dict[t][i + 1]['frame']]
                y_new = util.interpolate(points, time, grid)
                #print(y_new.shape, " hiasddddddddddddddddddddd")
                for intp in range(y_new.shape[1]):
                    points_array.append(y_new[:, intp])
                    frame_array.append(grid[intp])
                
                frame_actual_array.append(track_dict[t][i]['frame'])
                points_actual_array.append(track_dict[t][i]['coord'])
        
        points_dict[t] = {'points': points_array, "frame": frame_array, 'points_actual': points_actual_array, 'frame_actual': frame_actual_array}
        #print(t, points_dict[t]," HIIIIIIIIIIIIIIIIIIIASDDDDDDDDDDDDDDDDDDDDDDDDDDD")
    return points_dict

def get_track(data, min_size = 2):
    
    print(data)

    mean_array = []
    track_dict = {}
    for fr in list(data.keys()):
        print(data[fr], " THE DATA")
        for tr in list(data[fr].keys()):
            
            mean_array.append(data[fr][tr][0:2])
            if tr in track_dict: 
                track_dict[tr].append({'frame': fr, 'coord': data[fr][tr][0:2]})
            else:
                track_dict[tr] = [{'frame': fr, 'coord': data[fr][tr][0:2]}]

    mean_center = np.mean(mean_array, axis = 0)
    points_dict = {}
    for t in track_dict.keys():
        
        points_array = []
        frame_array = []
        frame_actual_array = []
        points_actual_array = []
            
        for i in range(len(track_dict[t])):

            points_array.append(track_dict[t][i]['coord'] - mean_center)
            frame_array.append(track_dict[t][i]['frame'])

            points_actual_array.append(track_dict[t][i]['coord'] - mean_center)
            frame_actual_array.append(track_dict[t][i]['frame'])
        
        points_dict[t] = {'points': points_array, "frame": frame_array, 'points_actual': points_actual_array, 'frame_actual': frame_actual_array}
        #print(t, points_dict[t]," HIIIIIIIIIIIIIIIIIIIASDDDDDDDDDDDDDDDDDDDDDDDDDDD")
    return points_dict