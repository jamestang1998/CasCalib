import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotting
import util
from scipy.spatial.distance import directed_hausdorff
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
import filter
import matplotlib.pyplot as plt
import math
import knn
import random

from scipy.spatial.distance import cdist
from simpleicp import PointCloud, SimpleICP
import Icp2d

def time_all(data_ref, data_sync, save_dir = None, sync = True, name = ''):

    best_shift_array = []
    best_scale_array = []
    sync_dict_array = []
    
    for i in range(len(data_sync)):
        scale_array_init = [1]#[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 2.25,2.5,2.75,3.0, 3.25, 3.5]
        shift_array_init = list(range(-int(len(list(data_sync[i].keys()))/2.0), int(len(list(data_sync[i].keys()))/2.0)))
        #print(data_ref)
        best_shift, best_scale, sync_dict = time_align(data_ref, data_sync[i], save_dir = save_dir, name = str(i) + '_' + name, sync = sync, scale_search = scale_array_init, shift_search = shift_array_init, trial_name = 'init')
        
        #scale_array_init = [best_scale]
        #shift_array_init = list(range(int(best_shift) - 20, int(best_shift) + 20))
        #best_shift, best_scale, sync_dict = time_align_corr(data_ref, data_sync[i], save_dir = save_dir, name = str(i) + '_' + name, window = 1, dilation = 15, sync = sync, scale_search = scale_array_init, shift_search = shift_array_init, trial_name = 'refine')

        best_shift_array.append(best_shift)
        best_scale_array.append(best_scale)
        sync_dict_array.append(sync_dict)
    return best_shift_array, best_scale_array, sync_dict_array

def time_align(data_ref, data_sync, save_dir = None, name = None, end = True, window = 10, dilation = 15, thresh_mag = 0.0, sync = True, trial_name = 'init', scale_search = [1], shift_search = [0]):

    if os.path.isdir(save_dir + '/search_time/' + name) == False:
        os.mkdir(save_dir + '/search_time/' + name)

    #print(data_ref)
    #print(data_sync)

    data_interp_ref = Icp2d.interpolate(data_ref, 2*dilation)
    data_interp_sync = Icp2d.interpolate(data_sync, 2*dilation)
    #print(data_interp_ref, " HIIII data 1")
    #print(data_interp_ref, " HIIII data 2")
    ref_vel_array = []
    ref_vel_dict = {} 
    for track in data_interp_ref.keys():
        
        ref_time = data_interp_ref[track]['frame']
        ref_points = data_interp_ref[track]['points']
        #print(len(ref_points), 2*dilation + 1, " dilation")
        if len(ref_points) < 2*dilation + 1:
            continue
        #indices = np.where(np.equal(ref_time, data_interp_ref[track]['frame_actual']))[0]
        B_set = set(data_interp_ref[track]['frame_actual'])
        indices = [i for i, x in enumerate(ref_time) if x in B_set]
        #print(ref_points, " ref points")
        #print(np.array(ref_points).shape, " reffff")
        ref_vel = torch.squeeze(util.central_diff(torch.unsqueeze(torch.transpose(torch.from_numpy(np.array(ref_points)), 0 , 1), dim = 1).double(), time = 1, dilation = dilation))
        
        ref_vel_array.append(ref_vel[:, indices])
        #print(ref_vel.shape, " r ef vel")
        for st in range(len(ref_time)):
            
            if ref_time[st] in ref_vel_dict:
                ref_vel_dict[ref_time[st]].append(ref_vel[:, st].numpy())

            else:
                ref_vel_dict[ref_time[st]] = [ref_vel[:, st].numpy()]

    
    ref_vel_dict_avg = {}
    #print(ref_vel_dict)
    for f in ref_vel_dict.keys():
        ref_vel_dict_avg[f] = np.linalg.norm(np.mean(np.array(ref_vel_dict[f]), axis = 0))
    
    ref_time = np.array(list(ref_vel_dict_avg.keys()))
    ref_vel = np.array(list(ref_vel_dict_avg.values()))
    #print(ref_time)
    #print(ref_vel)
    ref_average = filter.window_average(ref_vel, window = window)

    ref_index = np.where(ref_average > thresh_mag)[0]

    #############################################################
    sync_vel_dict = {} 

    ###################################
    sync_vel_array = []
    for track in data_interp_sync.keys():
        
        sync_frame = data_interp_sync[track]['frame']
        sync_points = data_interp_sync[track]['points']
        sync_time = sync_frame

        if len(sync_points) < 2*dilation + 1:
            continue

        B_set = set(data_interp_sync[track]['frame_actual'])
        indices = [i for i, x in enumerate(sync_time) if x in B_set]
        
        sync_vel = torch.squeeze(util.central_diff(torch.unsqueeze(torch.transpose(torch.from_numpy(np.array(sync_points)), 0 , 1), dim = 1).double(), time = 1, dilation = dilation))
        sync_vel_array.append(sync_vel[:, indices])

        for st in range(len(sync_time)):
        
            if sync_time[st] in sync_vel_dict:
                sync_vel_dict[sync_time[st]].append(sync_vel[:, st].numpy())
                
            else:
                sync_vel_dict[sync_time[st]] = [sync_vel[:, st].numpy()]

    #print(sync_vel_dict)
    #stop
        ####################################
    '''   
    if  save_dir is not None:
        fig2, ax2 = plt.subplots(1, 1)
        fig3, ax3 = plt.subplots(1, 1)

        print(ref_vel_array[0].shape, " REF VELLLL")
        print(ref_vel_array[10].shape, " R1111EF VELLLL")
        print(len(ref_time), " REF TIME")

        ref_vel_array = np.transpose(np.concatenate(ref_vel_array, axis = 1))
        sync_vel_array = np.transpose(np.concatenate(sync_vel_array, axis = 1))
        print(np.array(ref_vel_array).shape, " ASDASDASASDADADASDAS")
        for rf in range(0, len(ref_vel_array)):
            #ax2.arrow(0,0, np.array(ref_vel_array)[rf, 0], np.array(ref_vel_array)[rf, 1])
            a,b = util.cart2pol(np.array(ref_vel_array)[rf, 0], np.array(ref_vel_array)[rf, 1])
            ax2.scatter(b, a)

        for sf in range(0, len(sync_vel_array)):
            #ax3.arrow(0,0, np.array(sync_vel_array)[sf, 0], np.array(sync_vel_array)[sf, 1])
            a,b = util.cart2pol(np.array(sync_vel_array)[sf, 0], np.array(sync_vel_array)[sf, 1])
            ax3.scatter(b, a)

        fig2.savefig(save_dir + '/search_time/' + name + '/' + 'arrow_ref' + '.png')  
        fig3.savefig(save_dir + '/search_time/' + name + '/' + 'arrow_sync' + '.png')  
    '''
    sync_vel_dict_avg = {}
    for f in sync_vel_dict.keys():
        sync_vel_dict_avg[f] =  np.linalg.norm(np.mean(np.array(sync_vel_dict[f]), axis = 0))
    
    sync_time_init = np.array(list(sync_vel_dict_avg.keys()))
    sync_vel_init = np.array(list(sync_vel_dict_avg.values()))
            
    #sync_average = filter.window_average(sync_vel_init, window = window)
    #############################################################
    best_sync_time = None
    best_sync_average = None

    best_scale = 1.0
    best_shift = 0.0
    '''
    for shift in list(range(-int(sync_time_init.shape[0]/64), int(sync_time_init.shape[0]/64))):
        for scale in list(np.linspace(0.5, 2, 5)) + [1.0]:
    '''
    '''
    for shift in list(range(-30, 30)):
        for scale in [1.0]:
    '''
    '''
    #for shift in [0]:#list(range(-int(sync_time_init.shape[0]/64), int(sync_time_init.shape[0]/64))):
        #for scale in [1]:#list(np.linspace(0.5, 2, 5)) + [1.0]:
    '''
    error_array = []
    shift_array = []
    #scale_search = [1], shift_search
    #shift_val = list(range(-int(sync_time_init.shape[0]/30), int(sync_time_init.shape[0]/30)))
    #scale_val = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 2.25,2.5,2.75,3.0, 3.25, 3.5]
    shift_val = shift_search
    scale_val = scale_search
    #scale_val = list(np.linspace(0.25, 4, 10)) + [1.0]
    if sync == False:
        shift_val = [0]
        scale_val = [1]

    best_correlation = np.inf

    for scale in scale_val:#list(np.linspace(0.5, 2, 5)) + [1.0]:

        shift_array_loop = []
        error_array_loop = []

        correlation_array_loop = []
        for shift in shift_val:#list(range(-int(sync_time_init.shape[0]/64), int(sync_time_init.shape[0]/64))):
            sync_time = scale*np.array(list(sync_vel_dict_avg.keys())) + shift

            sync_vel = np.array(list(sync_vel_dict_avg.values()))

            sync_average = filter.window_average(sync_vel/scale, window = window)
            
            sync_index = np.where(sync_average > thresh_mag)[0]

            argmins, argmins1, d2_sorted = knn.knn(np.expand_dims(ref_time[ref_index], axis = 1), np.expand_dims(sync_time[sync_index], axis = 1))

            correlation = np.corrcoef(sync_average[sync_index][argmins1], ref_average[ref_index][argmins], rowvar = True)[0,1]
            mean_error = np.mean(np.abs(sync_average[sync_index][argmins1] - ref_average[ref_index][argmins]))
            
            #correlation = correlation_signed
            correlation_array_loop.append(correlation)
            error_array.append(mean_error)
            shift_array.append(shift)
            if best_correlation > mean_error:
                best_scale = scale
                best_shift = shift
                best_correlation = mean_error

                best_sync_time = sync_time
                best_sync_average = sync_average

            shift_array_loop.append(shift)
            error_array_loop.append(mean_error)
            ####################
        if  save_dir is not None:
            fig1, ax1 = plt.subplots(1, 1)
            ax1.plot(shift_array_loop, error_array_loop, c = 'r')
            ax1.set_yscale("linear")

            fig2, ax2 = plt.subplots(1, 1)
            ax2.plot(shift_array_loop, correlation_array_loop, c = 'r')
            ax2.set_yscale("linear")
            if name is not None:
                #fig1.savefig(save_dir + '/search_time/' + name + '/' + 'error_' + str(scale) + '_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
                fig1.savefig(save_dir + '/search_time/' + name + '/' + 'error_' + name + '_' + str(best_shift) + '_' + '.png')
            else:
                #fig1.savefig(save_dir + '/search_time/' + name + '/' + 'error_' + str(scale) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')  
                fig1.savefig(save_dir + '/search_time/' + name + '/' + 'error_' + name + '_' + str(best_shift) + '_' + '.png') 

            if name is not None:
                #fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + str(scale) + '_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
                fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + name + '_' + str(best_shift) + '_' + '.png')
            else:
                #fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + str(scale) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
                fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + name + '_' + str(best_shift) + '_' + '.png')    
            plt.close('all')
            
            #if  save_dir is not None and end == False:
            '''
            if  save_dir is not None:
                fig, ax1 = plt.subplots(1, 1)
                ax1.set_yscale("linear")
                ax1.plot(sync_time, sync_average, c = 'r')
                ax1.plot(ref_time, ref_average, c = 'b')
                ax1.set_title(str(correlation_signed))

                if name is not None:
                    fig.savefig(save_dir + '/search_time/' + name + '/' + str(name) + '_' + str(shift) + '_' + str(scale) +'_' + '.png')
                else:
                    fig.savefig(save_dir + '/search_time/' + name + '/' + 'average_' + str(shift) + '_' + str(scale) +'_' + '.png')
                
                plt.close('all')
            '''
    if  save_dir is not None:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        ax1.plot(best_sync_time, best_sync_average, c = 'r')
        ax1.plot(ref_time, ref_average, c = 'b')
        ax1.set_title(str(best_correlation))

        if name is not None:
            fig.savefig(save_dir + '/search_time/' + name + '/' + 'best_' + trial_name + '_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
        else:
            fig.savefig(save_dir + '/search_time/' + name + '/' + 'best_average_' + trial_name + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')  

        fig1, ax2 = plt.subplots(1, 1)
        ax2.set_yscale("linear")
        ax2.plot(shift_array, error_array, c = 'r')
        ax2.set_title(str(best_correlation))
        '''
        if name is not None:
            fig1.savefig(save_dir + '/search_time/' + name + '/' + 'error_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
        else:
            fig1.savefig(save_dir + '/search_time/' + name + '/' + 'error_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')  
        '''
        plt.close('all')

    shifted_time = best_scale*np.array(list(data_sync.keys())) + best_shift
    argmins, argmins1, d2_sorted = knn.knn(np.expand_dims(list(data_ref.keys()), axis = 1), np.expand_dims(shifted_time, axis = 1))

    sync_dict = {}
    for i in range(len(argmins)):
        sync_dict[list(data_ref.keys())[argmins[i]]] = int(list(data_sync.keys())[argmins1[i]])

    return best_shift, best_scale, sync_dict

def time_align_corr(data_ref, data_sync, save_dir = None, name = None, end = True, window = 10, dilation = 15, thresh_mag = 0.0, sync = True, trial_name = 'init', scale_search = [1], shift_search = [0]):
    
    if os.path.isdir(save_dir + '/search_time/' + name) == False:
        os.mkdir(save_dir + '/search_time/' + name)

    data_interp_ref = Icp2d.interpolate(data_ref, 2*dilation)
    data_interp_sync = Icp2d.interpolate(data_sync, 2*dilation)

    ref_vel_array = []
    ref_vel_dict = {} 
    for track in data_interp_ref.keys():
        
        ref_time = data_interp_ref[track]['frame']
        ref_points = data_interp_ref[track]['points']
        
        #indices = np.where(np.equal(ref_time, data_interp_ref[track]['frame_actual']))[0]
        B_set = set(data_interp_ref[track]['frame_actual'])
        indices = [i for i, x in enumerate(ref_time) if x in B_set]
        #print(ref_points, " ref points")
        #print(np.array(ref_points).shape, " reffff")
        ref_vel = torch.squeeze(util.central_diff(torch.unsqueeze(torch.transpose(torch.from_numpy(np.array(ref_points)), 0 , 1), dim = 1).double(), time = 1, dilation = dilation))
        
        ref_vel_array.append(ref_vel[:, indices])
        #print(ref_vel.shape, " r ef vel")
        for st in range(len(ref_time)):
            
            if ref_time[st] in ref_vel_dict:
                ref_vel_dict[ref_time[st]].append(ref_vel[:, st].numpy())

            else:
                ref_vel_dict[ref_time[st]] = [ref_vel[:, st].numpy()]

    
    ref_vel_dict_avg = {}
    #print(ref_vel_dict)
    for f in ref_vel_dict.keys():
        ref_vel_dict_avg[f] = np.linalg.norm(np.mean(np.array(ref_vel_dict[f]), axis = 0))
    
    ref_time = np.array(list(ref_vel_dict_avg.keys()))
    ref_vel = np.array(list(ref_vel_dict_avg.values()))
            
    ref_average = filter.window_average(ref_vel, window = window)

    ref_index = np.where(ref_average > thresh_mag)[0]

    #############################################################
    sync_vel_dict = {} 

    ###################################
    sync_vel_array = []
    for track in data_interp_sync.keys():
        
        sync_frame = data_interp_sync[track]['frame']
        sync_points = data_interp_sync[track]['points']
        sync_time = sync_frame

        B_set = set(data_interp_sync[track]['frame_actual'])
        indices = [i for i, x in enumerate(sync_time) if x in B_set]
        
        sync_vel = torch.squeeze(util.central_diff(torch.unsqueeze(torch.transpose(torch.from_numpy(np.array(sync_points)), 0 , 1), dim = 1).double(), time = 1, dilation = dilation))
        sync_vel_array.append(sync_vel[:, indices])

        for st in range(len(sync_time)):
        
            if sync_time[st] in sync_vel_dict:
                sync_vel_dict[sync_time[st]].append(sync_vel[:, st].numpy())
                
            else:
                sync_vel_dict[sync_time[st]] = [sync_vel[:, st].numpy()]

    sync_vel_dict_avg = {}
    for f in sync_vel_dict.keys():
        sync_vel_dict_avg[f] =  np.linalg.norm(np.mean(np.array(sync_vel_dict[f]), axis = 0))
    
    sync_time_init = np.array(list(sync_vel_dict_avg.keys()))
    sync_vel_init = np.array(list(sync_vel_dict_avg.values()))
            
    best_sync_time = None
    best_sync_average = None

    best_scale = 1.0
    best_shift = 0.0

    error_array = []
    shift_array = []

    shift_val = shift_search
    scale_val = scale_search

    if sync == False:
        shift_val = [0]
        scale_val = [1]

    best_correlation = -np.inf

    for scale in scale_val:#list(np.linspace(0.5, 2, 5)) + [1.0]:

        shift_array_loop = []
        error_array_loop = []

        correlation_array_loop = []
        for shift in shift_val:#list(range(-int(sync_time_init.shape[0]/64), int(sync_time_init.shape[0]/64))):
            sync_time = scale*np.array(list(sync_vel_dict_avg.keys())) + shift

            sync_vel = np.array(list(sync_vel_dict_avg.values()))
            
            sync_average = filter.window_average(sync_vel/scale, window = window)
            sync_index = np.where(sync_average > thresh_mag)[0]

            argmins, argmins1, d2_sorted = knn.knn(np.expand_dims(ref_time[ref_index], axis = 1), np.expand_dims(sync_time[sync_index], axis = 1))

            correlation = np.corrcoef(sync_average[sync_index][argmins1], ref_average[ref_index][argmins], rowvar = True)[0,1]
            
            #correlation = correlation_signed
            correlation_array_loop.append(correlation)
            error_array.append(correlation)
            shift_array.append(shift)
            if best_correlation < correlation:
                best_scale = scale
                best_shift = shift
                best_correlation = correlation

                best_sync_time = sync_time
                best_sync_average = sync_average

            shift_array_loop.append(shift)
            error_array_loop.append(correlation)
            ####################
        if  save_dir is not None:
            fig1, ax1 = plt.subplots(1, 1)
            ax1.plot(shift_array_loop, error_array_loop, c = 'r')
            ax1.set_yscale("linear")

            fig2, ax2 = plt.subplots(1, 1)
            ax2.plot(shift_array_loop, correlation_array_loop, c = 'r')
            ax2.set_yscale("linear")

            if name is not None:
                #fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + str(scale) + '_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
                fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + trial_name + '_' + str(scale) + '_' + '.png')
            else:
                #fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + str(scale) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
                fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + trial_name + '_' + str(scale) + '_' + '.png')    
            plt.close('all')
            
    if  save_dir is not None:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        ax1.plot(best_sync_time, best_sync_average, c = 'r')
        ax1.plot(ref_time, ref_average, c = 'b')
        ax1.set_title(str(best_correlation))

        if name is not None:
            fig.savefig(save_dir + '/search_time/' + name + '/' + 'best_' + trial_name + '_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
        else:
            fig.savefig(save_dir + '/search_time/' + name + '/' + 'best_average_' + trial_name + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')  

        fig1, ax2 = plt.subplots(1, 1)
        ax2.set_yscale("linear")
        ax2.plot(shift_array, error_array, c = 'r')
        ax2.set_title(str(best_correlation))

        plt.close('all')

    shifted_time = best_scale*np.array(list(data_sync.keys())) + best_shift
    argmins, argmins1, d2_sorted = knn.knn(np.expand_dims(list(data_ref.keys()), axis = 1), np.expand_dims(shifted_time, axis = 1))

    sync_dict = {}
    for i in range(len(argmins)):
        sync_dict[list(data_ref.keys())[argmins[i]]] = int(list(data_sync.keys())[argmins1[i]])

    return best_shift, best_scale, sync_dict

def icp(data_ref, data_sync, best_shift_array, best_scale_array, sync_dict_array, save_dir = None, name = '_cam_'):

    icp_rot_array = []
    icp_init_rot_array = []
    icp_shift_array = []
    init_ref_center_array = []
    init_sync_center_array = []

    time_shift_array = []
    time_scale_array = []
    #sync_dict_array = []

    index_array = []

    for i in range(len(data_sync)):
        #best_shift, best_scale, sync_dict = time_align(data_ref, data_sync[i], save_dir = save_dir, name = str(i))

        best_shift = best_shift_array[i]
        best_scale = best_scale_array[i]
        sync_dict = sync_dict_array[i]
        #print(sync_dict, " SYNC DICT")

        init_d, init_ref_center, init_sync_center, reflect_true = rot_shift_search_time_weight(data_ref, data_sync[i], sync_dict, best_scale, best_shift, save_dir = save_dir, name = str(i) + name)
        #init_d, init_ref_center, init_sync_center, reflect_true = rotation_pca.rotation(data_ref, data_sync[i], sync_dict, best_scale, best_shift, save_dir = save_dir, name = str(i))
        print(init_ref_center, i, " init")
        icp_rot, icp_shift, icp_init_rot, index = Icp2d.icp_normalize(data_ref, data_sync[i], sync_dict, init_d, init_ref_center, init_sync_center, reflect_true, save_dir = save_dir, name = str(i), best_scale = best_scale)
        #icp_rot, icp_shift, icp_init_rot, index = Icp2d.icp_no_vel(data_ref, data_sync[i], sync_dict, init_d, init_ref_center, init_sync_center, reflect_true, save_dir = save_dir, name = str(i), best_scale = best_scale)
        #icp_rot, icp_shift, icp_init_rot = Icp2d.icp_ransac(data_ref, data_sync[i], sync_dict, init_d, init_ref_center, init_sync_center, reflect_true, save_dir = save_dir, name = str(i))
        #
        #icp_rot = np.array([[1.0,0.0],[0.0,1.0]])
        #icp_shift = [0.0, 0.0]
        #icp_init_rot  = np.array([[1.0,0.0],[0.0,1.0]])
        
        #print(icp_rot, " icp rot")
        #print(icp_shift, " icp shift")
        #print(icp_init_rot, " icp_init_rot")
        #stop
        #index_array.append(index)
        '''
        icp_rot  = np.array([[1,0],
                  [0, 1]])
        
        icp_shift = [0,0]
        
        icp_init_rot = np.array([[np.cos(init_d), -np.sin(init_d)],
                  [np.sin(init_d),  np.cos(init_d)]])
        '''
        icp_rot_array.append(icp_rot)
        icp_init_rot_array.append(icp_init_rot)
        icp_shift_array.append(icp_shift)
        init_ref_center_array.append(init_ref_center)
        init_sync_center_array.append(init_sync_center)

        time_shift_array.append(best_shift)
        time_scale_array.append(best_scale)
        sync_dict_array.append(sync_dict)
    print(init_ref_center_array,  " ref CENTERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")

    return icp_rot_array, icp_init_rot_array, icp_shift_array, init_ref_center_array, init_sync_center_array, time_shift_array, time_scale_array, sync_dict_array, index_array

def one_to_one_match(dfB, dfA):
    knn = NearestNeighbors(n_neighbors=len(dfB)).fit(dfB)
    distances, indices = knn.kneighbors(dfA)

    matched = []
    pairs = []
    for indexA, candidatesB in enumerate(indices):
        personA = dfA.index[indexA]
        for indexB in candidatesB:
            if indexB not in matched:
                matched.append(indexB)
                personB = dfB.index[indexB]
                pairs.append([personA, personB])
                break

    return pairs

def rot_shift_search_time_weight(data_ref, data_sync, sync_time_dict, scale, offset, save_dir = None, name = None):
    if os.path.isdir(save_dir + '/search_rot/' + name) == False:
        os.mkdir(save_dir + '/search_rot/' + name)

    ref_coords = []
    ref_time = []
    ref_cor_time = []

    ref_dict = {}
    #print(sync_time_dict, " sync dict")
    #print("**********************************************")
    for fr in list(sync_time_dict.keys()):

        ref_dict[fr] = []
        for tr in list(data_ref[fr].keys()):
            
            ref_coords.append(data_ref[fr][tr][0:2])
            ref_time.append(fr)
            ref_cor_time.append(scale*sync_time_dict[fr] + offset)

            ref_dict[fr].append(data_ref[fr][tr][0:2])
    #print(scale, offset, " scale and offsetttt")
    sync_coords = []
    sync_time = []
    sync_cor_time = []

    sync_dict = {}
    
    for item in sync_time_dict.items():

        sync_dict[item[1]] = []
        for tr in list(data_sync[item[1]].keys()):
            
            sync_coords.append(data_sync[item[1]][tr][0:2])
            sync_time.append(item[1])
            sync_cor_time.append(item[0])
            sync_dict[item[1]].append(data_sync[item[1]][tr][0:2])

    #print(sync_cor_time)
    
    ref_center = np.mean(ref_coords, axis = 0)
    sync_center = np.mean(sync_coords, axis = 0)

    best_d = 0
    best_error = np.inf

    best_x = 1
    best_y = 1

    all_angles = []
    all_error = []
    reflect_true = False

    #################################################
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

    ref_normalize = np.transpose(np.stack([normalize_ref[:, 0]/ref_x_size, normalize_ref[:, 1]/ref_y_size]))
    sync_normalize = np.transpose(np.stack([normalize_rot[:, 0]/rot_x_size, normalize_rot[:, 1]/rot_y_size]))

    sync_space_time = []
    ref_space_time = []

    sync_time = []
    ref_time = []
    
    for i in sync_time_dict.keys():
    
        #sync_normalize1 = np.transpose(np.stack([(np.array(sync_dict[sync_time_dict[i]])[:, 0] - sync_center[0])/rot_x_size, (np.array(sync_dict[sync_time_dict[i]])[:, 1] - sync_center[1])/rot_y_size, i*np.ones(np.array(sync_dict[sync_time_dict[i]]).shape[0])]))
        #ref_normalize1 = np.transpose(np.stack([(np.array(ref_dict[i])[:, 0] - ref_center[0])/ref_x_size, (np.array(ref_dict[i])[:, 1] - ref_center[1])/ref_y_size, i*np.ones(np.array(ref_dict[i]).shape[0])]))
        #print((i*np.array(sync_dict[sync_time_dict[i]])[:, 1]).shape, " SYNC DICTTT")
        #print(((np.array(sync_dict[sync_time_dict[i]])[:, 0] - sync_center[0])/rot_x_size).shape, ((np.array(sync_dict[sync_time_dict[i]])[:, 1] - sync_center[1])/rot_y_size).shape, (i*np.ones(np.array(sync_dict[sync_time_dict[i]])[:, 1].shape)).shape, (i*np.ones(np.array(ref_dict[i])[:, 1].shape)).shape, " HELLO")
        sync_time.append((i*np.ones(np.array(sync_dict[sync_time_dict[i]])[:, 1].shape)))
        ref_time.append((i*np.ones(np.array(ref_dict[i])[:, 1].shape)))
        
        sync_normalize1 = np.transpose(np.stack([(np.array(sync_dict[sync_time_dict[i]])[:, 0] - sync_center[0])/rot_x_size, (np.array(sync_dict[sync_time_dict[i]])[:, 1] - sync_center[1])/rot_y_size]))
        ref_normalize1 = np.transpose(np.stack([(np.array(ref_dict[i])[:, 0] - ref_center[0])/ref_x_size, (np.array(ref_dict[i])[:, 1] - ref_center[1])/ref_y_size]))
        #print(sync_normalize1, " HELLOOSODASDADS")
        sync_space_time.append(sync_normalize1)
        ref_space_time.append(ref_normalize1)

    sync_normalize1 = np.concatenate(sync_space_time)
    ref_normalize1 = np.concatenate(ref_space_time)

    sync_time = np.concatenate(sync_time)
    ref_time = np.concatenate(ref_time)
    #print(sync_dict)
    #print("*** HIII ****")
    #print(ref_dict)
    for reflect in [False]:
        for d in range(50,300):

            #sync_normalize1 = np.transpose(np.stack([(np.array(sync_dict[sync_time_dict[i]])[:, 0] - sync_center[0])/rot_x_size, (np.array(sync_dict[sync_time_dict[i]])[:, 1] - sync_center[1])/rot_y_size]))
            #ref_normalize1 = np.transpose(np.stack([(np.array(ref_dict[i])[:, 0] - ref_center[0])/ref_x_size, (np.array(ref_dict[i])[:, 1] - ref_center[1])/ref_y_size]))
            #print(sync_normalize1)
            #stop
            rot_coords1, R, o, p, s = rotate(sync_normalize1, origin=[0,0], shift=[0,0], degrees=d, reflect = reflect)
            #print(rot_coords, " rot coords")
            #print(ref_normalize1.shape, rot_coords1.shape, sync_normalize1.shape, ref_time.shape, sync_time.shape, " lol lol lol")
            ref_normalize1_time = np.transpose(np.stack([ref_normalize1[:, 0], ref_normalize1[:, 1], ref_time]))
            rot_coords1_time = np.transpose(np.stack([rot_coords1[:, 0], rot_coords1[:, 1], sync_time]))
            #stop
            #print(ref_normalize1_time.shape, rot_coords1_time.shape, " THE ShaPE")
            #print(ref_normalize1.shape, rot_coords1.shape, " ROT COORDSSS")
            error = util.chamfer_distance(ref_normalize1_time, rot_coords1_time, metric='l2', direction='bi')
            #error = util.chamfer_distance(ref_normalize1, rot_coords1, metric='l2', direction='bi')
            
            if best_error > error:
                best_d = d
                best_error = error
                reflect_true = reflect

            
            all_angles.append(d)
            all_error.append(error)

    rot_normalize, R, o, p, s = rotate(sync_normalize1, origin=[0,0], shift=[0,0], degrees=best_d, reflect = reflect_true, scale_x = best_x, scale_y = best_y)
    #error = util.chamfer_distance_time(ref_coords, rot_coords, np.array(ref_time), scale*np.array(sync_time) + offset, scale*np.array(ref_cor_time) + offset, np.array(sync_cor_time), metric='l2', direction='bi')
    error = best_error
    if  save_dir is not None:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        #print(np.array(ref_coords)[::100, 0].shape,  "HIIII")
        ax1.scatter(all_angles, all_error)

        if name is not None:
            fig.savefig(save_dir + '/search_rot/' + name + '/error_' + str(name) + '_' + str(best_d) + '_degree_' + '.png')
        else:
            fig.savefig(save_dir + '/search_rot/' + name + '/' + 'error_' + str(best_d) + '_degree_' + '.png')
    plt.close('all')
    
    #print(np.array(rot_normalize).shape, " WAEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")

    if  save_dir is not None:
        fig, ax1 = plt.subplots(1, 1)
        #ax1.axis('square')
        #ax1.set_aspect('equal', adjustable='box')
        #ax1.set_yscale("linear")
        #print(np.array(ref_coords)[::100, 0].shape,  "HIIII")
        ax1.scatter(np.array(rot_normalize)[:, 0], np.array(rot_normalize)[:, 1], c = 'r')
        ax1.scatter(np.array(ref_normalize)[:, 0], np.array(ref_normalize)[:, 1], c = 'b')
        ax1.set_title(str(error))
        #ax1.axis('equal')
        #ax1.axis([-1.5*max(ref_x_size, rot_x_size), 1.5*max(ref_x_size, rot_x_size), -1.5*max(ref_y_size, rot_y_size), 1.5*max(ref_y_size, rot_y_size)])
        #ax1.axis([-1, 1, -1, 1])
        fig1, ax2 = plt.subplots(1, 1)
        #ax2.set_yscale("linear")
        #ax2.set_aspect('equal', adjustable='box')
        #ax2.axis('square')
        ax2.scatter(np.array(sync_normalize)[:, 0], np.array(sync_normalize)[:, 1], c = 'r')
        ax2.scatter(np.array(ref_normalize)[:, 0], np.array(ref_normalize)[:, 1], c = 'b')
        ax2.set_title("init")

        #ax2.axis([-1, 1, -1, 1])
        #ax2.axis([-1.5*max(ref_x_size, rot_x_size) + ref_center[0], 1.5*max(ref_x_size, rot_x_size) + ref_center[0], -1.5*max(ref_y_size, rot_y_size) + ref_center[1], 1.5*max(ref_y_size, rot_y_size) + ref_center[1]])
        if name is not None:
            fig.savefig(save_dir + '/search_rot/' + name + '/best_' + str(name) + '_' + str(best_d) + '_degree_' + '.png')
        else:
            fig.savefig(save_dir + '/search_rot/' + name + '/' + 'best_' + str(best_d) + '_degree_' + '.png')
        
        #ax1.set_aspect('equal', adjustable='box')
        #ax2.set_aspect('equal', adjustable='box')

        fig1.savefig(save_dir + '/search_rot/' + name + '/init.png')
    plt.close('all')
    
    return math.radians(best_d), ref_center, sync_center, reflect_true

def rot_shift_search_time(data_ref, data_sync, sync_time_dict, scale, offset, save_dir = None, name = None):
    if os.path.isdir(save_dir + '/search_rot/' + name) == False:
        os.mkdir(save_dir + '/search_rot/' + name)

    ref_coords = []
    ref_time = []
    ref_cor_time = []

    ref_dict = {}
    for fr in list(sync_time_dict.keys()):

        ref_dict[fr] = []
        for tr in list(data_ref[fr].keys()):
            
            ref_coords.append(data_ref[fr][tr][0:2])
            ref_time.append(fr)
            ref_cor_time.append(scale*sync_time_dict[fr] + offset)

            ref_dict[fr].append(data_ref[fr][tr][0:2])
    
    sync_coords = []
    sync_time = []
    sync_cor_time = []

    sync_dict = {}
    
    for item in sync_time_dict.items():

        sync_dict[item[1]] = []
        for tr in list(data_sync[item[1]].keys()):
            
            sync_coords.append(data_sync[item[1]][tr][0:2])
            sync_time.append(item[1])
            sync_cor_time.append(item[0])
            sync_dict[item[1]].append(data_sync[item[1]][tr][0:2])

    #print(sync_cor_time)
    
    ref_center = np.mean(ref_coords, axis = 0)
    sync_center = np.mean(sync_coords, axis = 0)
    print(ref_center, sync_center, " THE CENTERS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    best_d = None
    best_error = np.inf

    best_x = 1
    best_y = 1

    all_angles = []
    all_error = []
    reflect_true = False

    #################################################
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

    ref_normalize = np.transpose(np.stack([normalize_ref[:, 0]/ref_x_size, normalize_ref[:, 1]/ref_y_size]))
    sync_normalize = np.transpose(np.stack([normalize_rot[:, 0]/rot_x_size, normalize_rot[:, 1]/rot_y_size]))
    #print(ref_x_size)
    #print(ref_y_size)
    #print(rot_x_size)
    #print(rot_y_size)
    #print("asdsddasdad")
    for reflect in [False]:
        for d in range(0,360):
            for sc_x in [1]:
                for sc_y in [1]:

                    #print(d, " THIS IS THE DEGREEE")

                    #rot_coords, R, o, p, s = rotate(sync_coords, origin=sync_center, shift=ref_center, degrees=d, reflect = reflect)

                    #print(" AFTER ROT")
                    #error = util.chamfer_distance_time(ref_coords, rot_coords, np.array(ref_time), scale*np.array(sync_time) + offset, scale*np.array(ref_cor_time) + offset, np.array(sync_cor_time), metric='l2', direction='bi')
                    
                    #rot_coords1, R, o, p, s = rotate(sync_normalize, origin=sync_center, shift=ref_center, degrees=d, reflect = reflect, scale_x = sc_x, scale_y = sc_y)
                    '''
                    rot_coords1, R, o, p, s = rotate(sync_normalize, origin=np.array([0,0]), shift=np.array([0,0]), degrees=d, reflect = reflect, scale_x = sc_x, scale_y = sc_y)
                    print(np.array(ref_coords).shape,"dasad")
                    print("*********************")
                    
                    error1 = util.chamfer_distance(ref_normalize, rot_coords1, metric='l2', direction='bi')
                    '''
                    error_array = []
                    skip = 0
                    
                    for i in sync_time_dict.keys():
                        #print(np.array(sync_dict[sync_time_dict[i]]).shape)
                        #print(np.array(ref_dict[i]).shape)
                        #print("asdasdasdasdasd")
                        #######################
                        #print(np.array(sync_dict[sync_time_dict[i]]).shape, " SYNC DICT")

                        #######################
                    
                        #if skip%100 != 0:
                        #    skip = skip + 1
                        #    continue
                        

                        #sync_normalize1 = np.transpose(np.stack([(np.array(sync_dict[sync_time_dict[i]])[:, 0] - ref_center[0])/rot_x_size, (np.array(sync_dict[sync_time_dict[i]])[:, 1] - ref_center[1])/rot_y_size]))
                        #ref_normalize1 = np.transpose(np.stack([(np.array(ref_dict[i])[:, 0] - ref_center[0])/ref_x_size, (np.array(ref_dict[i])[:, 1] - ref_center[1])/ref_y_size]))

                        sync_normalize1 = np.transpose(np.stack([(np.array(sync_dict[sync_time_dict[i]])[:, 0] - sync_center[0])/rot_x_size, (np.array(sync_dict[sync_time_dict[i]])[:, 1] - sync_center[1])/rot_y_size]))
                        ref_normalize1 = np.transpose(np.stack([(np.array(ref_dict[i])[:, 0] - ref_center[0])/ref_x_size, (np.array(ref_dict[i])[:, 1] - ref_center[1])/ref_y_size]))
                        skip = skip + 1
                        rot_coords1, R, o, p, s = rotate(sync_normalize1, origin=[0,0], shift=[0,0], degrees=d, reflect = reflect)
                        #print(rot_coords, " rot coords")
                        error_frame = util.chamfer_distance(ref_normalize1, rot_coords1, metric='l2', direction='bi')
                        error_array.append(error_frame)
                    
                    #error = error1# + np.mean(error_array)
                    error = np.mean(error_array)
                    if best_error > error:
                        best_d = d
                        best_error = error
                        reflect_true = reflect

                        best_x = sc_x 
                        best_y = sc_y
                    
                    all_angles.append(d)
                    all_error.append(error)

    rot_normalize, R, o, p, s = rotate(sync_normalize, origin=[0,0], shift=[0,0], degrees=best_d, reflect = reflect_true, scale_x = best_x, scale_y = best_y)
    #error = util.chamfer_distance_time(ref_coords, rot_coords, np.array(ref_time), scale*np.array(sync_time) + offset, scale*np.array(ref_cor_time) + offset, np.array(sync_cor_time), metric='l2', direction='bi')
    error = best_error
    if  save_dir is not None:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        #print(np.array(ref_coords)[::100, 0].shape,  "HIIII")
        ax1.scatter(all_angles, all_error)

        if name is not None:
            fig.savefig(save_dir + '/search_rot/' + name + '/error_' + str(name) + '_' + str(best_d) + '_degree_' + '.png')
        else:
            fig.savefig(save_dir + '/search_rot/' + name + '/' + 'error_' + str(best_d) + '_degree_' + '.png')
    plt.close('all')
    
    #print(np.array(rot_normalize).shape, " WAEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
    if  save_dir is not None:
        fig, ax1 = plt.subplots(1, 1)
        ax1.axis('square')
        #ax1.set_aspect('equal', adjustable='box')
        #ax1.set_yscale("linear")
        #print(np.array(ref_coords)[::100, 0].shape,  "HIIII")
        ax1.scatter(np.array(rot_normalize)[:, 0], np.array(rot_normalize)[:, 1], c = 'r')
        ax1.scatter(np.array(ref_normalize)[:, 0], np.array(ref_normalize)[:, 1], c = 'b')
        ax1.set_title(str(error))
        #ax1.axis('equal')
        #ax1.axis([-1.5*max(ref_x_size, rot_x_size), 1.5*max(ref_x_size, rot_x_size), -1.5*max(ref_y_size, rot_y_size), 1.5*max(ref_y_size, rot_y_size)])
        ax1.axis([-1, 1, -1, 1])
        fig1, ax2 = plt.subplots(1, 1)
        #ax2.set_yscale("linear")
        #ax2.set_aspect('equal', adjustable='box')
        ax2.axis('square')
        ax2.scatter(np.array(sync_normalize)[:, 0], np.array(sync_normalize)[:, 1], c = 'r')
        ax2.scatter(np.array(ref_normalize)[:, 0], np.array(ref_normalize)[:, 1], c = 'b')
        ax2.set_title("init")

        ax2.axis([-1, 1, -1, 1])
        #ax2.axis([-1.5*max(ref_x_size, rot_x_size) + ref_center[0], 1.5*max(ref_x_size, rot_x_size) + ref_center[0], -1.5*max(ref_y_size, rot_y_size) + ref_center[1], 1.5*max(ref_y_size, rot_y_size) + ref_center[1]])
        if name is not None:
            fig.savefig(save_dir + '/search_rot/' + name + '/best_' + str(name) + '_' + str(best_d) + '_degree_' + '.png')
        else:
            fig.savefig(save_dir + '/search_rot/' + name + '/' + 'best_' + str(best_d) + '_degree_' + '.png')
        
        #ax1.set_aspect('equal', adjustable='box')
        #ax2.set_aspect('equal', adjustable='box')

        fig1.savefig(save_dir + '/search_rot/' + name + '/init.png')
    plt.close('all')
    
    return math.radians(best_d), ref_center, sync_center, reflect_true

def rot_shift_search(data_ref, data_sync, sync_time_dict, save_dir = None, name = None):
    '''
    print(data_ref.keys())
    print("********************************************")
    print(data_sync.keys())
    print("********************************************")
    print(sync_time_dict)
    '''
    if os.path.isdir(save_dir + '/search_rot/' + name) == False:
        os.mkdir(save_dir + '/search_rot/' + name)

    ref_coords = []
    for fr in list(sync_time_dict.keys()):
        for tr in list(data_ref[fr].keys()):
            
            ref_coords.append(data_ref[fr][tr][0:2])
    
    sync_coords = []
    for fr in list(sync_time_dict.values()):
        for tr in list(data_sync[fr].keys()):
            
            sync_coords.append(data_sync[fr][tr][0:2])
    
    ref_center = np.mean(ref_coords, axis = 0)
    sync_center = np.mean(sync_coords, axis = 0)

    #print(ref_center, sync_center, " THE CENTERS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    best_d = None
    best_error = np.inf

    reflect_true = False
    #for reflect in [False, True]:
    for reflect in [False]:
        for d in range(0,360):
            #print(np.array(ref_coords).shape, " SHAPEEEEE")

            #print(np.array(sync_coords).shape, np.array(sync_center).shape, np.array(ref_center).shape, " SHAPE SYNC REF")

            rot_coords, R, o, p, s = rotate(sync_coords, origin=sync_center, shift=ref_center, degrees=d, reflect = reflect)
            #print(np.array(ref_coords).shape, np.array(rot_coords).shape, " COORDSSSSSSSSSSSSSSSSSSS")
            error = util.chamfer_distance(ref_coords, rot_coords, metric='l2', direction='bi')

            if best_error > error:
                best_d = math.radians(d)
                best_error = error
                reflect_true = reflect
            '''
            if  save_dir is not None:
                fig, ax1 = plt.subplots(1, 1)
                ax1.set_yscale("linear")
                #print(np.array(ref_coords)[::100, 0].shape,  "HIIII")
                ax1.scatter(np.array(rot_coords)[::100, 0], np.array(rot_coords)[::100, 1], c = 'r')
                ax1.scatter(np.array(sync_coords)[::100, 0], np.array(sync_coords)[::100, 1], c = 'b')
                ax1.set_title(str(error))

                if name is not None:
                    fig.savefig(save_dir + '/search_rot/' + name + '/' + str(name) + '_' + str(d) + '_degree_' + '.png')
                else:
                    fig.savefig(save_dir + '/search_rot/' + name + '/' + 'average_' + str(d) + '_degree_' + '.png')
                
                plt.close('all')
            '''

    rot_coords, R, o, p, s = rotate(sync_coords, origin=sync_center, shift=ref_center, degrees=np.rad2deg(best_d), reflect = reflect_true)
    #print(np.array(ref_coords).shape, np.array(rot_coords).shape, " COORDSSSSSSSSSSSSSSSSSSS")
    error = util.chamfer_distance(ref_coords, rot_coords, metric='l2', direction='bi')

    if  save_dir is not None:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        #print(np.array(ref_coords)[::100, 0].shape,  "HIIII")
        ax1.scatter(np.array(rot_coords)[::10, 0], np.array(rot_coords)[::10, 1], c = 'r')
        ax1.scatter(np.array(ref_coords)[::10, 0], np.array(ref_coords)[::10, 1], c = 'b')
        ax1.set_title(str(error))

        fig1, ax2 = plt.subplots(1, 1)
        ax2.set_yscale("linear")
        
        ax2.scatter(np.array(sync_coords)[::10, 0], np.array(sync_coords)[::10, 1], c = 'r')
        ax2.scatter(np.array(ref_coords)[::10, 0], np.array(ref_coords)[::10, 1], c = 'b')
        ax2.set_title("init")

    if name is not None:
        fig.savefig(save_dir + '/search_rot/' + name + '/best_' + str(name) + '_' + str(best_d) + '_degree_' + '.png')
    else:
        fig.savefig(save_dir + '/search_rot/' + name + '/' + 'best_' + str(best_d) + '_degree_' + '.png')
    
    fig1.savefig(save_dir + '/search_rot/' + name + '/init.png')
    plt.close('all')
    return best_d, ref_center, sync_center

def icp_transform(data_ref, data_sync, sync_time_dict, init_angle, init_ref_center, init_sync_center, n_iter = 10, ransac_iter = 100, num_points = 15, thresh = 1.0, save_dir = None, name = None):
    
    if os.path.isdir(save_dir + '/ICP/' + name) == False:
        os.mkdir(save_dir + '/ICP/' + name)

    frame_sync = list(sync_time_dict.values())
    frame_ref = list(sync_time_dict.keys())

    R_init = np.array([[np.cos(init_angle), -np.sin(init_angle)],
                  [np.sin(init_angle),  np.cos(init_angle)]])

    best_init_sync_shift = init_sync_center
    #best_
    top_inlier = 0
    best_rot = np.array([[1,0], [0,1]])
    best_shift = init_ref_center
    for t in range(n_iter):
        for i in range(ransac_iter):
            frames = list(random.sample(frame_ref, num_points))

            X_fix = []
            X_mov = []

            for fr in frames:
                fix = list(data_ref[fr].values())
                mov = list(data_sync[sync_time_dict[fr]].values())

                X_fix = X_fix + fix
                X_mov = X_mov + mov
            
            X_fix =  np.array(X_fix)[:, :2]
            #print(np.array(X_mov)[:, :2].shape, " SHAPESEAES")
            #print(np.array(X_fix)[:, :2].shape, " SHAPESEAES")
            #best_rot @ (R_init @ np.transpose(np.array(X_mov)[:, :2] - init_sync_center)) 
            X_mov =  np.transpose(best_rot @ (R_init @ np.transpose(np.array(X_mov)[:, :2] - init_sync_center))) + best_shift
            
            #print(X_mov.shape, X_fix.shape, " HWEHAWEASDSDAS")
            ret = Icp2d.icp(X_mov, X_fix)

            #print(ret, " RET")
            '''
            X_fix =  np.concatenate((np.array(X_fix)[:, :2] - init_ref_center, np.zeros((np.array(X_fix).shape[0], 1))), axis = 1)
            X_mov =  np.concatenate((np.transpose(R_init @ np.transpose(np.array(X_mov)[:, :2] - init_sync_center)), np.zeros((np.array(X_mov).shape[0], 1))), axis = 1)
            print(X_fix, " x fix")
            print(np.array(X_fix).shape, " X FIX")
            print(np.array(X_mov).shape, " X MOV")
            # Create point cloud objects
            pc_fix = PointCloud(X_fix, columns=["x", "y", "z"])
            pc_mov = PointCloud(X_mov, columns=["x", "y", "z"])

            # Create simpleICP object, add point clouds, and run algorithm!
            frame_icp = SimpleICP()
            frame_icp.add_point_clouds(pc_fix, pc_mov)
            H, X_mov_transformed, rigid_body_transformation_params = frame_icp.run(max_overlap_distance=1, correspondences=min(X_fix.shape[0], X_mov.shape[0]))

            print(H, " H")
            print(X_mov_transformed, " X MOV TRANSFOMRD")
            print( rigid_body_transformation_params, "  rigid_body_transformation_params")
            '''
            inlier_sync = []
            
            #print(data_ref, " ref")
            #stop
            for j in range(len(frame_ref)):
                #print(j, " helaeokasd") 
                f_fix = list(data_ref[frame_ref[j]].values())
                f_mov = list(data_sync[sync_time_dict[frame_ref[j]]].values())
                #print(np.array(f_fix).shape, " f fix")
                #print(np.array(f_mov).shape, " f mov")
                frame_fix = np.array(f_fix)[:, :2]
                #print(ret.shape, "ret")
                #print(R_init.shape, " R INIT")
                #print(best_init_sync_shift.shape, " SHAPEEEE")
                frame_mov = np.transpose((ret[:,0:2] @ R_init @ np.transpose(np.array(f_mov)[:, :2] - best_init_sync_shift))) + ret[:,2]
                #print(frame_mov.shape, frame_fix.shape, " FRAME MOVVVV")
                error = util.chamfer_distance(frame_fix, frame_mov, metric='l2', direction='bi')

                if error < thresh:
                    inlier_sync.append(frame_ref[j])
            
            if len(inlier_sync) > top_inlier:
                top_inlier = len(inlier_sync)
                best_rot = ret[:,0:2] 
                best_shift = ret[:,2]
    return

def rotate(p, origin=(0, 0), shift=(0,0), degrees=0.0, reflect = False, scale_x = 1.0, scale_y = 1.0):
    angle = np.deg2rad(degrees)

    R = np.array([[scale_x*np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  scale_y*np.cos(angle)]])

    
    if reflect == True:
        R = np.array([[scale_x*np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  -1*scale_y*np.cos(angle)]])

    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)

    s = np.atleast_2d(shift)

    #print(p.shape, o.shape, " HELLO")
    if p.shape[0] > 1:
        return np.squeeze((R @ (p.T-o.T) + s.T).T), R, o, p, s
    else:
        return (R @ (p.T-o.T) + s.T).T, R, o, p, s