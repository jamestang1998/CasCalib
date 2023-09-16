import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotting
import util
from scipy.spatial.distance import directed_hausdorff
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
import torch
import scipy
import scipy.fftpack
import scipy.signal
import sklearn
import knn
# two kinds of velocity?
# mean position/ time
# set distance / time


def window_average(sequence, window = 5):
    average = np.convolve(sequence, np.ones(window)/window, mode='same')
    return average

#input is actual coords
def search_scale_translation_multi(coords, coords_ref, save_dir = None, name = None, end = True):

    window = 15
    dilation = 5*15

    #coords_ref_time = np.arange(coords_ref[2,:][0], coords_ref[2,:][-1])

    '''
    coords_ref_interpolate = torch.from_numpy(util.interpolate(coords_ref[:2,:], coords_ref[2,:], coords_ref_time))
    ref_vel = torch.squeeze(util.central_diff(torch.unsqueeze(coords_ref_interpolate, dim = 1).double(), time = 1, dilation = dilation))
    ref_average = window_average(np.linalg.norm(ref_vel, axis = 0), window = window)
    '''

    ref_dict_interp, coords_ref_time = util.interpolate_multi(coords_ref)
    #print(ref_dict_interp, " multi")
    ref_diff_array = util.diff_multi(ref_dict_interp, dilation = dilation)
    ref_vel = util.div_multi(ref_diff_array, time = 1)
    ref_average = window_average(ref_vel, window = window)

    #print(ref_vel, " vel multi")
    #print(ref_average, " average multi")
    '''
    for ra in range(100):
        print(ref_dict_interp[list(ref_dict_interp.keys())[ra]], " MULTI INTERP")
    '''
    
    #for ra in range(100):
    #    print(ref_average[ra], " verage multi")
    
    #print(len(ref_vel), len(ref_average), len(ref_diff_array), " ref diff array")
    #print(len(list(ref_dict_interp.keys())), len(list(coords_ref_time)), " DICT LENGTH !!!!!!!!!!!!!!!!!!!")

    sync_dict_interp, coords_time = util.interpolate_multi(coords)
    sync_diff_array = util.diff_multi(sync_dict_interp, dilation = dilation)

    #for ra in range(100):
    #    print(sync_dict_interp[list(sync_dict_interp.keys())[ra]], " MULTI sync INTERP")
    
    #print(sync_dict_interp, " mutli sync_diff_array")
    coords_ref_time = np.array(coords_ref_time)
    coords_time = np.array(coords_time)

    if save_dir is not None:
        if os.path.isdir(save_dir + '/error') == False:
            os.mkdir(save_dir + '/error')
            os.mkdir(save_dir + '/mag_plots')
            os.mkdir(save_dir + '/search')

    best_scale = 1
    best_shift = 0
    best_correlation = -np.inf

    best_sync_time = None
    best_sync_average = None

    #################
    thresh_mag = 0.02
    ref_index = np.where(ref_average > thresh_mag)[0]
    #################
    #print(coords_ref_time)
    #print(coords_time)
    #stop
    '''
    coords_time = np.arange(coords[2,:][0], coords[2,:][-1])
    coords_interpolate = torch.from_numpy(util.interpolate(coords[:2,:], coords[2,:], coords_time))
    '''
    
    for shift in list(range(-int(coords_time.shape[0]/64), int(coords_time.shape[0]/64))):
        for scale in list(np.linspace(0.5, 2, 5)) + [1.0]:

            sync_time = scale*coords_time + shift
            
            sync_vel = util.div_multi(sync_diff_array, time = scale)
            #print(sync_vel, " HIII")
            '''
            print(np.array(sync_vel).shape, " HELLOOOO !!!!")
            print(np.array(sync_time).shape, " HELLOOOO !!!!")
            print(coords_time[-1], " coords time HELLOOOO !!!!")
            '''
            sync_average = window_average(sync_vel, window = window)
            #print(np.array(sync_average).shape, " sync_averagesync_averagesync_averagesync_average !!!!")
            #sync_vel = torch.squeeze(util.central_diff(torch.unsqueeze(coords_interpolate, dim = 1).double(), time = scale, dilation = dilation))

            #sync_average = window_average(np.linalg.norm(sync_vel, axis = 0), window = window)
            '''
            for ra in range(500):
                print(sync_average[ra], ra, " sync average multi")
            '''
            #print(sync_average, " sync average !!!!!!!!!!!!!!!!!!!!!!!!!")
            sync_index = np.where(sync_average > thresh_mag)[0]

            #print(sync_index, " sync index")
            '''
            print(ref_average.shape, " ref average")
            print(coords_ref_time.shape, " coord ref trime")
            print(ref_index.shape, " ref shape")
            print(ref_index[-1], " last ref shape")
            '''
            argmins1, argmins, d2_sorted = knn.knn(np.expand_dims(coords_ref_time[ref_index], axis = 1), np.expand_dims(scale*coords_time[sync_index] + shift, axis = 1))

            correlation = np.corrcoef(sync_average[sync_index][argmins], ref_average[ref_index][argmins1], rowvar = True)[0,1]
            
            if best_correlation < correlation:
                best_scale = scale
                best_shift = shift
                best_correlation = correlation

                best_sync_time = sync_time
                best_sync_average = sync_average

                #print(best_sync_time)
                #print(sync_average)
            
            if  save_dir is not None and end == False:
                fig, ax1 = plt.subplots(1, 1)
                ax1.set_yscale("linear")
                ax1.plot(sync_time, sync_average, c = 'r')
                ax1.plot(coords_ref_time, ref_average, c = 'b')
                ax1.set_title(str(correlation))

                if name is not None:
                    fig.savefig(save_dir + '/search/' + str(name) + '_' + str(shift) + '_' + str(scale) +'_' + '.png')
                else:
                    fig.savefig(save_dir + '/search/average_' + str(shift) + '_' + str(scale) +'_' + '.png')
                
                plt.close('all')
    '''
    if  save_dir is not None and end == True:
        #print(len(list(ref_dict_interp.keys())), " multi coords_ref_interpolate")
        #print(np.array(list(ref_dict_interp.values())).shape, "ref shape")
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        ax1.plot(list(ref_dict_interp.keys()), np.linalg.norm(np.squeeze(np.array(list(ref_dict_interp.values()))), axis = 1), c = 'r')
        ax1.set_title(str("ref interpolate"))

        if name is not None:
            fig.savefig(save_dir + '/search/interp_' + str(name) + '_' + str(shift) + '_' + str(scale) +'_' + '.png')
        else:
            fig.savefig(save_dir + '/search/interp_average_' + str(shift) + '_' + str(scale) +'_' + '.png')
    '''
    if  save_dir is not None and end == True:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        ax1.plot(best_sync_time, best_sync_average, c = 'r')
        ax1.plot(coords_ref_time, ref_average, c = 'b')
        ax1.set_title(str(best_correlation))

        if name is not None:
            fig.savefig(save_dir + '/search/' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
        else:
            fig.savefig(save_dir + '/search/average_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
    
    plt.close('all')        

    print(best_shift, best_scale, " best_shift, best_scale")

    #return coords
    return best_shift, best_scale, ref_dict_interp, sync_dict_interp, coords_ref_time, coords_time

#input is actual coords
def search_scale_translation(coords, coords_ref, save_dir = None, name = None, end = True):
    #print(coords.shape, coords_ref.shape, " coords")
    window = 15
    dilation = 5*15
    '''
    ref_diff = np.gradient(coords_ref[:2,:], axis = 1)
    sync_diff = np.gradient(coords[:2,:], axis = 1)

    ref_diff_time = np.gradient(coords_ref[2,:], axis = 0)
    
    ref_vel = np.divide(ref_diff, ref_diff_time)
    '''
    coords_ref_time = np.arange(coords_ref[2,:][0], coords_ref[2,:][-1])
    coords_ref_interpolate = torch.from_numpy(util.interpolate(coords_ref[:2,:], coords_ref[2,:], coords_ref_time))
    '''
    print(coords_ref_interpolate.shape, " coords_ref_interpolate!!!!!!!!!!!!!!!!!!")
    for ra in range(100):
        print(coords_ref_interpolate[:, ra], " not MULTI INTERP")
    '''
    #print(coords_ref_interpolate, " not multi")
    #print(coords_ref_interpolate.shape, " SHAPEEEE")
    #print(coords_ref[:2,:].shape, " SHAEOASEEAESAES")
    #ref_vel = torch.squeeze(util.central_diff(torch.unsqueeze(coords_ref[:2,:], dim = 1).double(), time = 1, dilation = dilation))
    ref_vel = torch.squeeze(util.central_diff(torch.unsqueeze(coords_ref_interpolate, dim = 1).double(), time = 1, dilation = dilation))

    #ref_average = (window_average(ref_vel[0, :], window = window) + window_average(ref_vel[1, :], window = window))/2.0
    ref_average = window_average(np.linalg.norm(ref_vel, axis = 0), window = window)

    #print(ref_vel, " not vel multi")
    
    #for ra in range(100):
    #    print(ref_average[ra], " not average multi")
    
    #print(ref_average, " not average multi")

    #ref_average = ref_average[ref_ind]
    
    if save_dir is not None:
        if os.path.isdir(save_dir + '/error') == False:
            os.mkdir(save_dir + '/error')
            os.mkdir(save_dir + '/mag_plots')
            os.mkdir(save_dir + '/search')

    best_scale = 1
    best_shift = 0
    best_correlation = -np.inf

    best_sync_time = None
    best_sync_average = None

    #################
    '''
    sync_time = coords[2,:]
    sync_diff_time = np.gradient(sync_time, axis = 0)
    sync_vel = np.divide(sync_diff, sync_diff_time)
    #sync_average = (window_average(sync_vel[0, :], window = window) + window_average(sync_vel[1, :], window = window))/2.0
    sync_average = window_average(np.linalg.norm(sync_vel, axis = 0), window = window)
    sync_ind = np.where((sync_average > 0.1).all(axis=0))[0]
    '''

    #scale = 1.0#ref_median/sync_median
    
    thresh_mag = 0.02#0.005
    ref_index = np.where(ref_average > thresh_mag)[0]

    #################

    #for shift in range(0, coords_ref.shape[1] - int(coords.shape[1]/2), dilation):
    coords_time = np.arange(coords[2,:][0], coords[2,:][-1])
    coords_interpolate = torch.from_numpy(util.interpolate(coords[:2,:], coords[2,:], coords_time))
    
    #for ra in range(100):
    #    print(coords_interpolate[:, ra], " not MULTI SYNC INTERP")
    #print(coords_interpolate, " NOT mutli sync_diff_array")

    for shift in list(range(-int(coords.shape[1]/64), int(coords.shape[1]/64))):
    #for shift in list(range(-int(coords.shape[1]/4), int(coords.shape[1]/4))):
        for scale in list(np.linspace(0.5, 2, 5)) + [1.0]:
           
            #print(shift, scale, " H E L L O")
            sync_time = scale*coords_time + shift
            #sync_diff_time = np.gradient(sync_time, axis = 0)
            #sync_vel = np.divide(sync_diff, sync_diff_time)*scale
            print(shift, scale, " 21312qweeqdas")
            sync_vel = torch.squeeze(util.central_diff(torch.unsqueeze(coords_interpolate, dim = 1).double(), time = scale, dilation = dilation))
            print("********************************************************************************")
            sync_average = window_average(np.linalg.norm(sync_vel, axis = 0), window = window)

            for ra in range(500):
                print(sync_average[ra], ra, " not sync average multi")
            sync_index = np.where(sync_average > thresh_mag)[0]
            '''     
            corr = util.find_match_time(scale*torch.from_numpy(coords_time[sync_index]) + shift, torch.from_numpy(coords_ref_time[ref_index]), 5)

            argmins = list(corr[:, 0])
            argmins1 = list(corr[:, 1])
            '''
            #print(coords_ref_time[ref_index].shape, (scale*coords_time[sync_index] + shift).shape, " HSHDAASDDS")
            argmins1, argmins, d2_sorted = knn.knn(np.expand_dims(coords_ref_time[ref_index], axis = 1), np.expand_dims(scale*coords_time[sync_index] + shift, axis = 1))

            correlation = np.corrcoef(sync_average[sync_index][argmins], ref_average[ref_index][argmins1], rowvar = True)[0,1]
            
            #print(best_correlation, correlation)
            if best_correlation < correlation:
                best_scale = scale
                best_shift = shift
                best_correlation = correlation

                best_sync_time = sync_time
                best_sync_average = sync_average
            
            if  save_dir is not None and end == False:
                fig, ax1 = plt.subplots(1, 1)
                ax1.set_yscale("linear")
                ax1.plot(sync_time, sync_average, c = 'r')
                ax1.plot(coords_ref_time, ref_average, c = 'b')
                ax1.set_title(str(correlation))

                if name is not None:
                    fig.savefig(save_dir + '/search/' + str(name) + '_' + str(shift) + '_' + str(scale) +'_' + '.png')
                else:
                    fig.savefig(save_dir + '/search/average_' + str(shift) + '_' + str(scale) +'_' + '.png')
                
                plt.close('all')
    
    if  save_dir is not None and end == True:
        print(coords_ref_interpolate.shape, " coords_ref_interpolate")
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        ax1.plot(np.linalg.norm(coords_ref_interpolate, axis = 0), c = 'r')
        ax1.set_title(str("ref interpolate"))

        if name is not None:
            fig.savefig(save_dir + '/search/interp_' + str(name) + '_' + str(shift) + '_' + str(scale) +'_' + '.png')
        else:
            fig.savefig(save_dir + '/search/interp_average_' + str(shift) + '_' + str(scale) +'_' + '.png')

    if  save_dir is not None and end == True:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        ax1.plot(best_sync_time, best_sync_average, c = 'r')
        ax1.plot(coords_ref_time, ref_average, c = 'b')
        ax1.set_title(str(best_correlation))

        if name is not None:
            fig.savefig(save_dir + '/search/' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
        else:
            fig.savefig(save_dir + '/search/average_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
    
    plt.close('all')        

    print(best_shift, best_scale, " best_shift, best_scale")

    #return coords
    return best_shift, best_scale

def average_align(coords, coords_ref, opt_iter = 5000, save_dir = None, x_lr = 1e-3, y_lr = 1e-3, t_lr = 1e-3):
    """
    Iterative Closest Point
    """
    coords_time = torch.tensor(list(range(coords.shape[0]))).double()
    coords_ref_time = torch.tensor(list(range(coords_ref.shape[0]))).double()

    rot_matrix_array = []
    translation_array = []
    rot_point = []

    print(coords_time.shape, coords.shape, " COORDS!!!!!")

    P_values = []
    sync_ref = []
    ref_sync = []

    corr = np.correlate(coords, coords_ref,"full")

    idx = np.argmax(corr);

    shift = coords.shape[0] - idx

    scale = np.max(coords)/np.max(coords_ref)

    print(shift, scale)

    coords_vec = np.stack((coords_time, coords))
    coords_ref_vec = np.stack((coords_ref_time, coords_ref))
    print(coords_vec.shape, coords_ref_vec.shape, " COORDS!!!!!")
    corr = np.squeeze(scipy.signal.correlate2d(coords_vec, coords_ref_vec, mode='valid'))
    corr_mean = np.mean(corr)
    print(corr.shape, corr_mean)
    stop

    
    coords_fft = scipy.fftpack.fft(coords)
    coords_ref_fft = scipy.fftpack.fft(coords_ref)
    print(np.abs(coords_fft))
    fig, ax1 = plt.subplots(1, 1)

    ax1.plot(np.abs(coords_fft), c = 'r')
    ax1.plot(np.abs(coords_ref_fft), c = 'b')
    fig.savefig(save_dir + '/mag_plots/' + 'fft' + '.png')
    '''
    
    scale_x = torch.nn.Parameter(torch.tensor([1.0]).double())
    scale_y = torch.nn.Parameter(torch.tensor([1.0]).double())
    translate = torch.nn.Parameter(torch.tensor([0.0]).double())
    
    #use SGD here
    optimizer = torch.optim.SGD([
            {'params': scale_x, 'lr': x_lr},
            {'params': scale_y, 'lr': y_lr},
            {'params': translate, 'lr':t_lr}
        ], weight_decay=0.0)
    

    min_error = np.inf
    min_params = None
    error_array = []
    print(coords_ref_time.shape, coords_ref.shape, " COORD REF")
    coords_ref_vec = torch.unsqueeze(torch.stack((coords_ref_time, coords_ref)), dim = 0)
    
    for t in range(opt_iter):
        print(translate.repeat(coords.shape[0]).shape, " CPPPPPRDSAEW")
        coords_vec = torch.unsqueeze(torch.stack((scale_x*coords_time + translate.repeat(coords.shape[0]), scale_y*coords)), dim = 0)
        
        #loss = torch.nn.functional.conv2d(weight = coords_ref_vec)
        #loss = torch.nn.CrossEntropyLoss()
        print(coords_vec.shape, coords_ref_vec.shape)
        error = torch.nn.functional.conv2d(input = coords_vec, weight = coords_ref_vec)#loss(coords_vec)

        if min_error > error:
            min_error = error.item()
        
        if t % 1000 == 0 and save_dir is not None:
            fig, ax1 = plt.subplots(1, 1)

            ax1.plot(list(range(len(error_array))),error_array, c = 'blue')

            if os.path.isdir(save_dir + '/error') == False:
                os.mkdir(save_dir + '/error')
                os.mkdir(save_dir + '/mag_plots')
            #print(argmins, " argmins")
            plotting.display_2d_plane([np.transpose(coords_ref.detach().cpu().numpy()), np.transpose(np.squeeze(point_transform.detach().cpu().numpy()))], save_dir + '/mag_plots', img, from_pickle, 1, 10, "iter_" + str(t) + '_' + str(i), sync = [argmins], sync1 = [argmins1])
            plotting.plot_magnitude_time(np.transpose(np.array(coords_ref.detach().numpy())), np.transpose(np.squeeze(point_transform.detach().cpu().numpy())), save_dir + '/mag_plots', str(t) + '_' + str(i), sync_array = argmins, sync_array1 = argmins1)
            
            plotting.plot_velocity(np.transpose(np.array(coords_ref.detach().numpy())), np.transpose(np.squeeze(point_transform.detach().cpu().numpy())), save_dir + '/mag_plots', str(t) + '_' + str(i), sync_array = argmins, sync_array1 = argmins1)
            fig.savefig(save_dir + '/error/' + str(t) + '_.png')
            #print(error)
            plt.close('all')
            
    optimizer.zero_grad()
    error.backward(retain_graph=True)

    optimizer.step()
    
    #return coords
    return P_values, sync_ref, ref_sync, rot_matrix_array[-1], translation_array[-1], rot_point[-1]
    '''

def search_align(coords, coords_ref, save_dir = None):
    """
    Iterative Closest Point
    """
    coords_time = np.array(list(range(coords.shape[0])))
    coords_ref_time = np.array(list(range(coords_ref.shape[0])))

    rot_matrix_array = []
    translation_array = []
    rot_point = []

    P_values = []
    sync_ref = []
    ref_sync = []
    
    scale_x = 1.0
    scale_y = 1.0
    translate = 0.0

    min_error = np.inf
    min_params = None
    error_array = []
    print(coords_ref_time.shape, coords_ref.shape, " COORD REF")
    coords_ref_vec = torch.unsqueeze(torch.stack((coords_ref_time, coords_ref)), dim = 0)
    
    for t in range(0, coords_ref.shape[0]):

        #cdist = scipy.spatial.distance.cdist( coords_time, coords_ref_time )
        coords_vec = torch.unsqueeze(torch.stack((scale_x*coords_time + translate.repeat(coords.shape[0]), scale_y*coords)), dim = 0)
        error = torch.nn.functional.conv2d(input = coords_vec, weight = coords_ref_vec)#loss(coords_vec)

        if min_error > error:
            min_error = error.item()
        
        if t % 1000 == 0 and save_dir is not None:
            fig, ax1 = plt.subplots(1, 1)

            ax1.plot(list(range(len(error_array))),error_array, c = 'blue')

            if os.path.isdir(save_dir + '/error') == False:
                os.mkdir(save_dir + '/error')
                os.mkdir(save_dir + '/mag_plots')
            #print(argmins, " argmins")
            plotting.display_2d_plane([np.transpose(coords_ref.detach().cpu().numpy()), np.transpose(np.squeeze(point_transform.detach().cpu().numpy()))], save_dir + '/mag_plots', img, from_pickle, 1, 10, "iter_" + str(t) + '_' + str(i), sync = [argmins], sync1 = [argmins1])
            plotting.plot_magnitude_time(np.transpose(np.array(coords_ref.detach().numpy())), np.transpose(np.squeeze(point_transform.detach().cpu().numpy())), save_dir + '/mag_plots', str(t) + '_' + str(i), sync_array = argmins, sync_array1 = argmins1)
            
            plotting.plot_velocity(np.transpose(np.array(coords_ref.detach().numpy())), np.transpose(np.squeeze(point_transform.detach().cpu().numpy())), save_dir + '/mag_plots', str(t) + '_' + str(i), sync_array = argmins, sync_array1 = argmins1)
            fig.savefig(save_dir + '/error/' + str(t) + '_.png')
            #print(error)
            plt.close('all')

    #return coords
    return P_values, sync_ref, ref_sync, rot_matrix_array[-1], translation_array[-1], rot_point[-1]