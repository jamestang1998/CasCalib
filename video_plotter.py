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
import imageio

#The name of the run is the current date
name = 'ankle_path'

#Gets the hyperparamter from hyperparameter.json
threshold_euc, threshold_cos, angle_filter_video, confidence, termination_cond, num_points, h, iter, focal_lr, point_lr = util.hyperparameter('hyperparameter.json')

hyperparam_dict = {"threshold_euc": threshold_euc, "threshold_cos": threshold_cos, "angle_filter_video": angle_filter_video, "confidence": confidence, "termination_cond": termination_cond, "num_points": num_points, "h": h, "optimizer_iteration" :iter, "focal_lr" :focal_lr, "point_lr": point_lr}

#Making the directories, eval is the accuracy wit hthe ground truth, output is the calibration saved as a pickle file, plot is the plots that are created during optimization.
if os.path.isdir('./output') == False:
    os.mkdir('./output')

if os.path.isdir('./eval') == False:
    os.mkdir('./eval')

if os.path.isdir('./plots') == False:
    os.mkdir('./plots')

if os.path.isdir('./output/run_' + name) == False:
    os.mkdir('./output/run_' + name)

if os.path.isdir('./eval/run_' + name) == False:
    os.mkdir('./eval/run_' + name)

with open('./eval/run_' + name + '/all_runs.csv','a') as f:
    writer = csv.writer(f)
    writer.writerow(['camera', 'subject','focal_error_ransac', 'focal_error', 'focal', 'normal_error'])
    f.close

with open('./eval/run_' + name + '/average.csv','a') as f:
    writer1 = csv.writer(f)
    writer1.writerow(['focal_error_ransac', 'focal_error', 'normal_error'])
    f.close

#camera_array = ["54138969","55011271","58860488","60457274"]
#The paths to the required files. Detections is the json file of 2d detections, frame paths is the path to the frames, and ground truth calibration contains the ground truth calibration (may not be available in general)
detections = ['./example/detections/result__ALL_1.54138969.json', './example/detections/result__ALL 1.55011271.json']
frame_path = ['./example/frame/result__ALL_1.54138969_frame0.jpg', './example/frame/result__ALL 1.55011271_frame0.jpg']
calibration_path = ['/local/tangytob/Summer2021/DLT_focal/camera_calibration_validation_detections/output/run_20220620_160757/all_0_0_/calibration_0_0_.pickle', '/local/tangytob/Summer2021/DLT_focal/camera_calibration_validation_detections/output/run_20220620_160757/all_0_1_/calibration_0_1_.pickle']

ground_truth_calibration = './example/ground_truth_calibration/camera.json'

plot_scale = 1
line_amount = 50

ankles_array = []
for i in range(len(detections)):

    name_folder = '/example_calibration_' + str(i)

    save_dir = './plots/run_' + name + name_folder

    if os.path.isdir('./plots/run_' + name) == False:
        os.mkdir('./plots/run_' + name)

    if os.path.isdir('./plots/run_' + name + name_folder) == False:
        os.mkdir('./plots/run_' + name + name_folder)

    with open(calibration_path[i], 'rb') as handle:
        from_pickle = pickle.load(handle)

    from_pickle = {"ankleWorld":from_pickle['ankles'][:, 0], "cam_matrix":from_pickle['cam_matrix'], "normal":from_pickle['normal']}

    img = mpimg.imread(frame_path[i], format='jpeg')
    img_width = img.shape[1] #width of image
    img_height = img.shape[0] #height of image

    with open(detections[i], 'r') as f:
        datastore = json.load(f)

    datastore = data.dcpose_dataloader(datastore)
    au, av, hu, hv, h_conf, al_conf, ar_conf = util.get_ankles_heads(datastore, list(range(datastore.__len__())))
    print(len(au), " Number of ankles")

    for point in range(len(au)):
        #plotting.plot_plane(au, av, hu, hv, save_dir, img, from_pickle, plot_scale, line_amount, point, threshold_euc, threshold_cos, h)
        ankles_x, ankles_y = plotting.display_2d_grid([au[point]], [av[point]], [hu[point]], [hv[point]], save_dir, img, from_pickle, plot_scale, line_amount, point, threshold_euc, threshold_cos, h)
        ankles_array(np.stack(ankles_x, ankles_y, dim = 0))

    print(len(ankles_array), " ANKLES ARRAY")
    frames = sorted(os.listdir('./plots/run_' + name + name_folder))
    frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    images = []
    for fr in frames:
        filename = './plots/run_' + name + name_folder + '/' + fr
        images.append(imageio.imread(filename))
    imageio.mimsave('./gifs/' + str(i) +'.gif', images)