import sys
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_directory)
import util
import os
import data
import json
from datetime import datetime
import csv
import run_calibration_ransac
import matplotlib.image as mpimg
today = datetime.now()


#The name of the run is the current date
name = str(today.strftime('%Y%m%d_%H%M%S'))

#Gets the hyperparamter from hyperparameter.json
threshold_euc, threshold_cos, angle_filter_video, confidence, termination_cond, num_points, h, iter, focal_lr, point_lr = util.hyperparameter('SingleViewCalib/hyperparameter.json')

hyperparam_dict = {"threshold_euc": threshold_euc, "threshold_cos": threshold_cos, "angle_filter_video": angle_filter_video, "confidence": confidence, "termination_cond": termination_cond, "num_points": num_points, "h": h, "optimizer_iteration" :iter, "focal_lr" :focal_lr, "point_lr": point_lr}

#Making the directories, eval is the accuracy wit hthe ground truth, output is the calibration saved as a pickle file, plot is the plots that are created during optimization.
if os.path.isdir('SingleViewCalib/output') == False:
    os.mkdir('SingleViewCalib/output')

if os.path.isdir('SingleViewCalib/eval') == False:
    os.mkdir('SingleViewCalib/eval')

if os.path.isdir('SingleViewCalib/plots') == False:
    os.mkdir('SingleViewCalib/plots')

if os.path.isdir('SingleViewCalib/output/run_' + name) == False:
    os.mkdir('SingleViewCalib/output/run_' + name)

if os.path.isdir('SingleViewCalib/eval/run_' + name) == False:
    os.mkdir('SingleViewCalib/eval/run_' + name)

with open('SingleViewCalib/eval/run_' + name + '/all_runs.csv','a') as f:
    writer = csv.writer(f)
    writer.writerow(['camera', 'subject','focal_error_ransac', 'focal_error', 'focal', 'normal_error'])
    f.close

with open('SingleViewCalib/eval/run_' + name + '/average.csv','a') as f:
    writer1 = csv.writer(f)
    writer1.writerow(['focal_error_ransac', 'focal_error', 'normal_error'])
    f.close

#camera_array = ["54138969","55011271","58860488","60457274"]
#The paths to the required files. Detections is the json file of 2d detections, frame paths is the path to the frames, and ground truth calibration contains the ground truth calibration (may not be available in general)

#10 29 2022, something is wrong with the multi person version even for human36m, try checking the frame matching parrt, i think that might be wrong.
tsai_cal = ['SingleViewCalib/terrace-tsai-c0.xml', 'SingleViewCalib/terrace-tsai-c1.xml', 'SingleViewCalib/terrace-tsai-c2.xml', 'SingleViewCalib/terrace-tsai-c3.xml']
name_folder = 'SingleViewCalib//example_calibration'

save_dir = 'SingleViewCalib/plots/run_' + name + name_folder

if os.path.isdir('SingleViewCalib/plots/run_' + name) == False:
    os.mkdir('SingleViewCalib/plots/run_' + name)

if os.path.isdir('SingleViewCalib/plots/run_' + name + name_folder) == False:
    os.mkdir('SingleViewCalib/plots/run_' + name + name_folder)

if os.path.isdir('SingleViewCalib/plots/run_' + name + name_folder + '/search_time') == False:
    os.mkdir('SingleViewCalib/plots/run_' + name + name_folder + '/search_time')

if os.path.isdir('SingleViewCalib/plots/run_' + name + name_folder + '/search_rot') == False:
    os.mkdir('SingleViewCalib/plots/run_' + name + name_folder + '/search_rot')

if os.path.isdir('SingleViewCalib/plots/run_' + name + name_folder + '/ICP') == False:
    os.mkdir('SingleViewCalib/plots/run_' + name + name_folder + '/ICP')

if os.path.isdir('SingleViewCalib/plots/run_' + name + name_folder + '/bundle') == False:
    os.mkdir('SingleViewCalib/plots/run_' + name + name_folder + '/bundle')

plot_scale = 1
line_amount = 50

ankles_array = []

factor = 10
start = 0
end = 0#factor
index_array = []
skip = 1

all_videos = os.listdir('/local/tangytob/Summer2021/DCPose/demo/input_epfl')
all_videos.remove('All_Videos')
all_videos = [ x for x in all_videos if ".zip" not in x ]

#scene_list = ['4p', '6p', 'campus4', 'campus7', 'match5', 'passageway1', 'terrace1', 'terrace2']
scene_list = ['terrace1']

detection_path = '/local/tangytob/Summer2021/DCPose/demo/input_epfl/All_detections_n_tracks/'
frame_path = '/local/tangytob/Summer2021/DCPose/demo/input_epfl'
#calibration_path = '/local/tangytob/Summer2023/multiview_synchronization/calibration/20230122_151501/'
#subject_array = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
#########################
detections_list = os.listdir('/local/tangytob/Summer2021/DCPose/demo/output_epfl/json')
frame_path_list = os.listdir('/local/tangytob/Summer2021/DCPose/demo/input_epfl')
#calibration_path_list = os.listdir(calibration_path)
gt_path = '/local/tangytob/Summer2023/multiview_synchronization/ground_truth/calibration/'
#########################
with open('/local/tangytob/Summer2023/multiview_synchronization/configuration.json', 'r') as f:
    configuration = json.load(f)
#11 04 2022 maybe u should pick a one to one matching of frames
for sub in list(scene_list):
    
    all_view = [x for x in all_videos if sub in x]
    
    detections_array = []

    all_view = sorted(all_view, key=lambda x: int(x[-5]))
    print(all_view, " ALL VIEWSSS")
    for av in all_view:
        detections_array.append(detection_path + av + '_alphapose-results.json')

    frame_array = []
    for cn in all_view:
        frame_array.append(frame_path + '/' + cn + '/00000000.jpg')

    frame_path_array = []
    for cn in all_view:
        frame_path_array.append(frame_path + '/' + cn)

    img1 = mpimg.imread(frame_array[0], format='jpeg')
    img_width1 = img1.shape[1] #width of image
    img_height1 = img1.shape[0] #height of image

    view_array = []
    view_calibration = []

    img_array = []

    pose_array = []

    frame_dict_array = []

    plane_average_array = []
    pose_dict_array = []
    plane_mean_array = []

    plane_dict_array = []

    datastore_array = []

    plane_matrix_array = []

    ankle_coord_2d = []

    gt_array = []
    #for i in range(len(detections_array)):
    for i in [0]:
        img = mpimg.imread(frame_array[i], format='jpeg')


        all_view = [x for x in all_videos if sub in x]
    
        detections_array = []

        '''
        print(all_view, " ALL VIEWSSS")
        for av in all_view:
            detections_array.append(detection_path + av + '_alphapose-results.json')

        for j in range(len(detections_array)):
            with open(detections_array[j], 'r') as f:
                points_2d = json.load(f)
        '''
        with open(detection_path + av + '_alphapose-results.json', 'r') as f:
            points_2d = json.load(f)
        print("************************")
        #print(points_2d)
        datastore_cal = data.alphapose_tracking_dataloader(points_2d, cond = 0.85)
        ankles, cam_matrix, normal, ankleWorld, ransac_focal, datastore_filtered = run_calibration_ransac.run_calibration_ransac(datastore_cal, 'SingleViewCalib/hyperparameter.json', img, img.shape[1], img.shape[0], sub + '_', name, skip_frame = configuration['skip_frame'], max_len = configuration['max_len'], min_size = configuration['min_size'])