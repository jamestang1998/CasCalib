import sys
import json
import run_calibration_ransac
import data
import pickle
from datetime import datetime
import os
import matplotlib.image as mpimg
today = datetime.now()

#The name of the run is the current date
name = str(today.strftime('%Y%m%d_%H%M%S'))

#parsing arguments
detections_path = '/local/tangytob/Summer2021/DCPose/demo/input_epfl/All_detections_n_tracks/'
output_p = '/local/tangytob/Summer2023/camera_calibration_synchronization/calibration/' + name 
frame_path = '/local/tangytob/Summer2021/DCPose/demo/input_epfl/'

#detections_list = list(os.listdir(detections_path))

detections_list = [
'campus7-c2_avi_alphapose-results.json'
#'match5-c2_avi_alphapose-results.json',
#'passageway1-c0_avi_alphapose-results.json',
#'passageway1-c1_avi_alphapose-results.json',
#'passageway1-c2_avi_alphapose-results.json',
#'passageway1-c3_avi_alphapose-results.json',
#'terrace2-c0_avi_alphapose-results.json',
#'terrace2-c1_avi_alphapose-results.json',
#'terrace2-c2_avi_alphapose-results.json',
#'terrace2-c3_avi_alphapose-results.json'
]

'''
detections_list = [
'campus4-c0_avi_alphapose-results.json',
'campus4-c1_avi_alphapose-results.json',
'campus4-c2_avi_alphapose-results.json',
'campus7-c0_avi_alphapose-results.json',
'campus7-c1_avi_alphapose-results.json',
'campus7-c2_avi_alphapose-results.json',
'match5-c2_avi_alphapose-results.json',
'passageway1-c0_avi_alphapose-results.json',
'passageway1-c1_avi_alphapose-results.json',
'passageway1-c2_avi_alphapose-results.json',
'passageway1-c3_avi_alphapose-results.json',
'terrace2-c0_avi_alphapose-results.json',
'terrace2-c1_avi_alphapose-results.json',
'terrace2-c2_avi_alphapose-results.json',
'terrace2-c3_avi_alphapose-results.json']
'''

#Reading hyperparameter

if os.path.isdir(output_p) == False:
    os.mkdir(output_p)

for det in detections_list:

    cn = det.split('_')[0] + '_' + det.split('_')[1]
    frame_path_full = frame_path + '/' + cn + '/00000000.jpg'

    img = mpimg.imread(frame_path_full, format='jpeg')

    with open(detections_path + det, 'r') as f:
        datastore = json.load(f)

    with open('configuration.json', 'r') as f:
        configuration = json.load(f)

    output_path = output_p +'/' + cn
    if os.path.isdir(output_path) == False:
        os.mkdir(output_path)

    img_width = img.shape[1]
    img_height = img.shape[0]

    datastore = data.alphapose_tracking_dataloader(datastore)

    #Calibration Algorithm
    ankles, cam_matrix, normal, ankleWorld, ransac_focal, datastore_filtered = run_calibration_ransac.run_calibration_ransac(datastore, 'hyperparameter.json', None, img_width, img_height, '', str(datetime.now().strftime('%Y%m%d_%H%M%S')), skip_frame = configuration['skip_frame'], max_len = configuration['max_len'], min_size = configuration['min_size'])

    #Saving json file and pickle file
    save_dict = {"cam_matrix":cam_matrix.tolist(), "ground_normal":normal.tolist(), "ground_position":ankleWorld}
    print(save_dict, " save_dict")
    s = json.dumps(save_dict)
    open(output_path + '/calibration.json',"w").write(s)

    #Saving to pickle file
    with open(output_path + '/calibration.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #all ankles
    ankle_dict = {"ankles":ankles}

    with open(output_path + '/ankles.pickle', 'wb') as handle:
        pickle.dump(ankle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)