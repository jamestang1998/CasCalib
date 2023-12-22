import util
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy
import torch.nn.functional as F
import torch
import random
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def subtract_keys(dictionary):
    if not dictionary:
        return dictionary  # Return the original dictionary if it's empty or None.

    # Get the first key and its corresponding value.
    first_key = next(iter(dictionary))
    # Subtract the first value from all other keys in the dictionary.
    result = {key - first_key: value for key, value in dictionary.items()}

    return result

def get_random_windowed_subset(array, window_size, start = None, offset = 0):
    if window_size <= 0 or window_size > len(array):
        raise ValueError("Window size must be greater than 0 and less than or equal to the array size.")
    
    start_index = start 

    if start_index is None:
        start_index = random.randint(offset, len(array) - window_size - offset)
    
    subset = array[start_index:start_index + window_size]
    
    return subset, start_index

def hungarian_assignment_one2one(A, B):
    distances = cdist(A, B)

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(distances)

    # Extract the matched elements from Array A and the corresponding elements from Array B
    matched_A = A[row_ind]
    matched_B = B[col_ind]

    # Compute the minimum pairwise norm
    '''
    min_pairwise_norm = np.linalg.norm(matched_A - matched_B, axis=1).sum()

    # Print the results
    print("Matched A:", matched_A)
    print("Matched B:", matched_B)
    print("Minimum pairwise norm:", min_pairwise_norm)
    '''

    return row_ind, col_ind

def central_diff(input, time = 1.0, dilation = 1):
    #input is (dim, batch, length)
    kernal_central = torch.tensor([[[-(1/2)*(1/time), 0, (1/2)*(1/time)]]]).double()
    kernal_forward = torch.tensor([[[-(1/time),(1/time)]]]).double()
    
    c_diff = F.conv1d(input, kernal_central, dilation = dilation)
    f_diff = F.conv1d(input, kernal_forward, dilation = dilation)

    return torch.cat((f_diff[:,:, :dilation], c_diff, f_diff[:,:, -dilation:]), dim=2)

def interpolate(points, time, grid):
    
    f = scipy.interpolate.interp1d(time, points)
    y_new = f(grid)
    return y_new

def get_ankles_heads_pose_dictionary(datastore, cond_tol = 0.8, keep = -1, keep_list = []):
    '''
    gets the ankle and head detections from the detections json file
    
    Parameters: datastore: data.py dataloader object
                    Stores the poses for accessing keypoints
                image_index: python int list
                    The selected indices of the json object that contain 2d keypoints
    Returns:    ppl_ankle_u: float np.array 
                    list of ankle x camera coordinates
                ppl_ankle_v: float np.array 
                    list of ankle y camera coordinates
                ppl_head_u: float np.array 
                    list of head x camera coordinates
                ppl_head_v: float np.array 
                    list of head y camera coordinates
                 
    '''

    dict_2d = {}
    
    init_dict = datastore.getData()

    for fr in list(init_dict.keys()):

        pose_array = init_dict[fr]

        dict_2d[fr] = {}
        for ppl in list(pose_array):
            '''
            left_ankle = ppl["left_ankle"]
            right_ankle = ppl["right_ankle"]
            #print(ppl)            
            #stop
            ankle_x, ankle_y = determine_foot(right_ankle,left_ankle, hgt_threshold=8.0, wide_threshold=10.0)
            
            ankle_x = ppl['hip'][0]
            head_x = (ppl["Thorax"][0])
            head_y = (ppl["Thorax"][1])

            dict_2d[fr][ppl['idx']] = ([ankle_x, ankle_y, head_x, head_y, ppl])
            '''
            left_ankle = ppl["left_ankle"]
            right_ankle = ppl["right_ankle"]

            ankle_left_conf = ppl["left_ankle"][2]
            ankle_right_conf = ppl["right_ankle"][2]

            head_conf = (ppl["left_shoulder"][2] + ppl["right_shoulder"][2])/2.0

            avg_cond = (ankle_left_conf + ankle_right_conf + ppl["left_shoulder"][2] + ppl["right_shoulder"][2])/4.0
            #avg_cond = ppl["bbox"][4]
            #print(avg_cond, " HELLO?")
            if fr > keep:
                if avg_cond < cond_tol and fr not in keep_list:
                    continue
            head_x = ((ppl["left_shoulder"][0]) + (ppl["right_shoulder"][0]))/2.0
            head_y = ((ppl["left_shoulder"][1]) + (ppl["right_shoulder"][1]))/2.0
            #ankle_x, ankle_y = determine_foot_bbox(right_ankle, left_ankle, datastore.getitem(ppl))
            max_size = max(np.linalg.norm(np.array([left_ankle[0], left_ankle[1]]) - np.array([head_x, head_y])), np.linalg.norm(np.array([right_ankle[0], right_ankle[1]]) - np.array([head_x, head_y])))
            hgt_thresh = 0.2*max_size
            wide_tresh = 0.2*max_size
            ankle_x, ankle_y = util.determine_foot(right_ankle,left_ankle, hgt_threshold=hgt_thresh, wide_threshold=wide_tresh)
                
            #head_x = (datastore.getitem(ppl)["Thorax"][0])
            #head_y = (datastore.getitem(ppl)["Thorax"][1])


            ppl.pop("bbox", None)
            if 'id' in ppl:
                dict_2d[fr][ppl['id']] = ([ankle_x, ankle_y, head_x, head_y, ankle_left_conf, ankle_right_conf, head_conf])
            else:
                if 0 not in dict_2d[fr]:
                    dict_2d[fr][0] = ([ankle_x, ankle_y, head_x, head_y, ankle_left_conf, ankle_right_conf, head_conf])
                else:
                    new_key = max(list(dict_2d[fr].keys())) + 1
                    dict_2d[fr][new_key] = ([ankle_x, ankle_y, head_x, head_y, ankle_left_conf, ankle_right_conf, head_conf])
   
        if len(dict_2d[fr]) == 0:
            dict_2d.pop(fr, None)
    return dict_2d

def get_ankles_heads_dictionary(datastore, cond_tol = 0.8, keep = -1, keep_list = []):
    '''
    gets the ankle and head detections from the detections json file
    
    Parameters: datastore: data.py dataloader object
                    Stores the poses for accessing keypoints
                image_index: python int list
                    The selected indices of the json object that contain 2d keypoints
    Returns:    ppl_ankle_u: float np.array 
                    list of ankle x camera coordinates
                ppl_ankle_v: float np.array 
                    list of ankle y camera coordinates
                ppl_head_u: float np.array 
                    list of head x camera coordinates
                ppl_head_v: float np.array 
                    list of head y camera coordinates
                 
    '''

    dict_2d = {}
    
    init_dict = datastore.getData()

    for fr in list(init_dict.keys()):

        pose_array = init_dict[fr]

        dict_2d[fr] = {}
        for ppl in list(pose_array):
            '''
            left_ankle = ppl["left_ankle"]
            right_ankle = ppl["right_ankle"]

            ankle_x, ankle_y = determine_foot(right_ankle,left_ankle, hgt_threshold=8.0, wide_threshold=10.0)
            
            ankle_x = ppl['hip'][0]
            head_x = (ppl["Thorax"][0])
            head_y = (ppl["Thorax"][1])
            '''
            left_ankle = ppl["left_ankle"]
            right_ankle = ppl["right_ankle"]

            ankle_left_conf = ppl["left_ankle"][2]
            ankle_right_conf = ppl["right_ankle"][2]

            avg_cond = (ankle_left_conf + ankle_right_conf + ppl["left_shoulder"][2] + ppl["right_shoulder"][2])/4.0
            #avg_cond = ppl["bbox"][4]
            
            if fr > keep:
                if avg_cond < cond_tol and fr not in keep_list:
                    #print(avg_cond, cond_tol, "HI this shoudlnt happen")
                    continue
            #print(fr, " THIS IS THE FRAME")

            head_x = ((ppl["left_shoulder"][0]) + (ppl["right_shoulder"][0]))/2.0
            head_y = ((ppl["left_shoulder"][1]) + (ppl["right_shoulder"][1]))/2.0
            #ankle_x, ankle_y = determine_foot_bbox(right_ankle, left_ankle, datastore.getitem(ppl))
            max_size = max(np.linalg.norm(np.array([left_ankle[0], left_ankle[1]]) - np.array([head_x, head_y])), np.linalg.norm(np.array([right_ankle[0], right_ankle[1]]) - np.array([head_x, head_y])))
            hgt_thresh = 0.2*max_size
            wide_tresh = 0.2*max_size
            ankle_x, ankle_y = util.determine_foot(right_ankle,left_ankle, hgt_threshold=hgt_thresh, wide_threshold=wide_tresh)
                
            #head_x = (datastore.getitem(ppl)["Thorax"][0])
            #head_y = (datastore.getitem(ppl)["Thorax"][1])

            head_conf = (ppl["left_shoulder"][2] + ppl["right_shoulder"][2])/2.0

            if 'id' in ppl:
                dict_2d[fr][ppl['id']] = ([ankle_x, ankle_y, head_x, head_y, ankle_left_conf, ankle_right_conf, head_conf])
            else:
                if 0 not in dict_2d[fr]:
                    dict_2d[fr][0] = ([ankle_x, ankle_y, head_x, head_y, ankle_left_conf, ankle_right_conf, head_conf])
                else:
                    new_key = max(list(dict_2d[fr].keys())) + 1
                    dict_2d[fr][new_key] = ([ankle_x, ankle_y, head_x, head_y, ankle_left_conf, ankle_right_conf, head_conf])

        
        if len(dict_2d[fr]) == 0:
            dict_2d.pop(fr, None)


    return dict_2d

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    # INCOPORATE TIME !!!
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist