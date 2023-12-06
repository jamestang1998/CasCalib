import numpy as np
#import calibration_newfocal
import calibration_singlefocal
import util
#ALWAYS CHECK WHICH WAY THE NORMAL VECTOR IS POINTING

def ransac_search(datastore, termination_cond, img_width, img_height, num_points = 3, threshold_euc = 0.3, threshold_cos = 0.15, h = 1.6, f = None, image_index = None, calibration_dictionary_best = None, use_init = True, sort = 0, cond_tolerance = 20000, search_upper  = 5000, search_lower = 0, ransac_search_step = 500, post_ransac_search_step = 100):
    """ 
    Uses ransac algorithm to select the best N number of people based on maximal inlier set and run camera calibration on the selected people's keypoints
    
    Parameters: datastore: datastore: data.py dataloader object
                    stores the poses for accessing keypoints
                termination_cond: int
                    number of iterations
                img_width: int
                    Width of image
                img_height: int
                    Height of image
                num_points: int
                    Number of ankles used to solve DLT equations 
                threshold_euc: float
                    Euclidean threshold for inliers
                threshold_cos: float
                    Cosine threshold for inliers
                h: float
                    assumed height of the people
                f = int, optional (default = None)
                    given focal length, if available
                image_index = list
                    indices to be considered in the datastore, if none, all indices are considered.
                calibration_dictionary_best = None, optional (default = None)
                    given optimal calibration, if available
                use_init : boolean, optional (default = True)
                    Initalize the focal search with the RANSAC result
                sort = int, optional (default = 0)
                    if sort is 0, then the best solution in ransac is for the top inlier solutions is the one with the lowest condition number. 
                    if sort is 1, then the best solution is the minimum focal length solution in the top inlier solution set.
                    if sort is 2, then the best solution is the median focal length solution in the top inlier solution set.
                cond_tolerance = int, optional (default = 20000)
                    The maximum condition number for the inlier system before it defaults to the ransac solution.
                search_upper = int, optional (default  = 5000)
                    The upperbound for focal lengths in the search (Note that the ransac initalization is also included in the search so the actual output could be higher)
                search_lower = int, optional (default = 0)
                    The lowerbound for focal lengths in the search (Note that the ransac initalization is also included in the search so the actual output could be lower)
                ransac_search_step = int, optional (default = 500)
                    ransac focal length search step during the ransac iteration
                post_ransac_search_step = int, optional (default = 100)
                    search step for determining focal length for the inlier system of equations
    Returns:    calibration_dictionary_best: python dictionary
                    Python dictionary with the following keys:
                        normal: (3,) np.array
                            normal vector of ground plane
                        focal_predicted: float
                            focal length
                        cam_matrix: (3,3) np.array
                            intrinsic camera matrix
                        cam_inv: (3,3) np.array
                            inverse intrinsic camera matrix
                        ankleWorld: (3,) np.array
                            3d ankle used as the 3d position of the ground plane
                        world_coordinates: np.array
                            world_coordinates of all the ankles in the filtered dictionary
                        global_inlier: int
                            number of inliers
                        current: np.array
                            the indices of the people detections used to solve the DLT
                        global_error: float
                            Euclidean error of prediction
                        global_error_cos: float
                            cosine error of prediction
                        len(image_index): int
                            Total number of people considered by ransac
                        inlier_index: Int array
                            The indices of the inlier points
    """    
    ransac_dictionary = ransac(datastore, termination_cond, img_width, img_height, num_points, threshold_euc, threshold_cos, h, f, image_index, calibration_dictionary_best, sort = sort, ransac_search_step = ransac_search_step, search_upper  = search_upper, search_lower = search_lower)

    return ransac_dictionary
        

def ransac(datastore, termination_cond, img_width, img_height, num_points = 3, threshold_euc = 0.3, threshold_cos = 0.15, h = 1.6, f = None, image_index = None, calibration_dictionary_best = None, sort = 0, search_upper  = 5000, search_lower = 0, ransac_search_step = 500):    
    """ 
    Uses ransac algorithm to select the best N number of people based on maximal inlier set and run camera calibration on the selected people's keypoints
    
    Parameters: datastore: datastore: data.py dataloader object
                    stores the poses for accessing keypoints
                termination_cond: int
                    number of iterations
                img_width: int
                    Width of image
                img_height: int
                    Height of image
                num_points: int
                    Number of ankles used to solve DLT equations 
                threshold_euc: float
                    Euclidean threshold for inliers
                threshold_cos: float
                    Cosine threshold for inliers
                h: float
                    assumed height of the people
                f = int, optional (default = None)
                    given focal length, if available
                image_index = list
                    indices to be considered in the datastore, if none, all indices are considered.
                calibration_dictionary_best = None, optional (default = None)
                    given optimal calibration, if available
                sort = int, optional (default = 0)
                    if sort is 0, then the best solution in ransac is for the top inlier solutions is the one with the lowest condition number. 
                    if sort is 1, then the best solution is the minimum focal length solution in the top inlier solution set.
                    if sort is 2, then the best solution is the median focal length solution in the top inlier solution set.
                search_upper = int, optional (default = 5000)
                    The upperbound for focal lengths in the search (Note that the ransac initalization is also included in the search so the actual output could be higher)
                search_lower = int, optional (default = 0)
                    The lowerbound for focal lengths in the search (Note that the ransac initalization is also included in the search so the actual output could be lower)
                ransac_search_step = 500, optional (default = 500)
                    ransac focal length search step during the ransac iteration
    Returns:    calibration_dictionary_best: python dictionary
                    Python dictionary with the following keys:
                        normal: (3,) np.array
                            normal vector of ground plane
                        focal_predicted: float
                            focal length
                        cam_matrix: (3,3) np.array
                            intrinsic camera matrix
                        cam_inv: (3,3) np.array
                            inverse intrinsic camera matrix
                        ankleWorld: (3,) np.array
                            3d ankle used as the 3d position of the ground plane
                        world_coordinates: np.array
                            world_coordinates of all the ankles in the filtered dictionary
                        global_inlier: int
                            number of inliers
                        current: np.array
                            the indices of the people detections used to solve the DLT
                        global_error: float
                            Euclidean error of prediction
                        global_error_cos: float
                            cosine error of prediction
                        len(image_index): int
                            Total number of people considered by ransac
                        inlier_index: Int array
                            The indices of the inlier points
    """
    if image_index is None:
        image_index = list(range(datastore.__len__()))
    
    np.random.shuffle(image_index)
    point_set = util.random_combination(image_index, num_points, termination_cond)
    #print(point_set)
    #stop
    ppl_ankle_u, ppl_ankle_v, ppl_head_u, ppl_head_v, ppl_h_conf, ppl_al_conf, ppl_ar_conf = util.get_ankles_heads(datastore, image_index)

    joint_conf = (ppl_h_conf + ppl_al_conf + ppl_ar_conf)/3.0

    if calibration_dictionary_best is None:
        calibration_dictionary_best = {'weighted_inlier': 0, 'global_inlier': 0}

    all_runs = []
    '''
    if f is None:
        focal_array = list(range(search_lower, search_upper, ransac_search_step))
        focal_array.insert(0, None)
        if search_lower == 0:
            focal_array.remove(0)
    '''
    #for f in focal_array:
    for persons in point_set:        
        calibration_dictionary = {}
        ankle_u, ankle_v, head_u, head_v, h_conf, al_conf, ar_conf = util.get_ankles_heads(datastore, persons)

        #calibration_dictionary['normal'], calibration_dictionary['calcz'], calibration_dictionary['focal_predicted'], calibration_dictionary['cam_matrix'], calibration_dictionary['L'], calibration_dictionary['C'] = calibration_singlefocal_head_ankle.calibration_focalpoint_lstq_failure_single(num_points, head_v, ankle_v, head_u, ankle_u, h, img_width/2.0, img_height/2.0,  h_conf =  h_conf, al_conf = al_conf, ar_conf = ar_conf)
        calibration_dictionary['normal'], calibration_dictionary['calcz'], calibration_dictionary['focal_predicted'], calibration_dictionary['cam_matrix'], calibration_dictionary['L'], calibration_dictionary['C'] = calibration_singlefocal.calibration_focalpoint_lstq_failure_single(num_points, head_v, ankle_v, head_u, ankle_u, h, img_width/2.0, img_height/2.0, focal_predicted = f,  h_conf =  h_conf, al_conf = al_conf, ar_conf = ar_conf)       
        if calibration_dictionary['focal_predicted'] is None:
            continue
        if calibration_dictionary['focal_predicted'] <= 0.0:
            continue
        
        if calibration_dictionary['focal_predicted'] >= search_upper or calibration_dictionary['focal_predicted'] <= search_lower :
            continue

        calibration_dictionary['persons'] = persons

        if calibration_dictionary['normal'][1] > 0:
            calibration_dictionary['normal'] = -1*calibration_dictionary['normal']

        calibration_dictionary['cam_inv'] = np.linalg.inv(calibration_dictionary['cam_matrix'])
        
        ankle_2d_w = np.stack((ankle_u, ankle_v, np.ones(len(calibration_dictionary['calcz']))))
        calibration_dictionary['ankleWorld'] = np.average((calibration_dictionary['cam_inv'] @ ankle_2d_w)*np.absolute(calibration_dictionary['calcz']), axis = 1)

        calibration_dictionary['world_coordinates'] = util.plane_ray_intersection_np(ppl_ankle_u, ppl_ankle_v, calibration_dictionary['cam_inv'], calibration_dictionary['normal'], calibration_dictionary['ankleWorld'])
        
        ankle_ppl_2d = util.perspective_transformation(calibration_dictionary['cam_matrix'], calibration_dictionary['world_coordinates'])

        head_ppl_2d = util.perspective_transformation(calibration_dictionary['cam_matrix'], np.array(calibration_dictionary['world_coordinates']) + np.transpose(np.tile(calibration_dictionary['normal']*h, (calibration_dictionary['world_coordinates'].shape[1], 1))))

        head_2d_pred = np.stack((ppl_head_u, ppl_head_v))
        ankle_2d_pred = np.stack((ppl_ankle_u, ppl_ankle_v))

        head_vect_ransac_pred = head_ppl_2d - ankle_ppl_2d
        head_vect_2d_pred = head_2d_pred - ankle_2d_pred

        error_cos = np.ones(calibration_dictionary['world_coordinates'].shape[1]) - util.matrix_cosine(np.transpose(head_vect_ransac_pred), np.transpose(head_vect_2d_pred))
        error_euc = np.linalg.norm(head_ppl_2d - head_2d_pred, axis=0)/np.linalg.norm(head_vect_2d_pred, axis=0)
        error_norm_array = error_euc < threshold_euc 
        error_cos_array = error_cos < threshold_cos

        calibration_dictionary['world_coordinates'] = calibration_dictionary['ankleWorld']

        selected_persons = np.in1d(image_index, persons)

        inlier_array = (error_norm_array & error_cos_array) | selected_persons 
        inlier_index = np.where(inlier_array == True)
        
        calibration_dictionary['error_std'] = np.std(np.array(error_euc[inlier_index]))
        calibration_dictionary['global_error'] = np.average(np.array(error_euc[inlier_index]))
        calibration_dictionary['global_error_cos'] = np.average(np.array(error_cos[inlier_index]))
        calibration_dictionary['current'] = persons
        calibration_dictionary['global_inlier'] = list(inlier_array).count(True)
        calibration_dictionary['total_detections'] = len(image_index)
        calibration_dictionary['inlier_index'] = inlier_index[0]

        calibration_dictionary['weighted_inlier'] = np.sum(joint_conf[inlier_index[0]])

        all_runs.append(calibration_dictionary)
        #print(calibration_dictionary['weighted_inlier'],calibration_dictionary_best['weighted_inlier'], " INLIER")
        if (calibration_dictionary['weighted_inlier'] > calibration_dictionary_best['weighted_inlier'] or 'weighted_inlier' not in calibration_dictionary or 'world_coordinates' not in calibration_dictionary or 'focal_predicted' not in calibration_dictionary) or (calibration_dictionary['weighted_inlier'] == calibration_dictionary_best['weighted_inlier'] and calibration_dictionary_best['global_error'] > calibration_dictionary['global_error'] and calibration_dictionary_best['global_error_cos'] > calibration_dictionary['global_error_cos']):
            calibration_dictionary_best = calibration_dictionary  
    #print(calibration_dictionary_best)
    #print(calibration_dictionary_best['weighted_inlier'], " weighted inlierrr")
    '''
    top_runs = [run for run in all_runs if run['weighted_inlier'] == calibration_dictionary_best['weighted_inlier']]

    if sort == 0:
        for r in top_runs:
            r['cond'] = np.linalg.cond(r['C'] )

        top_runs = sorted(top_runs, key=lambda run: run['cond'])

    elif sort == 1 or sort == 2:
        top_runs = sorted(top_runs, key=lambda run: run['focal_predicted'])
    
    if sort == 0 or sort == 1:
        calibration_dictionary_best = top_runs[0]
    else:
        calibration_dictionary_best = top_runs[int(len(top_runs)/2)]
    '''
    return calibration_dictionary_best