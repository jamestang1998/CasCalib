import single_util
import single_ransac_refine
import single_calibration_singlefocal
import numpy as np
import single_plotting
import os
#import calibration_newfocal

def run_calibration_ransac(datastore, hyperparam_dict, img, img_width, img_height, run_name, result_name, use_init = True, skip_frame = 1, max_len = 1000, min_size = 0, line_amount = 50, plot_scale = 1, sort = 0, cond_tolerance = 20000, search_upper  = 5000, search_lower = 0, ransac_search_step = 50, post_ransac_search_step = 100, f_init = None):
    '''
    Selects indices of standing poses in detections files and runs the calibration algorithm
    Parameters: datastore: dcpose_dataloader object.
                    dcpose_dataloader object that contains detections (check data.py).
                hyperparam_dict: Dict,
                    Dictionary that contains the following keys:
                        threshold_euc: float
                            Euclidean threshold for RANSAC.
                        threshold_cos: float
                            Cosine threshold for RANSAC.
                        angle_filter_video: float
                            Angle threshold before a pose is considered non standing.
                        confidence: float
                            Confidence threshold for detection (detections with lower confidence than this get ignored).
                        termination_cond: int
                            Number of iterations for RANSAC.
                        num_points: int
                            Number of points to compute the DLT for each iterations of RANSAC.
                        h: float
                            Assumed (neck) height of the people in the scene.
                        iter: int
                            Number of iterations for pytorch optimizer
                        focal_lr: float
                            Learning rate for the focal length.
                        point_lr: float
                            Learning rate for the plane center.
                img: np.array
                    A frame from the sequence or video. If img is None, then there will be no plots, however the algorithm will run as long as img_width and img_height are given.
                img_width = float, optional (default = None)
                    width of image (usually its img.shape[1])
                img_height = float, optional (default = None)
                    height of image (usually its img.shape[0])
                run_name: String
                    Path to the folder to put image results in.
                result_name: String
                    Name of the subdirectory to put image results in (best to format it as run_name/result_name ...)
                use_init : boolean, optional (default = True)
                    Initalize the focal search with the RANSAC result
                skip_frame : int, optional (default = 1)
                    The amount of detection frames to skip (ex skip_frame = 1 means use every frame, frame = 2 means use every 2nd etc)
                max_len = int, optional (default = 1000)
                    The maximum amount of detections to be used (if detections exceed the max_len, then a random sample of size max_len is taken)
                min_size = int, optional (default = 0)
                    The minimum pixel height of detected peoples (People that are too small in the scene tend to be noisy and very far away)
                line_amount = int, optional (default = 50)
                    Amount of ground plane lines to plot.
                plot_scale = int, optional (default = 1)
                    The size of each grid (1 = 1 meter)
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

    Returns:    all_world_coordinates: (3,) np.array
                    World coordinates of the ankles in the detections file.
                calibration_dict['cam_matrix']: (3,3) np.array
                    The intrinsic amera matrix.
                calibration_dict['normal']: (3,) np.array
                    Normal vector of the ground plane.
                ransac_focal: int
                    Focal length returned by RANSAC
                datastore_filtered: dcpose_dataloader object.
                    The filtered datastore
    '''
    if os.path.isdir('./plots') == False:
        os.mkdir('./plots')
    
    threshold_euc, threshold_cos, angle_filter_video, confidence, termination_cond, num_points, h, iter, focal_lr, point_lr = util.hyperparameter(hyperparam_dict)
    
    if img is not None:
        img_width = img.shape[1] #width of image
        img_height = img.shape[0] #height of image

    if datastore.__len__() < num_points:
        print("NOT ENOUGH POINTS")
        return None, None, None, None, None, None, None, None

    datastore_filtered = util.select_indices(datastore, angle_filter_video, confidence, img_width, img_height, skip_frame, min_size = min_size, max_len = max_len)

    print(datastore_filtered.__len__(), " DATA STORE FILTERED")

    if datastore_filtered.__len__() < num_points:
        print("FILTERING IS TOO STRICT, TRY A LARGER NUMBER FOR THE ANGLE FILTER")
        return None, None, None, None, None, None, None, None
    
    #if os.path.isdir('./plots/run_' + run_name) == False:
    #    os.mkdir('./plots/run_' + run_name)

    if os.path.isdir('./plots/all_' + run_name) == False:
        os.mkdir('./plots/all_' + run_name)
    
    au, av, hu, hv, h_conf, al_conf, ar_conf = util.get_ankles_heads(datastore_filtered, list(range(datastore_filtered.__len__())))
    print(len(au), " ANKLE LENGTH")
    joint_conf = (h_conf + al_conf + ar_conf)/3.0

    calibration_dict = ransac_refine.ransac_search(datastore_filtered, termination_cond, img_width, img_height, num_points, threshold_euc, threshold_cos, h, f = f_init, image_index = None, calibration_dictionary_best = None, use_init = use_init, sort = sort, cond_tolerance = cond_tolerance, search_upper  = search_upper, search_lower = search_lower, ransac_search_step = ransac_search_step, post_ransac_search_step = post_ransac_search_step)

    ransac_focal = calibration_dict['focal_predicted']

    print(calibration_dict['focal_predicted'], " f RANSAC")
    print(calibration_dict['cam_matrix'], " c RANSAC")
    print(calibration_dict['normal'], " n RANSAC")

    inlier_index = list(calibration_dict['inlier_index'])
    print(len(inlier_index), " inliers amount")

    #################################
    calibration_dictionary = {}
    ankle_u, ankle_v, head_u, head_v, h_conf, al_conf, ar_conf = util.get_ankles_heads(datastore_filtered, list(calibration_dict['inlier_index']))
    joint_conf = (h_conf + al_conf + ar_conf)/3.0
    print(h, " THE HEIGHT !!!!!!")

    #num_points = len(ankle_u)
    if len(hv) > 500:
        ankle_avg = (ar_conf + al_conf + h_conf)/3.0
        index = sorted(range(len(ankle_avg)), key=lambda i: ankle_avg[i], reverse=True)[:500]

        head_v = head_v[index]
        ankle_v = ankle_v[index]
        head_u = head_u[index]
        ankle_u = ankle_u[index]
        h_conf = h_conf[index]
        al_conf = al_conf[index]
        ar_conf = ar_conf[index]

        #num_points = 500
    
    print(len(head_v), len(ankle_v), len(head_u), len(ankle_u), len(h_conf), len(al_conf), len(ar_conf), " HEAD AND ANKLE !!! ")
    calibration_dictionary['normal'], calibration_dictionary['calcz'], calibration_dictionary['focal_predicted'], calibration_dictionary['cam_matrix'], calibration_dictionary['L'], calibration_dictionary['C'] = calibration_singlefocal.calibration_focalpoint_lstq_failure_single(len(ankle_u), head_v, ankle_v, head_u, ankle_u, h, img_width/2.0, img_height/2.0, focal_predicted = f_init, upper_bound = np.inf, h_conf = h_conf, al_conf = al_conf, ar_conf = ar_conf)
    focal_batch = calibration_dictionary['focal_predicted']
    if calibration_dictionary['normal'] is None or calibration_dictionary['calcz'] is None :
        print("FAILED")
        return None, None, None, None, None, None, None, None
    
    if calibration_dictionary['normal'][1] > 0:
        calibration_dictionary['normal'] = -1*calibration_dictionary['normal']

    calibration_dictionary['cam_inv'] = np.linalg.inv(calibration_dictionary['cam_matrix'])
    
    ankle_2d_w = np.stack((ankle_u, ankle_v, np.ones(len(calibration_dictionary['calcz']))))
    calibration_dictionary['ankleWorld'] = np.average((calibration_dictionary['cam_inv'] @ ankle_2d_w)*np.absolute(calibration_dictionary['calcz']), axis = 1)
    #################################
    print(calibration_dictionary['focal_predicted'], " f BATCH")
    print(calibration_dictionary['cam_matrix'], " c BATCH")
    print(calibration_dictionary['normal'], " n BATCH")
    
    #**********************************
    #focal_opt, normal_opt, ankleWorld_opt, error_opt, focal_array_opt, normal_array_opt = pytorch_optimizer.optimization_focal_dlt_torch(calibration_dictionary, ankle_u, ankle_v, head_u, head_v, int(img_width/2.0), int(img_height/2.0), h, './plots/run_' + run_name + result_name, img, threshold_euc, threshold_cos, focal_lr = 1e-1, point_lr = 1e-3, iter = 10000, line_amount = 50, plot_scale = 1, conf_array = joint_conf)

    '''
    focal_opt, normal_opt, ankleWorld_opt, error_opt, focal_array_opt, normal_array_opt = pytorch_optimizer.single_view_optimization_torch_3d(calibration_dictionary, ankle_u, ankle_v, head_u, head_v, int(img_width/2.0), int(img_height/2.0), h, './plots/run_' + run_name + result_name, img, threshold_euc, threshold_cos, focal_lr = 1e-1, point_lr = 1e-3, iter = 10000, line_amount = 50, plot_scale = 1, conf_array = joint_conf)
    
    calibration_dictionary = {}
    calibration_dictionary['normal'] = normal_opt
    calibration_dictionary['ankleWorld'] = ankleWorld_opt
    calibration_dictionary['focal_predicted'] = focal_opt
    calibration_dictionary['cam_matrix'] = np.array([[focal_opt, 0.0, img_width/2.0], [0.0, focal_opt, img_height/2.0], [0.0, 0.0, 1.0]])
    calibration_dictionary['cam_inv'] = np.linalg.inv(calibration_dictionary['cam_matrix'])
    '''
    print(calibration_dictionary['focal_predicted'], " f opt")
    print(calibration_dictionary['cam_matrix'], " c opt")
    print(calibration_dictionary['normal'], " n opt")
    
    #**********************************
    t1 = img_width/2.0
    t2 = img_height/2.0
    
    ankles = util.plane_ray_intersection_np(au[inlier_index], av[inlier_index], calibration_dictionary['cam_inv'], calibration_dictionary['normal'], calibration_dictionary['ankleWorld'])
    ankles = np.transpose(ankles)
    
    focal_predicted = calibration_dictionary['focal_predicted']
    normal = calibration_dictionary['normal']

    ankleWorld = calibration_dictionary['ankleWorld'] 

    print(ankleWorld, " Ankleworld BEFORE")

    if img is not None:

        plotting.plot_plane(au, av, hu, hv, './plots/all_' + run_name, img, calibration_dictionary, plot_scale, line_amount, str(result_name), threshold_euc, threshold_cos, h)
        plotting.display_2d_grid(au, av, hu, hv, './plots/all_' + run_name, img, calibration_dictionary, plot_scale, line_amount, str(result_name), threshold_euc, threshold_cos, h)
    
    ppl_ankle_u, ppl_ankle_v, ppl_head_u, ppl_head_v, ppl_h_conf, ppl_al_conf, ppl_ar_conf = util.get_ankles_heads(datastore_filtered, list(inlier_index))
    
    all_world_coordinates = util.plane_ray_intersection_np(ppl_ankle_u, ppl_ankle_v, calibration_dictionary['cam_inv'], calibration_dictionary['normal'], calibration_dictionary['ankleWorld'])
    
    return all_world_coordinates, calibration_dictionary['cam_matrix'], calibration_dictionary['normal'], calibration_dictionary['ankleWorld'], calibration_dictionary['focal_predicted'], focal_batch, ransac_focal, datastore_filtered

