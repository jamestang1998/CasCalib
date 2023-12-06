import numpy as np
import util
np.set_printoptions(threshold=np.inf)

#COORDINATE SYSTEM
#Image: y point down, x points right
#3D: y point down, x points right, z points inwards (normal would have negative y, negative z (usually))

def calibration_focalpoint_lstq_failure_single(num_points, hv, av, hu, au, h, t1, t2, focal_predicted = None, upper_bound = np.inf, h_conf = [], al_conf = [], ar_conf = []):
    '''
    computes the camera intrsinsic parameters and normal vector of the ground plane
    
    Parameters: num_points: int
                    number of people used to solve the DLT equations
                hv: array
                    y camera coordinate of the head keypoint
                av: array
                    y camera coordinate of the ankle keypoint
                hu: array
                    x camera coordinate of the head keypoint
                au: array
                    x camera coordinate of the ankle keypoint
                h: float
                    assumed height of people in the scene
                t1: float
                    x focal center
                t2: float
                    y focal center
    Returns:    normal: (3,) np.array
                    normal vetor of the ground plane
                calcz: array
                    depths of each ankle
                focal_predicted: float
                    predicted focal length
                c_predicted: (3,3) np.array
                    predicted intrinsic camera matrix
    '''

    C = np.zeros([2*num_points,3 + num_points], dtype = float)
    
    for col in range(0, 2*num_points):
        if col % 2 == 0:
            C[col][1] = -1
            C[col][2] = hv[int(col/2)]
            C[col][3 + int(col/2)] = hv[int(col/2)] - av[int(col/2)]
            C[col + 1][0] = 1
            C[col + 1][2] = -hu[int(col/2)]
            C[col + 1][3 + int(col/2)] = au[int(col/2)] - hu[int(col/2)]
    try:

        if len(al_conf) == 0:
            U, S, Vh = np.linalg.svd(C)
            L = (Vh[-1, :])
        else:

            for i in range(num_points):

                avg_conf = (h_conf[i] + al_conf[i] + ar_conf[i])/3.0 
                C[2*i, :] = avg_conf*C[2*i, :]
                C[2*i + 1, :] = avg_conf*C[2*i + 1, :]
            U, S, Vh = np.linalg.svd(C)
            L = (Vh[-1, :])
    except:
        return None, None, None, None, None, None
    if focal_predicted is None:
        if len(al_conf) == 0:
            focal_predicted = compute_focal_failure_single_lstq(au, av, L, t1, t2, upper_bound = upper_bound)
        else:
            focal_predicted = compute_focal_failure_single_weighted_lstq(au, av, ar_conf, al_conf, L, t1, t2, upper_bound = upper_bound)

    if focal_predicted is None:
        return None, None, None, None, None, None
    
    h = float(h)
    t1 = float(t1)
    t2 = float(t2)
    
    L0 = float(L[0] - t1*L[2])/focal_predicted
    L1 = float(L[1] - t2*L[2])/focal_predicted
    L2 = float(L[2])
    
    normal = np.array([L0, L1, L2])
    c_predicted = np.array([[focal_predicted, 0, t1], [0, focal_predicted, t2], [0, 0, 1]])
    
    calcz = np.array(L[3:2*num_points])*h/(np.linalg.norm(normal) + 1e-18)

    normal = normal/(np.linalg.norm(normal) + 1e-18)
    calcz = np.absolute(calcz)
    
    return normal, calcz, focal_predicted, c_predicted, L, C

def compute_focal_failure_single_lstq(au, av, L, t1, t2, upper_bound = np.inf):
    
    '''
    computes the focal length given ankle detections and taking the average focal length over all pairs of ankles
    
    Parameters: au: np.array
                    array of x image coordinates of ankles
                av: np.array
                    array of y image coordinates of ankles
                L: np.array
                    solution from SVD
                h: float
                    assumed height of poses in meters
                t1: float
                    x focal center (image_width/2)
                t2: float 
                    y focal center (image_height/2)
                upper_bound: int
                    number of pairs of ankles to use for computing focal length (if none, then uses all n choose 2 ankles, which can be very slow)
    Returns:    focal_predicted: float
                    average focal length of inlier set
                    
    '''
        
    indices = np.array(list(range(len(au))))
    np.random.shuffle(indices)
    comb_array = util.random_combination(indices.tolist(), 2, upper_bound)
    
    A = []
    b = []
    for comb in comb_array:   

        eq1_a1 = -(L[0] - L[2]*t1)*(L[3 + comb[0]]*(au[comb[0]] - t1) - L[3 + comb[1]]*(au[comb[1]] - t1)) 
        eq1_a2 = -(L[1] - L[2]*t2)*(L[3 + comb[0]]*(av[comb[0]] - t2) - L[3 + comb[1]]*(av[comb[1]] - t2))
        eq1_b = (L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))
        
        #A = np.array([[eq1_a1, eq1_a2], [eq2_a1, eq2_a2]])
        #b = np.array([eq1_b, eq2_b])
        if eq1_b/(eq1_a1 + eq1_a2) < 0:
            continue
        A.append([eq1_a1 + eq1_a2])
        b.append(eq1_b)
    
    if len(b) == 0:
        return None
    soln = np.linalg.lstsq(np.array(A), np.array(b))

    f1_squared = 1/(soln[0][0] + 1e-18)

    if f1_squared < 0:
        return None
    f1 = np.sqrt(f1_squared)

    return f1

def compute_focal_failure_single_weighted_lstq(au, av, ar_conf, al_conf, L, t1, t2, upper_bound = np.inf):
       
    '''
    computes the focal length given ankle detections and taking the average focal length over all pairs of ankles
    
    Parameters: au: np.array
                    array of x image coordinates of ankles
                av: np.array
                    array of y image coordinates of ankles
                L: np.array
                    solution from SVD
                h: float
                    assumed height of poses in meters
                t1: float
                    x focal center (image_width/2)
                t2: float 
                    y focal center (image_height/2)
                upper_bound: int
                    number of pairs of ankles to use for computing focal length (if none, then uses all n choose 2 ankles, which can be very slow)
    Returns:    focal_predicted: float
                    average focal length of inlier set
                    
    '''
        
    indices = np.array(list(range(len(au))))
    np.random.shuffle(indices)
    comb_array = util.random_combination(indices.tolist(), 2, upper_bound)
    
    A = []
    b = []
    
    for comb in comb_array:   

        average_conf = (ar_conf[comb[0]] + al_conf[comb[0]] + ar_conf[comb[1]] + al_conf[comb[1]])/4.0

        eq1_a1 = -(L[0] - L[2]*t1)*(L[3 + comb[0]]*(au[comb[0]] - t1) - L[3 + comb[1]]*(au[comb[1]] - t1)) 
        eq1_a2 = -(L[1] - L[2]*t2)*(L[3 + comb[0]]*(av[comb[0]] - t2) - L[3 + comb[1]]*(av[comb[1]] - t2))
        eq1_b = (L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))
        
        #A = np.array([[eq1_a1, eq1_a2], [eq2_a1, eq2_a2]])
        #b = np.array([eq1_b, eq2_b])
        #print(eq1_b/(eq1_a1 + eq1_a2), " EQUATIONS !!")
        if eq1_b/(eq1_a1 + eq1_a2) < 0:
            continue
        A.append([average_conf*eq1_a1 + average_conf*eq1_a2])
        b.append(average_conf*eq1_b)

    if len(b) == 0:
        return None
    soln = np.linalg.lstsq(np.array(A), np.array(b))

    f1_squared = 1/(soln[0][0] + 1e-18)

    if f1_squared < 0:
        return None
    f1 = np.sqrt(f1_squared)

    return f1