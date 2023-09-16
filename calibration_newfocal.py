import numpy as np
import util
np.set_printoptions(threshold=np.inf)

#COORDINATE SYSTEM
#Image: y point down, x points right
#3D: y point down, x points right, z points inwards (normal would have negative y, negative z (usually))
def get_dlt(num_points, hv, av, hu, au):
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

    U, S, Vh = np.linalg.svd(C)
    L = (Vh[-1, :])

    return L, C

def calibration_focalpoint_lstq(num_points, hv, av, hu, au, h, t1, t2, focal_predicted = None, upper_bound = np.inf):
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
    
    hv = np.float128(hv)
    av = np.float128(av)
    hu = np.float128(hu)
    au = np.float128(au)
    h = np.float128(h)
    t1 = np.float128(t1)
    t2 = np.float128(t2)
    
    C = np.zeros([2*num_points,3 + num_points], dtype = float)
    
    for col in range(0, 2*num_points):
        if col % 2 == 0:
            C[col][1] = -1
            C[col][2] = hv[int(col/2)]
            C[col][3 + int(col/2)] = hv[int(col/2)] - av[int(col/2)]
            C[col + 1][0] = 1
            C[col + 1][2] = -hu[int(col/2)]
            C[col + 1][3 + int(col/2)] = au[int(col/2)] - hu[int(col/2)]
    
    #print(C)
    U, S, Vh = np.linalg.svd(C)
    L = (Vh[-1, :])
    #print(C @ L, " HELLOOOO")
    #print(L)
    if focal_predicted is None:
        focal_predicted = compute_focal(au ,av ,L , t1, t2, upper_bound)
    
    h = float(h)
    t1 = float(t1)
    t2 = float(t2)
    
    L0 = float(L[0] - t1*L[2])
    L1 = float(L[1] - t2*L[2])  
    L2 = float(L[2]*focal_predicted)
    
    normal = np.array([L0, L1, L2])
    c_predicted = np.array([[focal_predicted, 0, t1], [0, focal_predicted, t2], [0, 0, 1]])
    
    calcz = np.array(L[3:2*num_points])*h*focal_predicted/(np.linalg.norm(normal) + 1e-18)

    normal = normal/(np.linalg.norm(normal) + 1e-18)
    calcz = np.absolute(calcz)
    
    return normal, calcz, focal_predicted, c_predicted, L, C

def compute_focal(au, av, L, t1, t2, upper_bound = np.inf):
    
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
        
    focal_equation = 0
    focal_array = []
    comb_array = util.random_combination(list(range(len(au))), 2, upper_bound)
    
    for comb in comb_array:     
        focal_num = -(L[0] - L[2]*t1)*(L[3 + comb[0]]*(au[comb[0]] - t1) - L[3 + comb[1]]*(au[comb[1]] - t1)) - (L[1] - L[2]*t2)*(L[3 + comb[0]]*(av[comb[0]] - t2) - L[3 + comb[1]]*(av[comb[1]] - t2))
        focal_den = (L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))

        focal_array.append(np.sqrt(np.absolute(focal_num/(focal_den + 1e-18))))
        focal_equation = focal_equation + np.sqrt(np.absolute(focal_num/(focal_den + 1e-18)))

        #print("***************")
        #print(comb)

        #print(np.sqrt(np.absolute(focal_num/(focal_den + 1e-18))), " focal")

    focal_predicted = float(np.average(focal_array))
    return focal_predicted


##############################################################
def calibration_focalpoint_lstq_failure(num_points, hv, av, hu, au, h, t1, t2, focal_predicted = None, upper_bound = np.inf):
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
    
    hv = np.float128(hv)
    av = np.float128(av)
    hu = np.float128(hu)
    au = np.float128(au)
    h = np.float128(h)
    t1 = np.float128(t1)
    t2 = np.float128(t2)
    
    C = np.zeros([2*num_points,3 + num_points], dtype = float)
    
    for col in range(0, 2*num_points):
        if col % 2 == 0:
            C[col][1] = -1
            C[col][2] = hv[int(col/2)]
            C[col][3 + int(col/2)] = hv[int(col/2)] - av[int(col/2)]
            C[col + 1][0] = 1
            C[col + 1][2] = -hu[int(col/2)]
            C[col + 1][3 + int(col/2)] = au[int(col/2)] - hu[int(col/2)]
    
    U, S, Vh = np.linalg.svd(C)
    L = (Vh[-1, :])

    if focal_predicted is None:
        focal_predicted = compute_focal_failure(au ,av ,L , t1, t2, upper_bound)
    
    if focal_predicted is None:
        return None, None, None, None, None, None
    h = float(h)
    t1 = float(t1)
    t2 = float(t2)
    
    L0 = float(L[0] - t1*L[2])
    L1 = float(L[1] - t2*L[2])  
    L2 = float(L[2]*focal_predicted)
    
    normal = np.array([L0, L1, L2])
    c_predicted = np.array([[focal_predicted, 0, t1], [0, focal_predicted, t2], [0, 0, 1]])
    
    calcz = np.array(L[3:2*num_points])*h*focal_predicted/(np.linalg.norm(normal) + 1e-18)

    normal = normal/(np.linalg.norm(normal) + 1e-18)
    calcz = np.absolute(calcz)
    
    return normal, calcz, focal_predicted, c_predicted, L, C

def compute_focal_failure(au, av, L, t1, t2, upper_bound = np.inf):
    
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
        
    focal_array = []
    indices = np.array(list(range(len(au))))
    np.random.shuffle(indices)
    comb_array = util.random_combination(indices.tolist(), 2, upper_bound)
    
    for comb in comb_array:     
        focal_num = -(L[0] - L[2]*t1)*(L[3 + comb[0]]*(au[comb[0]] - t1) - L[3 + comb[1]]*(au[comb[1]] - t1)) - (L[1] - L[2]*t2)*(L[3 + comb[0]]*(av[comb[0]] - t2) - L[3 + comb[1]]*(av[comb[1]] - t2))
        focal_den = (L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))

        if focal_num/(focal_den + 1e-18) < 0:
            continue
        focal_array.append(np.sqrt(focal_num/(focal_den + 1e-18)))

    if len(focal_array) == 0:
        return None
    
    focal_predicted = float(np.average(focal_array))
    return focal_predicted

####################
####################

def calibration_focalpoint_lstq_failure_dual(num_points, hv, av, hu, au, h, t1, t2, focal_predicted = None, focal_predicted1 = None, upper_bound = np.inf):
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
        U, S, Vh = np.linalg.svd(C)
        L = (Vh[-1, :])
    except:
        return None, None, None, None, None, None, None
    if focal_predicted is None or focal_predicted1 is None:
        #focal_predicted, focal_predicted1 = compute_focal_failure_dual(au ,av ,L , t1, t2, upper_bound)
        focal_predicted, focal_predicted1 = compute_focal_failure_dual_lstq(au, av, L, t1, t2, upper_bound = np.inf)
    
    if focal_predicted is None:
        return None, None, None, None, None, None, None
    h = float(h)
    t1 = float(t1)
    t2 = float(t2)
    
    L0 = float(L[0] - t1*L[2])/focal_predicted
    L1 = float(L[1] - t2*L[2])/focal_predicted1  
    L2 = float(L[2])
    
    normal = np.array([L0, L1, L2])
    c_predicted = np.array([[focal_predicted, 0, t1], [0, focal_predicted1, t2], [0, 0, 1]])
    
    calcz = np.array(L[3:2*num_points])*h/(np.linalg.norm(normal) + 1e-18)

    normal = normal/(np.linalg.norm(normal) + 1e-18)
    calcz = np.absolute(calcz)
    
    return normal, calcz, focal_predicted, focal_predicted1, c_predicted, L, C

def compute_focal_failure_dual(au, av, L, t1, t2, upper_bound = np.inf):
    
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
        
    focal_array1 = []
    focal_array2 = []

    comb_array = util.random_combination(list(range(len(au))), 2, upper_bound)
    comb_array1 = util.random_combination(list(range(len(comb_array))), 2, upper_bound)
    
    for c in comb_array1:   

        comb = comb_array[c[0]]
        comb1 = comb_array[c[1]]

        eq1_a1 = -(L[0] - L[2]*t1)*(L[3 + comb[0]]*(au[comb[0]] - t1) - L[3 + comb[1]]*(au[comb[1]] - t1)) 
        eq1_a2 = -(L[1] - L[2]*t2)*(L[3 + comb[0]]*(av[comb[0]] - t2) - L[3 + comb[1]]*(av[comb[1]] - t2))
        eq1_b = (L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))

        eq2_a1 = -(L[0] - L[2]*t1)*(L[3 + comb1[0]]*(au[comb1[0]] - t1) - L[3 + comb1[1]]*(au[comb1[1]] - t1)) 
        eq2_a2 = -(L[1] - L[2]*t2)*(L[3 + comb1[0]]*(av[comb1[0]] - t2) - L[3 + comb1[1]]*(av[comb1[1]] - t2))
        eq2_b = (L[2]*(L[3 + comb1[0]] - L[3 + comb1[1]]))
        
        A = np.array([[eq1_a1, eq1_a2], [eq2_a1, eq2_a2]])
        b = np.array([eq1_b, eq2_b])
        soln = np.linalg.lstsq(A, b)

        f1_squared = 1/(soln[0][0] + 1e-18)
        f2_squared = 1/(soln[0][1] + 1e-18)
    
        if f1_squared < 0:
            continue
        f1 = np.sqrt(f1_squared)
        focal_array1.append(f1)
        if f2_squared < 0:
            continue
        f2 = np.sqrt(f2_squared)
        focal_array2.append(f2)
        
        print("***************")
        print(comb)
        print(comb1)
        print(f1,f2, " f1 f2")
        
    if len(focal_array1) == 0 or len(focal_array2) == 0:
        return None, None
    
    focal_predicted = float(np.average(focal_array1))
    focal_predicted1 = float(np.average(focal_array2))
    return focal_predicted, focal_predicted1

def compute_focal_failure_dual_lstq(au, av, L, t1, t2, upper_bound = np.inf):
    
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
        
    focal_array1 = []
    focal_array2 = []

    comb_array = util.random_combination(list(range(len(au))), 2, upper_bound)
    
    A = []
    b = []
    for comb in comb_array:   

        eq1_a1 = -(L[0] - L[2]*t1)*(L[3 + comb[0]]*(au[comb[0]] - t1) - L[3 + comb[1]]*(au[comb[1]] - t1)) 
        eq1_a2 = -(L[1] - L[2]*t2)*(L[3 + comb[0]]*(av[comb[0]] - t2) - L[3 + comb[1]]*(av[comb[1]] - t2))
        eq1_b = (L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))
        
        #A = np.array([[eq1_a1, eq1_a2], [eq2_a1, eq2_a2]])
        #b = np.array([eq1_b, eq2_b])
        A.append([eq1_a1, eq1_a2])
        b.append(eq1_b)

    soln = np.linalg.lstsq(np.array(A), np.array(b))

    f1_squared = 1/(soln[0][0] + 1e-18)
    f2_squared = 1/(soln[0][1] + 1e-18)

    if f1_squared < 0:
        return None, None
    f1 = np.sqrt(f1_squared)
    focal_array1.append(f1)
    if f2_squared < 0:
        return None, None
    
    f2 = np.sqrt(f2_squared)
    focal_array2.append(f2)

    return f1, f2