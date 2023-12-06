import numpy as np
import matplotlib.pyplot as plt
import util


def plot_plane(ppl_ankle_u, ppl_ankle_v, ppl_head_u, ppl_head_v, save_dir, img, from_pickle, scale, line_amount, name, threshold_euc, threshold_cos, h):
    '''
    Plots the ground plane and line from ankle to head onto an image and also plots the error line from head detection to ransac predicted head, red means it exceeds threshold. 

    Parameters: ppl_ankle_u: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                ppl_ankle_v: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                ppl_head_u: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                ppl_head_v: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                save_dir: string
                    directory to save plots
                img: np.array
                    frame from the sequuence that is used for plotting
                from_pickle: dictionary
                    dictionary that contains the calibration (must contain camera matrix, normal, and plane center (called ankle))
                scale: int
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)
                line_amount: int
                    amount of lines in the ground plane to plot
                name: string
                    subdirectory in save_dir to save plots
                threshold_euc: float
                    Euclidean threshold for inliers
                threshold_cos: float
                    Cosine threshold for inliers
                h: float
                    assumed height of the people    

    Returns:    None
    '''

    img_width = img.shape[1]
    img_height = img.shape[0]

    fig, ax1 = plt.subplots(1, 1)

    ax1.imshow(img)

    cam_matrix = from_pickle['cam_matrix']
    normal = from_pickle['normal']
    ankleWorld = from_pickle['ankleWorld']
    
    # print(normal, " NORMAL")
    # print(cam_matrix, " CAM_MATRIX")
    # print(ankleWorld, " ANKLEWORLD")

    cam_inv = np.linalg.inv(cam_matrix)
    
    display_homography_horiz_bottom(cam_matrix ,cam_inv ,scale , img_width, img_height, ankleWorld, normal, line_amount, ax_array = [ax1])
    ankle_distance_array = []

    for i in range(0, len(ppl_ankle_u)):
        #points that are not detected by openpose are assignned -1
        if ppl_ankle_u[i] < 0 or ppl_ankle_v[i] < 0 or ppl_head_u[i] < 0 or ppl_head_v[i] < 0:
            continue
            
        person_world = np.squeeze(util.plane_ray_intersection_np([ppl_ankle_u[i]], [ppl_ankle_v[i]], cam_inv, normal, ankleWorld))   
        ankle_distance_array.append(np.linalg.norm(person_world))
        ankle_ppl_2d = util.perspective_transformation(cam_matrix, person_world)
        head_ppl_2d = util.perspective_transformation(cam_matrix, np.array(person_world) + np.squeeze(normal)*h)

        head_vect_pred = np.array([head_ppl_2d[0], head_ppl_2d[1]]) - np.array([ankle_ppl_2d[0], ankle_ppl_2d[1]])
        head_vect_ground = np.array([ppl_head_u[i], ppl_head_v[i]]) - np.array([ppl_ankle_u[i], ppl_ankle_v[i]])

        head_vect_ground_norm = np.linalg.norm(head_vect_ground)
        
        error_cos = 1.0 - util.matrix_cosine(np.expand_dims(head_vect_pred, axis = 0), np.expand_dims(head_vect_ground, axis = 0))
        error_norm = np.linalg.norm(np.array([head_ppl_2d[0], head_ppl_2d[1]]) - np.array([ppl_head_u[i], ppl_head_v[i]]))/head_vect_ground_norm
        
        if error_cos < threshold_cos and error_norm < threshold_euc:
            ax1.scatter(x=head_ppl_2d[0], y=head_ppl_2d[1], c='green', s=10)
            ax1.scatter(x=ankle_ppl_2d[0], y=ankle_ppl_2d[1], c='green', s=30)
            ax1.plot([head_ppl_2d[0], ppl_head_u[i]],[head_ppl_2d[1], ppl_head_v[i]], c = 'green')
            ax1.plot([head_ppl_2d[0], ankle_ppl_2d[0]],[head_ppl_2d[1], ankle_ppl_2d[1]], c = 'green')
        
        else:
            ax1.scatter(x=ankle_ppl_2d[0], y=ankle_ppl_2d[1], c='red', s=30)
            ax1.scatter(x=head_ppl_2d[0], y=head_ppl_2d[1], c='red', s=10)
            ax1.plot([head_ppl_2d[0], ppl_head_u[i]],[head_ppl_2d[1], ppl_head_v[i]], c = 'red')
            ax1.plot([head_ppl_2d[0], ankle_ppl_2d[0]],[head_ppl_2d[1], ankle_ppl_2d[1]], c = 'red') 
        
            
    fig.savefig(save_dir + '/' + 'imgplot_' + str(name) + '.png')
    plt.close('all')
# ground plane
def display_homography_horiz_bottom(cam_matrix, cam_inv, scale, img_width, img_height, ankleWorld, normal, line_amount, ax_array = []):
    '''
    creates the groundplane as a mesh on an axes object

    Parameters: cam_matrix: (3,3) np.array
                    The intrinsic camera matrix.
                cam_inv: (3,3) np.array
                    The inverse intrinsic camera matrix.
                scale: Float
                    The scale of the plotted grid.
                img_width: int
                    Pixel width of the image.
                img_height: Int
                    Pixel height of the image.
                ankleWorld: (3,) np.array
                    3d point of an ankle used to initalize the ground plane.
                normal: (3,) np.array
                    Normal vector of the groundplane.
                line_amount: Int
                    The amount of lines to graph. Creates line_amount of horizontal lines and 2*line_amount in the x direction
                ax_array: Array of matplotlib axes objects
                    axes for the grid to be ploted on
    Returns:    ax_array: list of matplotlib ax object
    '''

    plane_world = np.squeeze(util.plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, ankleWorld))
    
    plane_world_2d = cam_matrix @ plane_world
    
    plane_world1 = np.squeeze(util.plane_ray_intersection_np([img_width/2.0 + 1], [img_height], cam_inv, normal, ankleWorld))
                
    v_basis = np.array(plane_world1) - np.array(plane_world)
    v_basis = v_basis/np.linalg.norm(v_basis)
    
    u_basis = np.cross(normal, v_basis)
    u_basis = u_basis/np.linalg.norm(u_basis)
                
    x_basis_3d = [plane_world[0] + scale*v_basis[0], plane_world[1] + scale*v_basis[1], plane_world[2] + scale*v_basis[2]]
    y_basis_3d = [plane_world[0] + scale*u_basis[0], plane_world[1] + scale*u_basis[1], plane_world[2] + scale*u_basis[2]]
    z_basis_3d = [plane_world[0] + scale*normal[0], plane_world[1] + scale*normal[1], plane_world[2] + scale*normal[2]]

    x_basis = util.perspective_transformation(cam_matrix, x_basis_3d)
    y_basis = util.perspective_transformation(cam_matrix, y_basis_3d)
    z_basis = util.perspective_transformation(cam_matrix, z_basis_3d)
    
    ax_array[0].scatter(x = plane_world_2d[0]/plane_world_2d[2], y = plane_world_2d[1]/plane_world_2d[2], c = 'black', s = 40)
    ax_array[0].scatter(x = img_width/2.0,y = img_height, c='red',  s=20)
    ax_array[0].scatter(x = x_basis[0],y = x_basis[1], c='orange',  s=20)
    ax_array[0].scatter(x = y_basis[0],y = y_basis[1], c='lime',  s=20)
    ax_array[0].scatter(x = z_basis[0],y = z_basis[1], c='red',  s=20)

    ax_array[0].plot([img_width/2.0, x_basis[0]],[img_height, x_basis[1]], '-', c='orange',  linewidth=1.0)
    ax_array[0].plot([img_width/2.0, y_basis[0]],[img_height, y_basis[1]], '-', c='lime',  linewidth=1.0)
    ax_array[0].plot([img_width/2.0, z_basis[0]],[img_height, z_basis[1]], '-', c='red',  linewidth=1.0)

    rot_matrix = util.basis_change_rotation_matrix(cam_matrix, cam_inv, ankleWorld, normal, img_width, img_height)
    
    color = 'cyan'
    color1 = 'cyan'
    for i in range(-line_amount,line_amount):
            p00, p00_3d = util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(0)*scale], plane_world, normal, img_width, img_height)
            p01, p01_3d = util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(line_amount)*scale], plane_world, normal, img_width, img_height)
            p10, p10_3d = util.project_point_horiz_bottom(cam_matrix, cam_inv,[(-line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
            p11, p11_3d = util.project_point_horiz_bottom(cam_matrix, cam_inv,[(line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
            
            if p00_3d[2] < 0:
                p00_3d = util.plane_line_intersection(p00_3d, p01_3d, [0,0,1], [0,0,0.1])
                p00 = util.perspective_transformation(cam_matrix, p00_3d)
            if p01_3d[2] < 0:
                p01_3d = util.plane_line_intersection(p01_3d, p00_3d, [0,0,1], [0,0,0.1])
                p01 = util.perspective_transformation(cam_matrix, p01_3d)
            if p10_3d[2] < 0:
                p10_3d = util.plane_line_intersection(p10_3d, p11_3d, [0,0,1], [0,0,0.1])
                p10 = util.perspective_transformation(cam_matrix, p10_3d)
            if p11_3d[2] < 0:
                p11_3d = util.plane_line_intersection(p11_3d, p10_3d, [0,0,1], [0,0,0.1])
                p11 = util.perspective_transformation(cam_matrix, p11_3d)
            
            p00_plane = rot_matrix @ p00_3d
            p01_plane = rot_matrix @ p01_3d
            p10_plane = rot_matrix @ p10_3d 
            p11_plane = rot_matrix @ p11_3d
            
            alpha = 1
            alpha1 = 1
            linewidth = 1
            
            thresh1, thresh2, thresh3, thresh4 = 2,4,6,8
            if np.linalg.norm(p00_3d) < thresh1 and np.linalg.norm(p01_3d) < thresh1:
                alpha = 1.0
            elif np.linalg.norm(p00_3d) < thresh2 and np.linalg.norm(p01_3d) < thresh2:
                alpha = 0.8
            elif np.linalg.norm(p00_3d) < thresh3 and np.linalg.norm(p01_3d) < thresh3:
                alpha = 0.6
            elif np.linalg.norm(p00_3d) < thresh4 and np.linalg.norm(p01_3d) < thresh4:
                alpha = 0.4
            else:
                alpha = 0.2
                
            if np.linalg.norm(p10_3d) < thresh1 and np.linalg.norm(p11_3d) < thresh1:
                alpha1 = 1.0
            elif np.linalg.norm(p10_3d) < thresh2 and np.linalg.norm(p11_3d) < thresh2:
                alpha1 = 0.8
            elif np.linalg.norm(p10_3d) < thresh3 and np.linalg.norm(p11_3d) < thresh3:
                alpha1 = 0.6
            elif np.linalg.norm(p10_3d) < thresh4 and np.linalg.norm(p11_3d) < thresh4:
                alpha1 = 0.4
            else:
                alpha1 = 0.2
            
            x = [p00[0],p01[0]]
            y = [p00[1],p01[1]] 
            ax_array[0].plot(x,y, '-', c=color,  linewidth=linewidth, alpha = alpha, zorder=1) #this part creates the plane

            x = [p11[0],p10[0]]
            y = [p11[1],p10[1]]
            if p11_plane[1] > 0 or p10_plane[1] > 0:
                ax_array[0].plot(x,y, '-', c=color1,  linewidth=linewidth, alpha = alpha1, zorder=1) #this part creates the plane
                
    ax_array[0].set_xlim([-50, img_width + 50])
    ax_array[0].set_ylim([img_height + 50, -50])

    return ax_array

def display_2d_grid(ppl_ankle_u, ppl_ankle_v, ppl_head_u, ppl_head_v, save_dir, img, from_pickle, scale, line_amount, name, threshold_euc, threshold_cos, h, color_plot = 'green'):
    '''
    Plots the ground plane and line from ankle to head onto an image and also plots the error line from head detection to ransac predicted head, red means it exceeds threshold. 

    Parameters: ppl_ankle_u: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                ppl_ankle_v: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                ppl_head_u: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                ppl_head_v: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                save_dir: string
                    directory to save plots
                img: np.array
                    frame from the sequuence that is used for plotting
                from_pickle: dictionary
                    dictionary that contains the calibration (must contain camera matrix, normal, and plane center (called ankle))
                scale: float
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)
                line_amount: int
                    amount of lines in the ground plane to plot
                name: string
                    subdirectory in save_dir to save plots
                threshold_euc: float
                    Euclidean threshold for inliers
                threshold_cos: float
                    Cosine threshold for inliers
                h: float
                    assumed height of the people    

    Returns:    None
    '''
    
    img_width = img.shape[1]
    img_height = img.shape[0]
            
    ankleWorld = from_pickle['ankleWorld']
    cam_matrix = from_pickle['cam_matrix']
    normal = from_pickle['normal']

    cam_inv = np.linalg.inv(cam_matrix)
    
    rot_matrix = util.basis_change_rotation_matrix(cam_matrix, cam_inv, ankleWorld, normal, img_width, img_height)
    
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Ground plane overlay (birds eye view)')   
    
    plane_world = np.squeeze(util.plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, ankleWorld)) 

    #GRAPHING THE IMAGE VIEW
    
    alpha = 1
    color = 'cyan'
    linewidth = 1
    ax1.scatter(x=0.0, y=0.0, c='black', s=30)

    p_center_proj, p_center_3d = util.project_point_horiz_bottom(cam_matrix, cam_inv,[0,0], plane_world, normal, img_width, img_height)
    p_center = rot_matrix @ p_center_3d  

    for i in range(-line_amount,line_amount):
            p00, p00_3d = util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(-line_amount)*scale], plane_world, normal, img_width, img_height)
            p01, p01_3d = util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(line_amount)*scale], plane_world, normal, img_width, img_height)
            p10, p10_3d = util.project_point_horiz_bottom(cam_matrix, cam_inv,[(-line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
            p11, p11_3d = util.project_point_horiz_bottom(cam_matrix, cam_inv,[(line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
                
            p00 = (rot_matrix @ p00_3d) - p_center 
            p01 = (rot_matrix @ p01_3d) - p_center  
            p10 = (rot_matrix @ p10_3d) - p_center   
            p11 = (rot_matrix @ p11_3d) - p_center  
            
            x = [p00[0],p01[0]]
            y = [p00[1],p01[1]]        
            ax1.plot(x,y, '-', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane
            
            x = [p11[0],p10[0]]
            y = [p11[1],p10[1]]
            
            ax1.plot(x,y, '-', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane
    
    ankle_x = []
    ankle_y = []
    for i in range(0, len(ppl_ankle_u)):
        #points that are not detected by openpose are assignned -1
        if ppl_ankle_u[i] < 0 or ppl_ankle_v[i] < 0 or ppl_head_u[i] < 0 or ppl_head_v[i] < 0:
            continue
        
        ankle_3d = np.squeeze(util.plane_ray_intersection_np([ppl_ankle_u[i]], [ppl_ankle_v[i]], cam_inv, normal, ankleWorld))
        person_plane = (rot_matrix @ ankle_3d) 
        #ax1.scatter(x=person_plane[0], y=person_plane[1], c='green', s=30)
        
        ankle_x.append(person_plane[0])
        ankle_y.append(person_plane[1])

        
        #############
        ankle_ppl_2d = util.perspective_transformation(cam_matrix, ankle_3d)
        head_ppl_2d = util.perspective_transformation(cam_matrix, np.array(ankle_3d) + np.squeeze(normal)*h)

        head_vect_pred = np.array([head_ppl_2d[0], head_ppl_2d[1]]) - np.array([ankle_ppl_2d[0], ankle_ppl_2d[1]])
        head_vect_ground = np.array([ppl_head_u[i], ppl_head_v[i]]) - np.array([ppl_ankle_u[i], ppl_ankle_v[i]])

        head_vect_ground_norm = np.linalg.norm(head_vect_ground)
        
        error_cos = 1.0 - util.matrix_cosine(np.expand_dims(head_vect_pred, axis = 0), np.expand_dims(head_vect_ground, axis = 0))
        error_norm = np.linalg.norm(np.array([head_ppl_2d[0], head_ppl_2d[1]]) - np.array([ppl_head_u[i], ppl_head_v[i]]))/head_vect_ground_norm
        
        if error_cos < threshold_cos and error_norm < threshold_euc:
            ax1.scatter(x=person_plane[0], y=person_plane[1], c=color_plot, s=30)
        
        else:
            ax1.scatter(x=person_plane[0], y=person_plane[1], c='red', s=30)
        
    plt.annotate("Camera position", (0, 0))
    
    img_bl_world = (rot_matrix @ cam_inv @ np.array([0,img_height/2.0, 1]))
    img_br_world = (rot_matrix @ cam_inv @ np.array([img_width,img_height/2.0, 1]))
    # print(np.degrees(util.angle_between(img_bl_world, img_br_world)), " ANGlE !!!!!!!!!!!!!!!!!!!!!")

    img_bl_world = (img_bl_world/np.linalg.norm(img_bl_world))*np.sqrt(2*np.square(scale*line_amount))
    img_br_world = (img_br_world/np.linalg.norm(img_br_world))*np.sqrt(2*np.square(scale*line_amount))
    
    x_lim = max([np.absolute(min(ankle_x + [0])), np.absolute(max(ankle_x + [0]))])

    ax1.set_xlim([-x_lim - 1, x_lim + 1])
    ax1.set_ylim([min(ankle_y + [0]) - 1, max(ankle_y + [0]) + 1])
    ax1.plot([0, img_bl_world[0]], [0, img_bl_world[1]], c='red')
    ax1.plot([0, img_br_world[0]], [0, img_br_world[1]], c='red')
    ax1.set_aspect('equal', adjustable='box')
    
    fig.savefig(save_dir + '/' + 'topview_' + str(name) + '.png')
    plt.close('all')

    return ankle_x, ankle_y