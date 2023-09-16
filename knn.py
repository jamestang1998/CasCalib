from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import tslearn
from tslearn import clustering
import util

'''
d1 = np.array(
  [[187.0, 489.0, 1501.0, 575.0],
  [1810.0, 1967.0, 1917.0, 2052.0],
  [1360.0, 2187.0, 1467.0, 2275.0],
  [1256.0, 2188.0, 1361.0, 2276.0],
  [506.0, 2197.0, 615.0, 2284.0],
  [199.0, 2288.0, 306.0, 2372.0]]
)

d2 = np.array(
  [[200.0, 490.0, 1491.0, 588.0],
  [1813.0, 1966.0, 1919.0, 2053.0],
  [1370.0, 2188.0, 1473.0, 2276.0],
  [1265.0, 2189.0, 1365.0, 2275.0],
  [520.0, 2200.0, 629.0, 2288.0],
  [222.0, 2291.0, 327.0, 2376.0]]
)
'''
def multi_person_frame_match(frame_dict):
    
    frame_array = []
    vel_dict = {}

    frame_key = list(frame_dict.keys())
    for k in range(len(frame_key) - 1):
        vel_dict
      
    return

def cam_2d_frame_match(datastore, image_index, conf):

    frame_dict = {}
    
    for ppl in image_index:

        left_ankle = datastore.getitem(ppl)["left_ankle"]
        right_ankle = datastore.getitem(ppl)["right_ankle"]

        ankle_left_conf.append(datastore.getitem(ppl)["left_ankle"][2])
        ankle_right_conf.append(datastore.getitem(ppl)["right_ankle"][2])

        ankle_x, ankle_y = determine_foot(right_ankle,left_ankle, hgt_threshold=4.0, wide_threshold=4.0)
        ankle_x = (datastore.getitem(ppl)["left_hip"][0] + datastore.getitem(ppl)["right_hip"][0])/2.0
            
        head_x = (datastore.getitem(ppl)["middle"][0])
        head_y = (datastore.getitem(ppl)["middle"][1])

        frame_name = int(datastore.getitem(ppl)["left_ankle"][3].split('/')[-1].split('.')[0]) 

        if frame_name not in frame_dict.keys():

          frame_dict[frame_name] = [[ankle_x, ankle_y, head_x, head_y]]
        else:
          frame_dict.append([ankle_x, ankle_y, head_x, head_y])

    pose_dict = {k:v for k,v in pose_dict.items() if v}
    frame_dict = {k:v for k,v in frame_dict.items() if v}

    frame_names = list(pose_dict.keys())

    return frame_dict, pose_dict, all_points, np.array(ppl_ankle_u), np.array(ppl_ankle_v), np.array(ppl_head_u), np.array(ppl_head_v)

def multi_camera_match(dict1, dict2, sync1, sync2):

    index1_array, index2_array, d2_sorted_array = []

    for i in range(len(sync1)):
          
      d1 = dict1[sync1[i]]
      d2 = dict2[sync2[i]]
      classes = np.arange(len(d1))
      
      knn = KNeighborsClassifier(n_neighbors=1)
      knn.fit(d1, y=classes)
      matches = knn.predict(d2)

      index_dict = {}
      for i in range(len(d2)):
          if matches[i] not in index_dict.keys():
              index_dict[matches[i]] = i
          else:
              if np.linalg.norm(d1[matches[i]] - d2[i]) < np.linalg.norm(d1[matches[i]] - d2[index_dict[matches[i]]]):
                  index_dict[matches[i]] = i
  
      index1 = list(index_dict.keys())  
      index2 = list(index_dict.values())

      d2_sorted = d2[np.argsort(index1)] 

      index1_array.append(index1)
      index2_array.append(index2)
      d2_sorted_array.append(d2_sorted)
    return index1_array, index2_array, d2_sorted_array

def knn(d1,d2):
    classes = np.arange(len(d1))
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(d1, y=classes)
    matches = knn.predict(d2)

    index_dict = {}
    for i in range(len(d2)):
        if matches[i] not in index_dict.keys():
            index_dict[matches[i]] = i
        else:
            if np.linalg.norm(d1[matches[i]] - d2[i]) < np.linalg.norm(d1[matches[i]] - d2[index_dict[matches[i]]]):
                index_dict[matches[i]] = i
 
    index1 = list(index_dict.keys())  
    index2 = list(index_dict.values())

    d2_sorted = d2[np.argsort(index1)]         
    return index1, index2, d2_sorted
'''
index1, index2, d2_sorted = knn(d1,d2)

print(d1[index1])
print(d2[index2])
print(index1)
print(index2)
'''
'''
d1 = np.array(
  [[187.0, 489.0, 1501.0, 575.0],
  [1810.0, 1967.0, 1917.0, 2052.0],
  [1360.0, 2187.0, 1467.0, 2275.0],
  [1256.0, 2188.0, 1361.0, 2276.0],
  [506.0, 2197.0, 615.0, 2284.0],
  [199.0, 2288.0, 306.0, 2372.0]]
)

d2 = np.array(
  [[200.0, 490.0, 1491.0, 588.0],
  [1813.0, 1966.0, 1919.0, 2053.0],
  [1370.0, 2188.0, 1473.0, 2276.0],
  [1265.0, 2189.0, 1365.0, 2275.0],
  [520.0, 2200.0, 629.0, 2288.0],
  [222.0, 2291.0, 327.0, 2376.0]]
)

d3 = np.array([[ 524.0, 2182.0,  632.0, 2294.0],
  [1368.0, 2173.0, 1471.0, 2287.0],
  [ 182.0,  474.0, 1473.0,  605.0],
  [1797.0, 1975.0, 1930.0, 2055.0],
  [1281.0, 2202.0, 1356.0, 2263.0],
  [ 227.0, 2295.0,  339.0, 2394.0]]
)

classes = np.arange(len(d1))

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(d1, y=classes)
matches = knn.predict(d3)
d3_sorted = d3[np.argsort(matches)]
# returns:
#array([0, 1, 2, 3, 4, 5])
print(d3_sorted)
'''