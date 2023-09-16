import numpy as np
import util
import re
import matplotlib.pyplot as plt

class vitpose_hrnet():
    def __init__(self, data_json, scale_x = 1.0, scale_y = 1.0, bound_lower = 0, bound = None):
        '''
        constructor for the dcpose_dataloader class, a class takes a json file of dcpose detections and creates an indexable object to easily access poses and keypoints.
        Parameters: data: json object
                        json file of open pose detections. If data is None, then initalize self.data as an empty array
        Returns: None
        '''
        self.average_height = None

        if data_json is None:
            self.data = []
            return
        #Thorax = neck
        keypoint_array = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        
        data_obj = {}
        for img in range(0, len(data_json)):
            pose = {}
            #print(img, data_json[img]["image_id"], " this is IMAGE ID")
            if bound is not None:
                if int(data_json[img]["image_id"].split('/')[-1].split('.')[0]) > bound:
                    continue
                if int(data_json[img]["image_id"].split('/')[-1].split('.')[0]) < bound_lower:
                    continue
            #print(img, data_json[img]["image_id"], " this is IMAGE ID")
            #if img > 100 and img < 200  :
            #    continue
            for i in range(len(keypoint_array)):
                #print(data_json[img], " data json ")
                keypoint_u = scale_x*data_json[img]['keypoints'][i][0]
                keypoint_v = scale_y*data_json[img]['keypoints'][i][1]
                confidence = data_json[img]['keypoints'][i][2]
                    
                pose[keypoint_array[i]] = [keypoint_u, keypoint_v, confidence]
                
            frame_name = int(data_json[img]["image_id"].split('/')[-1].split('.')[0])
            
            pose['id'] = data_json[img]["bbox_id"]

            if frame_name in data_obj:
                data_obj[frame_name].append(pose)
            else:
                data_obj[frame_name] = [pose]
        
        self.data = data_obj

    def __len__(self):
        '''
        Returns the length of self.data
        
        Parameters: None
        Returns: output
                    Length of self.data
        '''
        return len(self.data)
    
    def getitem(self, idx):  
        '''
        Gets each poses keypoint detections
        
        Parameters: idx: int
                        index of pose
        Returns: output: python dictionary
                        dictionary of keypoints for each pose
        '''
        key = list(self.data.keys())[idx]
        return self.data[key]
    
    def getData(self):  
        return self.data
    
class vitpose_frame_dataloader():
    def __init__(self, data_json):
        '''
        constructor for the dcpose_dataloader class, a class takes a json file of dcpose detections and creates an indexable object to easily access poses and keypoints.
        Parameters: data: json object
                        json file of open pose detections. If data is None, then initalize self.data as an empty array
        Returns: None
        '''
        self.average_height = None

        if data_json is None:
            self.data = []
            return
        #Thorax = neck
        keypoint_array = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        
        data_obj = {}
        for img in range(0, len(data_json)):
            pose = {}
            #if len(list(data_obj.keys())) == 1000:
            #    break
            for p in range(0, len(data_json[img])):
                
                for i in range(len(keypoint_array)):
                    keypoint_u = data_json[img][p]['keypoints']['__ndarray__'][i][0]
                    keypoint_v = data_json[img][p]['keypoints']['__ndarray__'][i][1]
                    confidence = data_json[img][p]['keypoints']['__ndarray__'][i][2]
                        
                    pose[keypoint_array[i]] = [keypoint_u, keypoint_v, confidence]
                frame_name = data_json[img][p]["image_id"]

                if frame_name in data_obj:
                    data_obj[frame_name].append(pose)
                else:
                    data_obj[frame_name] = [pose]

        self.data = data_obj

    def __len__(self):
        '''
        Returns the length of self.data
        
        Parameters: None
        Returns: output
                    Length of self.data
        '''
        return len(self.data)
    
    def getitem(self, idx):  
        '''
        Gets each poses keypoint detections
        
        Parameters: idx: int
                        index of pose
        Returns: output: python dictionary
                        dictionary of keypoints for each pose
        '''
        key = list(self.data.keys())[idx]
        return self.data[key]
    
    def getData(self):  
        return self.data
    
class coco_mmpose_dataloader():
    def __init__(self, data_json, scale_x = 1.0, scale_y = 1.0, bound_lower = 0, bound = None):
        '''
        constructor for the dcpose_dataloader class, a class takes a json file of dcpose detections and creates an indexable object to easily access poses and keypoints.
        Parameters: data: json object
                        json file of open pose detections. If data is None, then initalize self.data as an empty array
        Returns: None
        '''
        self.average_height = None

        if data_json is None:
            self.data = []
            return
        #Thorax = neck
        keypoint_array = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        
        data_obj = {}
        for img in range(0, len(data_json["Info"])):
            pose = {}
            #print(img, " THE IMG ")
            #print(data_json["Info"][img]["frame"], " image id ")
            if bound is not None:
                if int(data_json["Info"][img]["frame"]) > bound:
                    continue
                if int(data_json["Info"][img]["frame"]) < bound_lower:
                    continue
            
            cond_array = []
            #if len(list(data_obj.keys())) == 100:
            #    break
            '''
            joint_array =  [[0,1],  #Nose - Right Eye               #0
                    [0,2],  #Nose - Left Eye                #1
                    [1,2],  #Right Eye - Left Eye           #2
                    [1,3],  #Right Eye - Right Ear          #3
                    [2,4],  #Left Eye - Left Ear            #4
                    [3,5],  #Right Ear - Right Shoulder     #5
                    [4,6],  #Left Ear - Left Shoulder       #6
                    [5,6],  #Right Shoulder - Left Shoulder #7              
                    [6,8],  #Left Shoulder - Left Elbow     #8
                    [5,7],  #Right Shoulder - Right Elbow   #9
                    [7,9],  #Right Elbow - Right Wrist      #10
                    [8,10], #Left Elbow - Left Wrist        #11
                    [6,12], #Left Shoulder - Left Hip       #12
                    [5,11], #Right Shoulder - Right Hip     #13
                    [11,12],#Right Hip - Left Hip           #14
                    [12,14],#Left Hip - Left Knee           #15
                    [11,13],#Right Hip - Right Knee         #16
                    [14,16],#Left Knee - Left Ankle         #17
                    [13,15]]#Right Knee - Right Ankle       #18
            '''
            #fig, ax = plt.subplots()
            #if int(data_json["Info"][img]["frame"]) > 30:
            #    break
            #if int(data_json["Info"][img]["frame"]) != 154:
            #    continue

            for i in range(len(keypoint_array)):
                keypoint_u = scale_x*data_json["Info"][img]['keypoints'][i][0]
                keypoint_v = scale_y*data_json["Info"][img]['keypoints'][i][1]
                confidence = data_json["Info"][img]['keypoints'][i][2]
                    
                pose[keypoint_array[i]] = [keypoint_u, keypoint_v, confidence]
                
                #ax.scatter(keypoint_u, keypoint_v)
                #ax.annotate(keypoint_array[i] + ' ' + str(i), (keypoint_u, keypoint_v))
                
                if keypoint_array[i] == 'left_ankle' or keypoint_array[i] == 'right_ankle' or keypoint_array[i] == 'neck':
                    cond_array.append(confidence)
            '''
            for i in range(len(joint_array)):
                k1 = joint_array[i][0]
                k2 = joint_array[i][1]

                keypoint_u1 = data_json["Info"][img]['keypoints'][k1][0]
                keypoint_v1 = data_json["Info"][img]['keypoints'][k1][1]

                keypoint_u2 = data_json["Info"][img]['keypoints'][k2][0]
                keypoint_v2 = data_json["Info"][img]['keypoints'][k2][1]
                print(k1, k2, "K1 k2 !!")
                print(keypoint_u1, keypoint_v1, " HII")  
                print(keypoint_u2, keypoint_v2, " HIIiii")        
                ax.plot([keypoint_u1, keypoint_u2], [keypoint_v1, keypoint_v2])

            ax.axis('equal')
            plt.show()
            '''
            frame_name = data_json["Info"][img]["frame"]
            person_id = data_json["Info"][img]["track_id"]
            pose['id'] = int(person_id)
            pose["bbox"] = data_json["Info"][img]["bbox"]
            #print(frame_name, " THIS IS THE FRANE")
            if int(frame_name) in data_obj:
                data_obj[int(frame_name)].append(pose)
            else:
                data_obj[int(frame_name)] = [pose]

        self.data = data_obj

    def __len__(self):
        '''
        Returns the length of self.data
        
        Parameters: None
        Returns: output
                    Length of self.data
        '''
        return len(self.data)
    
    def getitem(self, idx):  
        '''
        Gets each poses keypoint detections
        
        Parameters: idx: int
                        index of pose
        Returns: output: python dictionary
                        dictionary of keypoints for each pose
        '''
        key = list(self.data.keys())[idx]
        return self.data[key]
    
    def getData(self):  
        return self.data
    
    def writeData(self, data):  
        self.data = data
    
class h36m_multi_gt_dataloader():
    def __init__(self, data_json_array, cond = 0.0):
        '''
        constructor for the dcpose_dataloader class, a class takes a json file of dcpose detections and creates an indexable object to easily access poses and keypoints.
        Parameters: data: json object
                        json file of open pose detections. If data is None, then initalize self.data as an empty array
        Returns: None
        '''
        ''' 
        0 = 'Hip'
        1 = 'RHip'
        2 = 'RKnee'
        3 = 'RFoot'
        6 = 'LHip'
        7 = 'LKnee'
        8 = 'LFoot'
        12 = 'Spine'
        13 = 'Thorax' 
        14 = 'Neck/Nose'
        15 = 'Head'
        17 = 'LShoulder'
        18 = 'LElbow'
        19 = 'LWrist'
        25 = 'RShoulder'
        26 = 'RElbow'
        27 = 'RWrist'
        '''
        keypoint_index = [
                    0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27 
        ]
        #keypoint_array = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 
        # 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
        #  'right_ankle', 'head', 'neck', 'hip','left_toe','right_toe','left_small_toe', "right_small_toe", "left_heel", "right_heel"]
        keypoint_array = ['hip', # 0
        'right_hip',             # 1
        'right_knee',            # 2
        'right_ankle',           # 3
        'left_hip',              # 4
        'left_knee',             # 5
        'left_ankle',            # 6
        'Spine',                 # 7
        'Thorax',                # 8
        'neck',                  # 9
        'head',                  # 10
        'left_shoulder',         # 11
        'left_elbow',            # 12
        'left_wrist',            # 13
        'right_shoulder',        # 14
        'right_elbow',           # 15
        'right_wrist']           # 16

        joint_array = [[0,1],  #hip - right hip
                       [0,4],  #hip - left hip
                       [1,2],  #right hip - right_knee
                       [4,5],  #left hip - left_knee
                       [2,3],  #right_knee - right_ankle
                       [5,6],  #left_knee - left_ankle
                       [0,7],  #hip - Spine
                       [7,8],  #Spine - Thorax
                       [8,9],  #Thorax - neck
                       [9,10], #neck - head
                       [8,14], #Thorax - right_shoulder
                       [8,11], #Thorax - left_shoulder
                       [14,15],#right_shoulder - right_elbow 
                       [11,12],#left_shoulder - left_elbow
                       [15,16],#right_elbow - right_wrist
                       [12,13]]#left_elbow - left_wrist
        self.average_height = None

        for dj in range(len(data_json_array)):

            data_json = data_json_array[dj]
            if data_json is None:
                self.data = []
                return
            
            #keypoint_array = ['left_ankle', 'right_ankle', 'neck']
            
            data_obj = {}
            for img in range(0, data_json.shape[0]):
            #for img in range(0, 100):
                pose = {}
                #if len(list(data_obj.keys())) == 1000:
                #    break
                for i in range(len(keypoint_array)):
                    keypoint_u = data_json[img, i, 0]
                    keypoint_v = data_json[img, i, 1]
                        
                    pose[keypoint_array[i]] = [keypoint_u, keypoint_v, 1.0]
                pose['idx'] = dj
                data_obj[img] = [pose]

            self.data = data_obj
            #print(self.data)
            #stop
    def __len__(self):
        '''
        Returns the length of self.data
        
        Parameters: None
        Returns: output
                    Length of self.data
        '''
        return len(self.data)
    
    def getitem(self, idx):  
        '''
        Gets each poses keypoint detections
        
        Parameters: idx: int
                        index of pose
        Returns: output: python dictionary
                        dictionary of keypoints for each pose
        '''
        key = list(self.data.keys())[idx]
        return self.data[key]
    
    def getData(self):  
        return self.data
    
class h36m_gt_dataloader():
    def __init__(self, data_json, cond = 0.0):
        '''
        constructor for the dcpose_dataloader class, a class takes a json file of dcpose detections and creates an indexable object to easily access poses and keypoints.
        Parameters: data: json object
                        json file of open pose detections. If data is None, then initalize self.data as an empty array
        Returns: None
        '''
        ''' 
        0 = 'Hip'
        1 = 'RHip'
        2 = 'RKnee'
        3 = 'RFoot'
        6 = 'LHip'
        7 = 'LKnee'
        8 = 'LFoot'
        12 = 'Spine'
        13 = 'Thorax' 
        14 = 'Neck/Nose'
        15 = 'Head'
        17 = 'LShoulder'
        18 = 'LElbow'
        19 = 'LWrist'
        25 = 'RShoulder'
        26 = 'RElbow'
        27 = 'RWrist'
        '''
        keypoint_index = [
                    0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27 
        ]
        #keypoint_array = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 
        # 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
        #  'right_ankle', 'head', 'neck', 'hip','left_toe','right_toe','left_small_toe', "right_small_toe", "left_heel", "right_heel"]
        keypoint_array = ['hip', # 0
        'right_hip',             # 1
        'right_knee',            # 2
        'right_ankle',           # 3
        'left_hip',              # 4
        'left_knee',             # 5
        'left_ankle',            # 6
        'Spine',                 # 7
        'Thorax',                # 8
        'neck',                  # 9
        'head',                  # 10
        'left_shoulder',         # 11
        'left_elbow',            # 12
        'left_wrist',            # 13
        'right_shoulder',        # 14
        'right_elbow',           # 15
        'right_wrist']           # 16

        joint_array = [[0,1],  #hip - right hip
                       [0,4],  #hip - left hip
                       [1,2],  #right hip - right_knee
                       [4,5],  #left hip - left_knee
                       [2,3],  #right_knee - right_ankle
                       [5,6],  #left_knee - left_ankle
                       [0,7],  #hip - Spine
                       [7,8],  #Spine - Thorax
                       [8,9],  #Thorax - neck
                       [9,10], #neck - head
                       [8,14], #Thorax - right_shoulder
                       [8,11], #Thorax - left_shoulder
                       [14,15],#right_shoulder - right_elbow 
                       [11,12],#left_shoulder - left_elbow
                       [15,16],#right_elbow - right_wrist
                       [12,13]]#left_elbow - left_wrist
        self.average_height = None

        if data_json is None:
            self.data = []
            return
        
        #keypoint_array = ['left_ankle', 'right_ankle', 'neck']
        
        data_obj = {}
        for img in range(0, data_json.shape[0]):
        #for img in range(0, 100):
            pose = {}
            #if len(list(data_obj.keys())) == 1000:
            #    break
            for i in range(len(keypoint_array)):
                keypoint_u = data_json[img, i, 0]
                keypoint_v = data_json[img, i, 1]
                    
                pose[keypoint_array[i]] = [keypoint_u, keypoint_v, 1.0]
            pose['id'] = 0
            data_obj[img] = [pose]

        self.data = data_obj
        #print(self.data)
        #stop
    def __len__(self):
        '''
        Returns the length of self.data
        
        Parameters: None
        Returns: output
                    Length of self.data
        '''
        return len(self.data)
    
    def getitem(self, idx):  
        '''
        Gets each poses keypoint detections
        
        Parameters: idx: int
                        index of pose
        Returns: output: python dictionary
                        dictionary of keypoints for each pose
        '''
        key = list(self.data.keys())[idx]
        return self.data[key]
    
    def getData(self):  
        return self.data
    
class alphapose_tracking_dataloader():
    def __init__(self, data_json, cond = 0.0):
        '''
        constructor for the dcpose_dataloader class, a class takes a json file of dcpose detections and creates an indexable object to easily access poses and keypoints.
        Parameters: data: json object
                        json file of open pose detections. If data is None, then initalize self.data as an empty array
        Returns: None
        '''
        self.average_height = None

        if data_json is None:
            self.data = []
            return
        #Thorax = neck
        keypoint_array = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'Thorax', 'hip','left_toe','right_toe','left_small_toe', "right_small_toe", "left_heel", "right_heel"]
        
        data_obj = {}
        for img in range(0, len(data_json)):
            pose = {}
            cond_array = []
            #if len(list(data_obj.keys())) == 1000:
            #    break
            for i in range(len(keypoint_array)):
                keypoint_u = data_json[img]['keypoints'][3*i]
                keypoint_v = data_json[img]['keypoints'][3*i + 1]
                confidence = data_json[img]['keypoints'][3*i + 2]
                    
                pose[keypoint_array[i]] = [keypoint_u, keypoint_v, confidence]

                if keypoint_array[i] == 'left_ankle' or keypoint_array[i] == 'right_ankle' or keypoint_array[i] == 'neck':
                    cond_array.append(confidence)

            ankle_x, ankle_y = util.determine_foot(pose['right_ankle'],pose['left_ankle'], hgt_threshold=8.0, wide_threshold=10.0)
            pose['box'] = data_json[img]['box']
            pose['idx'] = data_json[img]['idx']
            pose['ankle_center']  = [ankle_x, ankle_y]
            #print(data_json[img]['image_id'], " IMAGE IDDDD")
            #########
            frame_name = re.sub("[^0-9]", "", data_json[img]['image_id']).split('.')[0].lstrip("0")
            #print(frame_name)
            #frame_name = data_json[img]['image_id'].split('.')[0].lstrip("0")
            frame_number = 0
            if len(frame_name) > 0:
                frame_number = int(frame_name)#int(data_json[img]['image_id'].split('.')[0].lstrip("0"))

            if np.mean(cond_array) > cond:
                if frame_number not in data_obj:
                    data_obj[frame_number] = [pose]
                else:
                    data_obj[frame_number].append(pose)

        self.data = data_obj

    def __len__(self):
        '''
        Returns the length of self.data
        
        Parameters: None
        Returns: output
                    Length of self.data
        '''
        return len(self.data)
    
    def getitem(self, idx):  
        '''
        Gets each poses keypoint detections
        
        Parameters: idx: int
                        index of pose
        Returns: output: python dictionary
                        dictionary of keypoints for each pose
        '''
        key = list(self.data.keys())[idx]
        return self.data[key]
    
    def getData(self):  
        return self.data

class alphapose_dataloader():
    def __init__(self, data): 
        '''
        constructor for the openpose_dataloader class, a class takes a json file of openpose detections and creates an indexable object to easily access poses and keypoints.
        Parameters: data: json object
                        json file of open pose detections. If data is None, then initalize self.data as an empty array
        Returns: None
        '''
        if data is None:
            self.data = []
            '''
            self.data is the array of pose detections. Each pose detection is represented as a dictionary where the name of each keypoint is the key and the value is [x_coordinate, y_coordinate, confidence of detection]
            '''
            return
        #Middle is the bottom of the neck

        '''
        {0,  "Nose"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "RHip"},
        {9,  "RKnee"},
        {10, "RAnkle"},
        {11, "LHip"},
        {12, "LKnee"},
        {13, "LAnkle"},
        {14, "REye"},
        {15, "LEye"},
        {16, "REar"},
        {17, "LEar"},
        '''
        keypoint_array = ['nose','middle','left_shoulder','left_elbow','left_wrists','right_shoulder','right_wrist','right_elbow','left_hip','left_knee','left_ankle','right_hip','right_knee','right_ankle','left_eye','right_eye','left_ear','right_ear']
        
        data_array = []
        for f in data.keys():
            for ppl in range(len(data[f]["people"])):
                    
                pose = {}
                #print(f, ppl, " asdasdasd")
                name = 0
                for i in range(0, 51, 3):
                    keypoint_u = (data[f]["people"][ppl]["pose_keypoints_2d"][i])
                    keypoint_v = (data[f]["people"][ppl]["pose_keypoints_2d"][i + 1])
                    confidence = (data[f]["people"][ppl]["pose_keypoints_2d"][i + 2])
                    pose[keypoint_array[name]] = [keypoint_u, keypoint_v, confidence]

                    name = name + 1
                data_array.append(pose)
                
        self.data = data_array  
    def __len__(self):
        '''
        Returns the length of self.data
        
        Parameters: None
        Returns: output
                    Length of self.data
        '''
        return len(self.data)
    
    def getitem(self, idx):    
        '''
        Gets each poses keypoint detections
        
        Parameters: idx: int
                        index of pose
        Returns: output: python dictionary
                        dictionary of keypoints for each pose
        '''
        return self.data[idx]
    
    def remove(self, idx):    
        '''
        Removes an index from self.data
        
        Parameters: idx: int
                        index of pose
        Returns: output: None
        '''
        self.data = self.data.pop(idx)
        
    def new_data(self, data):   
        '''
        Replaces self.data with a new array of keypoint dictionaries
        
        Parameters: data: array of dictionaries
        Returns: output: None
        '''
        self.data = data

class openpose_dataloader():
    def __init__(self, data): 
        '''
        constructor for the openpose_dataloader class, a class takes a json file of openpose detections and creates an indexable object to easily access poses and keypoints.
        Parameters: data: json object
                        json file of open pose detections. If data is None, then initalize self.data as an empty array
        Returns: None
        '''
        if data is None:
            self.data = []
            '''
            self.data is the array of pose detections. Each pose detection is represented as a dictionary where the name of each keypoint is the key and the value is [x_coordinate, y_coordinate, confidence of detection]
            '''
            return
        #Middle is the bottom of the neck
        keypoint_array = ['nose','middle','left_shoulder','left_elbow','left_wrists','right_shoulder','right_wrist','right_elbow','left_hip','left_knee','left_ankle','right_hip','right_knee','right_ankle','left_eye','right_eye','left_ear','right_ear']
        
        data_array = []
        for ppl in range(len(data["Info"])):

            for k in range(len(data["Info"][ppl]["body"]["NEURAL_NETWORK"]["Poses"])):
                
                pose = {}
                confidence = data["Info"][ppl]["body"]["NEURAL_NETWORK"]["Poses"][k]['Confidence']
                
                for i in range(0, 18):
                    keypoint_u = (data["Info"][ppl]["body"]["NEURAL_NETWORK"]["Poses"][k]["Keypoints"][i]['x'])
                    keypoint_v = (data["Info"][ppl]["body"]["NEURAL_NETWORK"]["Poses"][k]["Keypoints"][i]['y'])
                    
                    pose[keypoint_array[i]] = [keypoint_u, keypoint_v, confidence]
                
                data_array.append(pose)
                
        self.data = data_array  
    def __len__(self):
        '''
        Returns the length of self.data
        
        Parameters: None
        Returns: output
                    Length of self.data
        '''
        return len(self.data)
    
    def getitem(self, idx):    
        '''
        Gets each poses keypoint detections
        
        Parameters: idx: int
                        index of pose
        Returns: output: python dictionary
                        dictionary of keypoints for each pose
        '''
        return self.data[idx]
    
    def remove(self, idx):    
        '''
        Removes an index from self.data
        
        Parameters: idx: int
                        index of pose
        Returns: output: None
        '''
        self.data = self.data.pop(idx)
        
    def new_data(self, data):   
        '''
        Replaces self.data with a new array of keypoint dictionaries
        
        Parameters: data: array of dictionaries
        Returns: output: None
        '''
        self.data = data
    
class dcpose_dataloader():
    def __init__(self, data_json):
        '''
        constructor for the dcpose_dataloader class, a class takes a json file of dcpose detections and creates an indexable object to easily access poses and keypoints.
        Parameters: data: json object
                        json file of open pose detections. If data is None, then initalize self.data as an empty array
        Returns: None
        '''
        self.average_height = None

        if data_json is None:
            self.data = []
            return

        if not isinstance(data_json, list):
            data_json = [data_json]
        
        #Middle is 'head_bottom'
        keypoint_array = ['nose', 'middle', 'head_top', 'box1', 'box2', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        
        data_array = []
        for data in data_json:
            for ppl in range(len(data["Info"])):
                
                pose = {}
                frame_name = data["Info"][ppl]["image_path"]
                for i in range(0, 17):
                    keypoint_u = data["Info"][ppl]["keypoints"][i][0]
                    keypoint_v = data["Info"][ppl]["keypoints"][i][1]
                    confidence = data["Info"][ppl]["keypoints"][i][2]
                    
                    if keypoint_array[i] == 'box1' or keypoint_array[i] == 'box2':
                        continue
                    pose[keypoint_array[i]] = [keypoint_u, keypoint_v, confidence, frame_name]
                    
                data_array.append(pose)

        self.data = data_array  

    def __len__(self):
        '''
        Returns the length of self.data
        
        Parameters: None
        Returns: output
                    Length of self.data
        '''
        return len(self.data)
    
    def getitem(self, idx):  
        '''
        Gets each poses keypoint detections
        
        Parameters: idx: int
                        index of pose
        Returns: output: python dictionary
                        dictionary of keypoints for each pose
        '''
        return self.data[idx]

    def remove(self, idx):    
        '''
        Removes an index from self.data
        
        Parameters: idx: int
                        index of pose
        Returns: output: None
        '''
        self.data = self.data.pop(idx)
        
    def new_data(self, data):   
        '''
        Replaces self.data with a new array of keypoint dictionaries
        
        Parameters: data: array of dictionaries
        Returns: output: None
        '''
        self.data = data

    def get_height(self):
        return self.average_height

    def write_height(self, height):
        self.average_height = height

class human3_6_dataloader():
    def __init__(self, data):
        '''
        constructor for the dcpose_dataloader class, a class takes a json file of dcpose detections and creates an indexable object to easily access poses and keypoints.
        Parameters: data: json object
                        json file of open pose detections. If data is None, then initalize self.data as an empty array
        Returns: None
        '''
        if data is None:
            self.data = []
            return
        
        #Middle is 'head_bottom'
        #keypoint_array = ['left_hip','left_knee','left_ankle','right_hip','right_knee','right_ankle','middle_hip','middle','nose','head','left_shoulder','left_elbow','left_wrist','right_shoulder','right_elbow','right_wrist']
        #keypoint_array = ['nose', 'middle', 'head_top', 'box1', 'box2', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        
        #keypoint_array = ['middle_hip','right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','middle','neck','jaw','head','left_shoulder','left_elbow','left_wrist','right_shoulder','right_elbow','right_wrist']
        keypoint_array = ['middle_hip','right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','middle1','neck','jaw','middle','left_shoulder','left_elbow','left_wrist','right_shoulder','right_elbow','right_wrist']

        data_array = []
        print(data.shape, " SHAPE")
        for ppl in range(data.shape[0]):
            
            pose = {}
            for i in range(0, 17):
                keypoint_u = data.reshape(-1,2,17)[ppl,0,i]
                keypoint_v = data.reshape(-1,2,17)[ppl,1,i]
                confidence = 1.0
                    
                pose[keypoint_array[i]] = [keypoint_u, keypoint_v, confidence]
                
            data_array.append(pose)
                
        self.data = data_array  
    def __len__(self):
        '''
        Returns the length of self.data
        
        Parameters: None
        Returns: output
                    Length of self.data
        '''
        return len(self.data)
    
    def getitem(self, idx):  
        '''
        Gets each poses keypoint detections
        
        Parameters: idx: int
                        index of pose
        Returns: output: python dictionary
                        dictionary of keypoints for each pose
        '''
        return self.data[idx]

    def remove(self, idx):    
        '''
        Removes an index from self.data
        
        Parameters: idx: int
                        index of pose
        Returns: output: None
        '''
        self.data = self.data.pop(idx)
        
    def new_data(self, data):   
        '''
        Replaces self.data with a new array of keypoint dictionaries
        
        Parameters: data: array of dictionaries
        Returns: output: None
        '''
        self.data = data
    
class human3_6_dataloader_gt():
    def __init__(self, data, cam, subject):
        '''
        constructor for the dcpose_dataloader class, a class takes a json file of dcpose detections and creates an indexable object to easily access poses and keypoints.
        Parameters: data: json object
                        json file of open pose detections. If data is None, then initalize self.data as an empty array
        Returns: None
        '''
        if data is None:
            self.data = []
            return

        print([i for i in data.item().keys()], " SUBJECT")
        subject_key = [i for i in data.item().keys()][subject]
        
        #Middle is 'head_bottom'
        #keypoint_array = ['left_hip','left_knee','left_ankle','right_hip','right_knee','right_ankle','middle_hip','middle','nose','head','left_shoulder','left_elbow','left_wrist','right_shoulder','right_elbow','right_wrist']
        #keypoint_array = ['nose', 'middle', 'head_top', 'box1', 'box2', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        
        #keypoint_array = ['middle_hip','right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','middle','neck','jaw','head','left_shoulder','left_elbow','left_wrist','right_shoulder','right_elbow','right_wrist']
        #data.item()['S1']['Smoking 1']
        
        keypoint_array = ['middle_hip','right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','middle1','middle','jaw','head','left_shoulder','left_elbow','left_wrist','right_shoulder','right_elbow','right_wrist']

        data_array = []
        #print(data.shape, " SHAPE")
        print(len(data.item()[subject_key].keys()), " NUMBER OF VIDEOS")
        joint_2d_array = []
        for video in data.item()[subject_key].keys():
            
            joint_array = ['middle_hip','right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','middle1','neck','jaw','middle','left_shoulder','left_elbow','left_wrist','right_shoulder','right_elbow','right_wrist']

            for ppl in range(data.item()[subject_key][video][cam].shape[0]):
                pose = {}
                joint_2d = []
                for i in range(0, 17):
                    #print(data.item()[subject_key][video][cam].shape, " SHAPE")
                    keypoint_u = data.item()[subject_key][video][cam][ppl][i][0]
                    keypoint_v = data.item()[subject_key][video][cam][ppl][i][1]
                    confidence = 1.0
                        
                    pose[keypoint_array[i]] = [keypoint_u, keypoint_v, confidence, video, ppl]

                    joint_2d.append(keypoint_u)
                    joint_2d.append(keypoint_v)
                    
                data_array.append(pose)
                joint_2d_array.append(tuple(joint_2d))
            
        print(len(list(set(joint_2d_array))), len(data_array), " UNIQUE POSES")
        self.data = data_array  
    def __len__(self):
        '''
        Returns the length of self.data
        
        Parameters: None
        Returns: output
                    Length of self.data
        '''
        return len(self.data)
    
    def getitem(self, idx):  
        '''
        Gets each poses keypoint detections
        
        Parameters: idx: int
                        index of pose
        Returns: output: python dictionary
                        dictionary of keypoints for each pose
        '''
        return self.data[idx]

    def remove(self, idx):    
        '''
        Removes an index from self.data
        
        Parameters: idx: int
                        index of pose
        Returns: output: None
        '''
        self.data = self.data.pop(idx)
        
    def new_data(self, data):   
        '''
        Replaces self.data with a new array of keypoint dictionaries
        
        Parameters: data: array of dictionaries
        Returns: output: None
        '''
        self.data = data