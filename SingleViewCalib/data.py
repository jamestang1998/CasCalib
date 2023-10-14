import numpy as np
import util
import re

   
    
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