   
class coco_mmpose_dataloader():

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