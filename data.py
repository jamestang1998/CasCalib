import pdb
#pdb.set_trace()

class alphapose_dataloader():
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
        keypoint_array = ["Nose", "LEye", "REye", "LEar", "REar", "left_shoulder", "right_shoulder", "LElbow", "RElbow", "LWrist", "RWrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "Head", "Neck", "Hip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"]
        
        data_obj = {}
        for img in range(0, len(data_json)):
            pose = {}

            #if len(list(data_obj.keys())) == 100:
            #    break
            '''
            {0,  "Nose"},
            {1,  "LEye"},
            {2,  "REye"},
            {3,  "LEar"},
            {4,  "REar"},
            {5,  "LShoulder"},
            {6,  "RShoulder"},
            {7,  "LElbow"},
            {8,  "RElbow"},
            {9,  "LWrist"},
            {10, "RWrist"},
            {11, "LHip"},
            {12, "RHip"},
            {13, "LKnee"},
            {14, "Rknee"},
            {15, "LAnkle"},
            {16, "RAnkle"},
            {17,  "Head"},
            {18,  "Neck"},
            {19,  "Hip"},
            {20, "LBigToe"},
            {21, "RBigToe"},
            {22, "LSmallToe"},
            {23, "RSmallToe"},
            {24, "LHeel"},
            {25, "RHeel"},
            '''
            #fig, ax = plt.subplots()
            #if int(data_json["Info"][img]["frame"]) > 30:
            #    break
            #if int(data_json["Info"][img]["frame"]) != 154:
            #    continue
            kp_name = 0
            for i in range(0, 51, 3):
                keypoint_u = (data_json[img]["keypoints"][i])
                keypoint_v = (data_json[img]["keypoints"][i + 1])
                confidence = (data_json[img]["keypoints"][i + 2])
                pose[keypoint_array[kp_name]] = [keypoint_u, keypoint_v, confidence]
                    
                pose[keypoint_array[kp_name]] = [keypoint_u, keypoint_v, confidence]
                
                #ax.scatter(keypoint_u, keypoint_v)
                #ax.annotate(keypoint_array[i] + ' ' + str(i), (keypoint_u, keypoint_v))
                kp_name = kp_name + 1

            frame_name = data_json[img]["image_id"]
            #print(frame_name, " THIS IS THE FRANE")
            if int(frame_name.split(".")[0]) in data_obj:
                data_obj[int(frame_name.split(".")[0])].append(pose)
            else:
                data_obj[int(frame_name.split(".")[0])] = [pose]

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

    def write_height(self, height):  
        self.average_height = height

class vitpose_easy_dataloader():
    def __init__(self, data_json):
        '''
        constructor for the dcpose_dataloader class, a class takes a json file of dcpose detections and creates an indexable object to easily access poses and keypoints.
        Parameters: data: json object
                        json file of open pose detections. If data is None, then initalize self.data as an empty array
        Returns: None

        This is for a single frame
        '''
        self.average_height = None

        if data_json is None:
            self.data = []
            return

        #if not isinstance(data_json, list):
        #    data_json = [data_json]
        
        #Middle is 'head_bottom'
        #keypoint_array = ['nose', 'middle', 'head_top', 'box1', 'box2', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        
        keypoint_dict = {"0": "nose", "1": "left_eye", "2": "right_eye", "3": "left_ear", "4": "right_ear", "5": "left_shoulder", "6": "right_shoulder", "7": "left_elbow", "8": "right_elbow", "9": "left_wrist", "10": "right_wrist", "11": "left_hip", "12": "right_hip", "13": "left_knee", "14": "right_knee", "15": "left_ankle", "16": "right_ankle", "17": "left_big_toe", "18": "left_small_toe", "19": "left_heel", "20": "right_big_toe", "21": "right_small_toe", "22": "right_heel", "23": "face-0", "24": "face-1", "25": "face-2", "26": "face-3", "27": "face-4", "28": "face-5", "29": "face-6", "30": "face-7", "31": "face-8", "32": "face-9", "33": "face-10", "34": "face-11", "35": "face-12", "36": "face-13", "37": "face-14", "38": "face-15", "39": "face-16", "40": "face-17", "41": "face-18", "42": "face-19", "43": "face-20", "44": "face-21", "45": "face-22", "46": "face-23", "47": "face-24", "48": "face-25", "49": "face-26", "50": "face-27", "51": "face-28", "52": "face-29", "53": "face-30", "54": "face-31", "55": "face-32", "56": "face-33", "57": "face-34", "58": "face-35", "59": "face-36", "60": "face-37", "61": "face-38", "62": "face-39", "63": "face-40", "64": "face-41", "65": "face-42", "66": "face-43", "67": "face-44", "68": "face-45", "69": "face-46", "70": "face-47", "71": "face-48", "72": "face-49", "73": "face-50", "74": "face-51", "75": "face-52", "76": "face-53", "77": "face-54", "78": "face-55", "79": "face-56", "80": "face-57", "81": "face-58", "82": "face-59", "83": "face-60", "84": "face-61", "85": "face-62", "86": "face-63", "87": "face-64", "88": "face-65", "89": "face-66", "90": "face-67", "91": "left_hand_root", "92": "left_thumb1", "93": "left_thumb2", "94": "left_thumb3", "95": "left_thumb4", "96": "left_forefinger1", "97": "left_forefinger2", "98": "left_forefinger3", "99": "left_forefinger4", "100": "left_middle_finger1", "101": "left_middle_finger2", "102": "left_middle_finger3", "103": "left_middle_finger4", "104": "left_ring_finger1", "105": "left_ring_finger2", "106": "left_ring_finger3", "107": "left_ring_finger4", "108": "left_pinky_finger1", "109": "left_pinky_finger2", "110": "left_pinky_finger3", "111": "left_pinky_finger4", "112": "right_hand_root", "113": "right_thumb1", "114": "right_thumb2", "115": "right_thumb3", "116": "right_thumb4", "117": "right_forefinger1", "118": "right_forefinger2", "119": "right_forefinger3", "120": "right_forefinger4", "121": "right_middle_finger1", "122": "right_middle_finger2", "123": "right_middle_finger3", "124": "right_middle_finger4", "125": "right_ring_finger1", "126": "right_ring_finger2", "127": "right_ring_finger3", "128": "right_ring_finger4", "129": "right_pinky_finger1", "130": "right_pinky_finger2", "131": "right_pinky_finger3", "132": "right_pinky_finger4"}

        data_array = []
        for frame in range(len(data_json['keypoints'])):
            for ppl in data_json["keypoints"][frame].keys():
                
                pose = {}
                #frame_name = data_json["keypoints"][0][ppl]
                for i in keypoint_dict.keys():
                    #print(data_json["keypoints"][frame][ppl])
                    keypoint_u = data_json["keypoints"][frame][ppl][int(i)][0]
                    keypoint_v = data_json["keypoints"][frame][ppl][int(i)][1]
                    confidence = data_json["keypoints"][frame][ppl][int(i)][2]
                    
                    pose[keypoint_dict[i]] = [keypoint_u, keypoint_v, confidence, frame]
                    
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

    def write_height(self, height):  
        self.average_height = height