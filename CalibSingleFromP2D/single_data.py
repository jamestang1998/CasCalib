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