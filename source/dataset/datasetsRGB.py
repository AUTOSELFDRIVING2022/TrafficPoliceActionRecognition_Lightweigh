import torch
import json 

import glob2
import cv2

import numpy as np
import random 
from PIL import Image
from torch.utils.data import Dataset, DataLoader, IterableDataset

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self, slowfast_alpha=4):
        super().__init__()
        self.slowfast_alpha = slowfast_alpha
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list
    
def make_circle(js, idx, img=None):
    x_list = []
    y_list = []
    color = []
    dat = js.get('sequence').get('2d_pos')[idx]
    bbox = js.get('sequence').get('bounding_box')[idx]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    for i in range(len(dat)):
        if i % 3 == 0:
            x_list.append(int(float(dat[i]) - x1))
        elif i % 3 == 1:
            y_list.append(int(float(dat[i]) - y1))
        else:
            if int(dat[i]) == 0:
                color.append((0, 0, 255))
            else:
                color.append((255, 0, 0))
    if img is None:
        img = np.zeros((int(y2-y1), int(x2-x1), 3), np.uint8) + 255
    for j in range(len(x_list)):
        img = cv2.circle(img, (x_list[j],y_list[j]), 2, color[j], 5)

    return img

labelEncode_gist = {'Go':0, 'No_signal':1, 'Slow':2, 'Stop_front':3, 'Stop_side':4,'Turn_left':5, 'Turn_right':6}
labelEncode_aihub = {'11':0, '12':1, '31':2, '32':3, '33':4,'43':5}

class ActionDataset(Dataset):
    def __init__(self, file, interval, max_len, transform=None, train=True, mode="image", slowfast_alpha=4, db_type='aihub'):
        super().__init__()
        self.file = file
        self.len = len(self.file)
        self.interval = interval
        self.max_len = max_len
        self.transform = transform
        self.train = train
        self.datalayer = PackPathway()
        self.mode = mode
        self.slowfast_alpha = slowfast_alpha
        self.db_type == db_type
    
    def __getitem__(self, idx):
        file = self.file[idx]
        folder_name = file.split("/")[-1]
        #
        
        
        if self.db_type == 'gist':
            labelStr = folder_name[31:]   
            label = labelEncode_gist[labelStr]
            label = torch.as_tensor(label, dtype=torch.long)
        else: 
            labelStr = folder_name[-2:]    
            label = labelEncode_aihub[labelStr]
            label = torch.as_tensor(label, dtype=torch.long)
        
        imageFolder = sorted(glob2.glob(file + "/*.jpg"))
    
        trainImages = []
        start = random.randint(0, len(imageFolder)-self.interval*self.max_len)
        for i in range(start, (start+self.interval*self.max_len)):
            if (i - start) % self.interval == 0:
                if self.mode == "image":
                    pil_image = Image.open(imageFolder[i])               
                    arr = np.array(pil_image)       

                if self.transform:
                    augmented = self.transform(image=arr) 
                    image = augmented['image']
                    # augmented = self.transform(arr) 
                    # image = augmented
                trainImages.append(image)
        C, H, W = image.shape
        video = torch.stack(trainImages)
        video = self._add_padding(video, self.max_len)
        
        frames = self.datalayer(video.permute(1,0,2,3))

        return frames, label, folder_name, folder_name
        
    def __len__(self):
        return self.len

    def _add_padding(self, video, max_len):
        if video.shape[0] < max_len:
            T, C, H, W = video.shape
            pad = torch.zeros(max_len-T, C, H, W)
            video = torch.cat([video, pad], dim=0)
        else:
            video = video[:max_len]

        return video
    
class ActionDatasetLSTM(Dataset):
    def __init__(self, file, interval, max_len, transform=None, train=True, mode="image", db_type='aihub'):
        super().__init__()
        self.file = file
        self.len = len(self.file)
        self.interval = interval
        self.max_len = max_len
        self.transform = transform
        self.train = train
        self.datalayer = PackPathway()
        self.mode = mode
        self.db_type = db_type
    
    def __getitem__(self, idx):
        file = self.file[idx]
        folder_name = file.split("/")[-1]
        
        if self.db_type == 'gist':
            labelStr = folder_name[31:]   
            label = labelEncode_gist[labelStr]
            label = torch.as_tensor(label, dtype=torch.long)
        elif self.db_type == 'gist_aihub':
            #labelStr = folder_name[:-5]  
            #label = labelEncode_gist[labelStr]
            label = int(folder_name.split('_')[-1])
            label = torch.as_tensor(label, dtype=torch.long)
        else: 
            labelStr = folder_name[-2:]    
            label = labelEncode_aihub[labelStr]
            label = torch.as_tensor(label, dtype=torch.long)
        
            
        imageFolder = sorted(glob2.glob(file + "/*.jpg"))

        trainImages = []
        #start = random.randint(0, len(imageFolder)-1-self.interval*self.max_len)
        start = random.randint(0, len(imageFolder)-self.interval*self.max_len)
        for i in range(start, (start+self.interval*self.max_len)):
            if (i - start) % self.interval == 0:
                
                cv2_use = True
                if cv2_use:
                    img = cv2.imread(imageFolder[i])
                    img = cv2.resize(img,(224,224))  
                    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    #img = np.ascontiguousarray(img) 
                    
                    image = img / 255  
                    image = torch.from_numpy(image)
                else: 
                    if self.mode == "image":
                        pil_image = Image.open(imageFolder[i])         
                        arr = np.array(pil_image)       
                    if self.transform:
                        augmented = self.transform(image=arr) 
                        image = augmented['image']
                        #augmented = self.transform(arr) 
                        #image = augmented
                        image = image / 255.
                
                trainImages.append(image)
        C, H, W = image.shape
        video = torch.stack(trainImages)
        video = self._add_padding(video, self.max_len)
        
        frames = video.permute(0,1,2,3)

        return video, label, folder_name, folder_name
        
    def __len__(self):
        return self.len

    def _add_padding(self, video, max_len):
        if video.shape[0] < max_len:
            T, C, H, W = video.shape
            pad = torch.zeros(max_len-T, C, H, W)
            video = torch.cat([video, pad], dim=0)
        else:
            video = video[:max_len]

        return video

class ActionTestDataset(Dataset):
    def __init__(self, file, interval, max_len, transform=None, train=True, mode="image"):
        super().__init__()
        self.file = file
        self.len = len(self.file)
        self.interval = interval
        self.max_len = max_len
        self.transform = transform
        self.train = train
        self.datalayer = PackPathway()
        self.mode = mode
    
    def __getitem__(self, idx):
        file = self.file[idx]
        imageFolder = sorted(glob2.glob(file + "/*.jpg"))
        folderName = file.split("/")[-1]
        jsonFile = file +  "/" + folderName + ".json"
        with open(jsonFile, "rb") as f:
            js = json.load(f)  

        label = None
        if "action" in js:
            label = js["action"] 
            label = torch.as_tensor(label, dtype=torch.long)

        vid = []
        for idx in range(len(js.get('sequence').get('2d_pos'))):
            img = make_circle(js, idx, img=None)
            vid.append(img)

        videos = []
        N = len(imageFolder)-1-self.interval*self.max_len
        startRange = range(0, N, int(N//1))
        for r in range(len(startRange)):
            start = startRange[r]
            trainImages = []
            for i in range(start, start+self.interval*self.max_len):
                if i % self.interval == 0:
                    if self.mode == "image":
                        pil_image = Image.open(imageFolder[i])               
                        arr = np.array(pil_image)       
                    else:
                        arr = vid[i]
                    if self.transform:
                        augmented = self.transform(image=arr) 
                        image = augmented['image']
                    trainImages.append(image)
            video = torch.stack(trainImages)
            video = self._add_padding(video, self.max_len)
            frames = self.datalayer(video.permute(1,0,2,3))
            videos.append(frames)
            #####
        #videos = torch.stack(videos)
        return videos
    def __len__(self):
        return self.len

    def _add_padding(self, video, max_len):
        if video.shape[0] < max_len:
            T, C, H, W = video.shape
            pad = torch.zeros(max_len-T, C, H, W)
            video = torch.cat([video, pad], dim=0)
        else:
            video = video[:max_len]

        return video
