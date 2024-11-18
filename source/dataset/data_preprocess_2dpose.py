import os
import glob2
import json
from PIL import Image
import numpy as np
import shutil 
import os
import glob2
import json
from PIL import Image, ImageDraw
import numpy as np
import shutil 
import glob2
import cv2
from natsort import natsorted

import multiprocessing as mp
from multiprocessing import Pool


# if not os.path.exists("/dataset/TrafficPoliceData_GIST/train/"): os.mkdir("/dataset/TrafficPoliceData_GIST/train/")

# if not os.path.exists("/dataset/TrafficPoliceData_GIST/cropped_train/"): os.mkdir("/dataset/TrafficPoliceData_GIST/cropped_train/")
# if not os.path.exists("/dataset/TrafficPoliceData_GIST/cropped_train2/"): os.mkdir("/dataset/TrafficPoliceData_GIST/cropped_train2/")

# if not os.path.exists("/dataset/TrafficPoliceData_GIST/cropped_val/"): os.mkdir("/dataset/TrafficPoliceData/cropped_val/")
# if not os.path.exists("/dataset/TrafficPoliceData_GIST/cropped_val2/"): os.mkdir("/dataset/TrafficPoliceData/cropped_val2/")

#if not os.path.exists("/dataset/TrafficPoliceData_GIST/train/"): os.mkdir("/dataset/TrafficPoliceData_GIST/train/")

if not os.path.exists("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_wand_train50_k17/"): os.mkdir("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_wand_train50_k17/")
if not os.path.exists("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_wand_train50_k17_draw/"): os.mkdir("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_wand_train50_k17_draw/")

if not os.path.exists("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_wand_val50_k17/"): os.mkdir("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_wand_val50_k17/")
if not os.path.exists("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_wand_val50_k17_draw/"): os.mkdir("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_wand_val50_k17_draw/")

# imgPathAll = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/val/"
# savePath = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_val/"
# savePathDraw = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_val_draw/"

# imgPathAll = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/val_wand/"
# savePath = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_wand_val50_k17/"
# savePathDraw = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_wand_val50_k17_draw/"

imgPathAll = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/train_wand/"
savePath = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_wand_train50_k17/"
savePathDraw = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_wand_train50_k17_draw/"

# imgPathAll = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/train_hand/"
# savePath = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_hand_train10_k17/"
# savePathDraw = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_hand_train10_k17_draw/"

THR = 50

filePathList = natsorted(glob2.glob(imgPathAll + "*"))

def make_crop(idx):
    PathImgs = natsorted(glob2.glob(filePathList[idx] + "/*.jpg"))
    folder_name = PathImgs[0].split('/')[-2]
    jsonData = filePathList[idx] + '/' + folder_name + '.json'
    with open(jsonData, "rb") as f:
        js = json.load(f) 
    
    cropImgPath = savePath + folder_name
    cropImgPathDraw = savePathDraw + folder_name
    
    if not os.path.exists(cropImgPath): os.mkdir(cropImgPath)
    if not os.path.exists(cropImgPathDraw): os.mkdir(cropImgPathDraw)
    
    for idx, img_path in enumerate(PathImgs):   
        
        basename = os.path.basename(img_path)
        if not os.path.exists(os.path.join(cropImgPath, basename)):
            x_list = []
            y_list = []
            visible_list = []
            color = []
            dat = js.get('sequence').get('2d_pos')[idx]
            bbox = js.get('sequence').get('bounding_box')[idx]
            #x1, y1, x2, y2 = int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3]))
            x1, y1, x2, y2 = int(float(bbox[0]))-THR, int(float(bbox[1]))-THR, int(float(bbox[2]))+THR, int(float(bbox[3]))+THR
            
            # Keypoint 24 -> 17
            #remove_index = [0, 1, 2, 24, 25, 26, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
            remove_index = [0, 1, 2, 21, 22, 23, 24, 25, 26, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]
            for i in range(len(dat)):
                if i in remove_index:
                    continue
                
                if i % 3 == 0:
                    x_list.append(int(float(dat[i]) - x1))
                elif i % 3 == 1:
                    y_list.append(int(float(dat[i]) - y1))
                else:
                    if int(dat[i]) == 0:
                        color.append((0, 255, 0))
                    else:
                        color.append((255, 255, 255))
                    visible_list.append(int(dat[i]))
            
            try:
                pil_image = Image.open(img_path)
                pil_image_ = pil_image.crop((x1, y1, x2, y2))
                pil_image_draw = pil_image_.copy()
                #print(img_path, x1, y1, x2, y2)
                #if img_path is None:
                #_img = np.zeros((int(y2-y1), int(x2-x1), 3), np.uint8) + 255
                draw = ImageDraw.Draw(pil_image_draw)
                for j in range(len(x_list)):
                    #draw.point((x_list[j],y_list[j]), fill=color[j]) #xy, fill=None, outline=None, width=1
                    radius = 2
                    draw.ellipse((x_list[j] - radius, y_list[j] - radius, x_list[j] + radius, y_list[j] + radius), outline=color[j])

                basename = os.path.basename(img_path)
                pil_image_.save(os.path.join(cropImgPath, basename))
                pil_image_draw.save(os.path.join(cropImgPathDraw, basename))
                np.savez(os.path.join(cropImgPath, basename[:-3]+'npz'), x_list=np.array(x_list), y_list=np.array(y_list), visible_list=np.array(visible_list))
            except:
                print(os.path.join(cropImgPath, basename))
                
prepare_file = True
if prepare_file:
    #for idx, imgPath in enumerate(filePathList):
    #    make_crop(idx)
    
    num_cores = 64
    pool = Pool(num_cores)
    pool.map(make_crop,range(len(filePathList)))