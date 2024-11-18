import os
import glob2
import json
from PIL import Image
import numpy as np
import shutil 

#if not os.path.exists("/dataset/TrafficPoliceData_GIST/train/"): os.mkdir("/dataset/TrafficPoliceData_GIST/train/")

#if not os.path.exists("/dataset/TrafficPoliceData_GIST/cropped_train/"): os.mkdir("/dataset/TrafficPoliceData_GIST/cropped_train/")
#if not os.path.exists("/dataset/TrafficPoliceData_GIST/cropped_train2/"): os.mkdir("/dataset/TrafficPoliceData_GIST/cropped_train2/")

#if not os.path.exists("/dataset/TrafficPoliceData_GIST/cropped_val/"): os.mkdir("/dataset/TrafficPoliceData/cropped_val221028/")
#if not os.path.exists("/dataset/TrafficPoliceData_GIST/cropped_val2/"): os.mkdir("/dataset/TrafficPoliceData/cropped_val221028_2/")

ACTION_TYPE = 'wand'
#ACTION_TYPE = 'hand' #not implemented
assert ACTION_TYPE == 'wand'
margins_size_w = 50
margins_size_h = 50

# imgPathAll = "/dataset/TrafficPoliceData_GIST/train/"
# jsonPathAll = "/dataset/TrafficPoliceData_GIST/train/"

# newCroppedPath = "/dataset/TrafficPoliceData_GIST/cropped_train/"
# newCroppedPath2 = "/dataset/TrafficPoliceData_GIST/cropped_train2/"

imgPathAll = "/dataset/Gist/train_all/"
jsonPathAll = "/dataset/Gist/train_all/"

newCroppedPath = "/dataset/Gist/Static_crop/"
newCroppedPath2 = "/dataset/Gist/Static_crop2/"

prepare_file = False
if prepare_file:
    filePath = "/dataset/Gist/Static/"
    filePathList = sorted(glob2.glob(filePath + "*"))

    for idx, labels in enumerate(filePathList):
        _label = os.path.basename(labels)
        labelPathImg = sorted(glob2.glob(labels + "/*"))
        for folder in labelPathImg:
            _fName = os.path.basename(folder) + '_' + str(_label)
            
            shutil.copytree(folder, os.path.join(imgPathAll,_fName))
        print(labelPathImg)
    print(filePathList)

# path = "/dataset/TrafficPoliceData/test/"
# train = sorted(glob2.glob(path + "*"))
# for i in range(len(train)):
#     train_img = sorted(glob2.glob(train[i] + "/*.jpg"))
#     folder_name = train[i].split("/")[-1]
#     for j in range(len(train_img)):
#         filename = train_img[j].split("/")[-1].split(".jpg")[0]
#         new_filename = str(filename).zfill(3) + ".jpg"
#         #print(train_img[j], '=>', train[i] + "/" + new_filename)
#         os.rename(train_img[j], train[i] + "/" + new_filename)

# path = "/dataset/TrafficPoliceData/train/"
# train = sorted(glob2.glob(path + "*"))
# for i in range(len(train)):
#     train_img = sorted(glob2.glob(train[i] + "/*.jpg"))
#     folder_name = train[i].split("/")[-1]
#     for j in range(len(train_img)):
#         filename = train_img[j].split("/")[-1].split(".jpg")[0]
#         new_filename = str(filename).zfill(3) + ".jpg"
#         #print(train_img[j], '=>', train[i] + "/" + new_filename)
#         os.rename(train_img[j], train[i] + "/" + new_filename)


trainFolder = sorted(glob2.glob(imgPathAll + "*"))
for idx, _folder in enumerate(trainFolder):
    train_img = sorted(glob2.glob(_folder + "/*.jpg"))
    folder_name = _folder.split("/")[-1]
    label = folder_name[31:]
    #print("folderName:", folder_name)

    train_json = sorted(glob2.glob(_folder + "/*.json"))
    #train_json_path = path + folder_name + f"/{folder_name}.json"
    #with open(train_json_path, "rb") as f:
    #    js = json.load(f)  
    _newCroppedPath = os.path.join(newCroppedPath, folder_name)
    _newCroppedPath2 = os.path.join(newCroppedPath2, folder_name)
    
    if not os.path.exists(_newCroppedPath):
        os.mkdir(_newCroppedPath)
    if not os.path.exists(_newCroppedPath2):
        os.mkdir(_newCroppedPath2)
            
    #newTrainJson = new_path + folder_name + f"/{folder_name}.json"
    #newTrainJson2 = new_path2 + folder_name + f"/{folder_name}.json"
    #with open(newTrainJson, "w") as f:
    #    json.dump(js, f)
    #with open(newTrainJson2, "w") as f:
    #    json.dump(js, f)
    
    for _idx, (imgFile, jsonFile) in enumerate(zip(train_img, train_json)):
        pil_image = Image.open(imgFile)
        #arr = np.array(pil_image)
        with open(jsonFile, "rb") as f:
            js = json.load(f)  
        
        annotations = js[1].get('annotations')
        bounding_box = []
        for _anno in annotations:
            try:
                if _anno['iswand'] == 1 and ACTION_TYPE == 'wand':
                    bounding_box = _anno['bbox']
                    break
            except: 
                pass
        
        if len(bounding_box) == 0:
            break

        pil_image_ = pil_image.crop((bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]))
        filename = os.path.basename(imgFile)
        pil_image_.save(_newCroppedPath + f"/{filename}")


        bbox = []
        
        bbox.append((bounding_box[0] - margins_size_w) if (bounding_box[0] - margins_size_w) > 0 else 0 )
        bbox.append((bounding_box[1] - margins_size_h) if (bounding_box[1] - margins_size_h) > 0 else 0 )
        bbox.append(bounding_box[2] + margins_size_w * 2 if (bounding_box[2] + margins_size_w * 2) < pil_image.width else bounding_box[2] + margins_size_w)
        bbox.append(bounding_box[3] + margins_size_h * 2 if (bounding_box[3] + margins_size_h * 2) < pil_image.height else bounding_box[3] + margins_size_h)
        
        pil_image__ = pil_image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        filename = os.path.basename(imgFile)
        pil_image__.save(_newCroppedPath2 + f"/{filename}")
        #print(bounding_box)
        #print(bbox)
        # bbox = []
        # for i in range(len(bounding_box)):
        #     bbox.append(float(bounding_box[i]))

        # pil_image_ = pil_image.crop(bbox)
        
        # filename = str(j).zfill(3)
        # pil_image_.save(new_folder_path2 + f"/{filename}.jpg")

# path = "/dataset/TrafficPoliceData/train/"
# new_path = "/dataset/TrafficPoliceData/cropped_train/"
# new_path2 = "/dataset/TrafficPoliceData/cropped_train2/"
# train = sorted(glob2.glob(path + "*"))
# for i in range(len(train)):
#     if i > -1:
#         train_img = sorted(glob2.glob(train[i] + "/*.jpg"))
#         folder_name = train[i].split("/")[-1]
#         print("folderName:", folder_name)
#         train_json_path = path + folder_name + f"/{folder_name}.json"
#         with open(train_json_path, "rb") as f:
#             js = json.load(f)  
#         new_folder_path = new_path + folder_name
#         new_folder_path2 = new_path2 + folder_name
#         if not os.path.exists(new_folder_path):
#             os.mkdir(new_folder_path)
#         if not os.path.exists(new_folder_path2):
#             os.mkdir(new_folder_path2)
                
#         newTrainJson = new_path + folder_name + f"/{folder_name}.json"
#         newTrainJson2 = new_path2 + folder_name + f"/{folder_name}.json"
#         with open(newTrainJson, "w") as f:
#             json.dump(js, f)
#         with open(newTrainJson2, "w") as f:
#             json.dump(js, f)
        
#         for j in range(len(train_img)):
#             pil_image = Image.open(train_img[j])

#             #arr = np.array(pil_image)
#             bounding_box = js.get('sequence').get('bounding_box')[j]
#             bbox = []
#             for i in range(len(bounding_box)):
#                 if i > 1:
#                     bbox.append(float(bounding_box[i]) + 20)
#                 else:
#                     bbox.append(float(bounding_box[i]) - 20)

#             pil_image_ = pil_image.crop(bbox)
            
#             filename = str(j).zfill(3)
#             pil_image_.save(new_folder_path + f"/{filename}.jpg")

#             bbox = []
#             for i in range(len(bounding_box)):
#                 bbox.append(float(bounding_box[i]))

#             pil_image_ = pil_image.crop(bbox)
            
#             filename = str(j).zfill(3)
#             pil_image_.save(new_folder_path2 + f"/{filename}.jpg")
