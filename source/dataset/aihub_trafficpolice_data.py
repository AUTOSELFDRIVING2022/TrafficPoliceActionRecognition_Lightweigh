import os
import json
import shutil

# Set the source folder and destination folder
#src_folder = '/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/val'
#dst_folder = '/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/val_hand'

# Set the source folder and destination folder
src_folder = '/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/train'
dst_folder = '/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/train_wand'

# Loop through all subfolders in the source folder
for folder in os.listdir(src_folder):
    folder_path = os.path.join(src_folder, folder)
    if os.path.isdir(folder_path):
        # Open the JSON file in the subfolder
        json_file_name = folder + '.json'
        json_file = os.path.join(folder_path, json_file_name)
        with open(json_file, 'r') as f:
            jdata = json.load(f)
        
        signal_type = jdata['info']['hs_tool_yn']
        actor_direction = jdata['info']['direction']
        record_time = jdata['info']['recode_time']
        clothes = jdata['info']['actor_cloth']
        
        # Check::: wand and only front signal
        if signal_type == '있음' and actor_direction != '후면' and actor_direction != '좌측면' and actor_direction != '우측면' and clothes != '헌병' and clothes != '모범운전자' and clothes != '자치경찰':
            
        # Check::: hand and only front signal
        #if signal_type == '없음' and actor_direction != '후면' and actor_direction != '좌측면' and actor_direction != '우측면' and clothes != '헌병' and clothes != '모범운전자' and clothes != '자치경찰':
            # Copy the subfolder to the destination folder
            shutil.copytree(folder_path, os.path.join(dst_folder, folder))
            print(f"Copied {folder} to {dst_folder}")
        else:
            print(f"Skipping {folder} due to other type: Hand, Backward")