import os
from pathlib import Path

def rename_folders(directory):
    # Iterate through all subdirectories in the given directory
    for folder in Path(directory).iterdir():
        if folder.is_dir():  # Ensure it's a directory
            folder_name = folder.name
            
            # Check if the folder name ends with '_11' or '_12'
            if folder_name.endswith('_11'):
                new_name = folder_name + '_0001'
            elif folder_name.endswith('_12'):
                new_name = folder_name + '_0002'
            elif folder_name.endswith('_31'):
                new_name = folder_name + '_0003'
            elif folder_name.endswith('_32'):
                new_name = folder_name + '_0004'
            elif folder_name.endswith('_33'):
                new_name = folder_name + '_0005'
            elif folder_name.endswith('_43'):
                new_name = folder_name + '_0006'
            elif folder_name.endswith('_Go'):
                new_name = folder_name + '_0007'
            elif folder_name.endswith('_Turn_left'):
                new_name = folder_name + '_0008'
            elif folder_name.endswith('_Turn_right'):
                new_name = folder_name + '_0009'
            elif folder_name.endswith('_Stop_front'):
                new_name = folder_name + '_0010'
            elif folder_name.endswith('_Stop_side'):
                new_name = folder_name + '_0011'
            elif folder_name.endswith('_No_signal'):
                new_name = folder_name + '_0012'
            elif folder_name.endswith('_Slow'):
                new_name = folder_name + '_0013'
            else:
                continue  # Skip folders that don't match the condition

            # Construct the full new path
            new_folder_path = folder.parent / new_name
            
            # Rename the folder
            os.rename(folder, new_folder_path)
            print(f"Renamed: {folder} -> {new_folder_path}")

# Example usage
directory_path = '/dataset/Gist_aihub_AC/train/gist_over_60/'
rename_folders(directory_path)