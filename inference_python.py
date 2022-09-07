import torch
import random
import numpy as np
import os
from argparse import Namespace

os.environ["CUDA_VISIBLE_DEVICES"]= "0"

from train_inference_python import inference_action_rec_model

labelEncode = {'Go':0, 'No_signal':1, 'Slow':2, 'Stop_front':3, 'Stop_side':4,'Turn_left':5, 'Turn_right':6}

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
os.environ["PYTHONHASHSEED"] = str(random_seed)

if __name__ == "__main__":
    
    opt = {
        "batch_size": 4,
        "num_workers": 2,
        "lr": 5e-5,
        "max_epochs": 100,
        "warmup_ratio": 0.2,
        "print_step": 100,
        "save_path": "model_weights",
        "device": "cuda",
        "classes": 7, 
        "mean": [0.45, 0.45, 0.45],
        "std": [0.225, 0.225, 0.225],
        "num_frames": 32,
        "sampling_rate": 1,
        "frames_per_second": 30,
        "slowfast_alpha": 4,
        "num_clips": 10,
        "num_crops": 3,
        "cropped_box_size": 224,
        "pathTrain": "/dataset/TrafficPoliceData_GIST/cropped_train2_g/",
        "pathVal": "/dataset/TrafficPoliceData_GIST/cropped_val2_g/",  
    } 
    args = Namespace(**opt)
    
    inference_action_rec_model(train_network="SlowFast", mode="image", cfg=args)
    inference_action_rec_model(train_network="ResNetLSTM", mode="image", cfg=args)