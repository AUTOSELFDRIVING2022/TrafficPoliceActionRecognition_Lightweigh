import torch
import random
import numpy as np
import os
from argparse import Namespace
from datetime import datetime
import glob2
from tqdm.auto import tqdm
import albumentations
import albumentations.pytorch

import csv
import cv2 
from pathlib import Path
import glob 

os.environ["CUDA_VISIBLE_DEVICES"]= "0"

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

class LoadImages:  # for inference
    def __init__(self, path, img_size=224, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

       
        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, 'Image Not Found ' + path
        #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        #img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)


            
        return path, img, img0, self.cap

#from train_inference_python import ActionBasicModule#, inference_action_rec_model

#labelEncode = {'Go':0, 'No_signal':1, 'Slow':2, 'Stop_front':3, 'Stop_side':4,'Turn_left':5, 'Turn_right':6}
labelEncode = {'right_to_left':0, 'left_to_right':1, 'front_stop':2, 'rear_stop':3, 'left_and_right_stop':4,'front_and_rear_stop':5}

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
os.environ["PYTHONHASHSEED"] = str(random_seed)
from torchvision import transforms
from source.dataset.datasetsRGB import ActionDataset, ActionTestDataset, ActionDatasetLSTM
from source.dataset.datasetsAIHUB import ActionDatasetAttention
from torch.utils.data import DataLoader
import pytorch_model_summary
from source.resnet18LSTM import ResNetLSTM, BasicBlock
import pytorchvideo.models.hub as pyvideo
import torch.nn as nn
from source.losses import LabelSmoothingLoss
from source.ResNetAttention import ResNetAttention

from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, accuracy_score, multilabel_confusion_matrix
from source.plotcm import plot_confusion_matrix

def counts_from_confusion(confusion):
    """
    Obtain TP, FN FP, and TN for each class in the confusion matrix
    """

    counts_list = []

    # Iterate through classes and store the counts
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]

        fn_mask = np.zeros(confusion.shape)
        fn_mask[i, :] = 1
        fn_mask[i, i] = 0
        fn = np.sum(np.multiply(confusion, fn_mask))

        fp_mask = np.zeros(confusion.shape)
        fp_mask[:, i] = 1
        fp_mask[i, i] = 0
        fp = np.sum(np.multiply(confusion, fp_mask))

        tn_mask = 1 - (fn_mask + fp_mask)
        tn_mask[i, i] = 0
        tn = np.sum(np.multiply(confusion, tn_mask))

        # counts_list.append({'Class': i,
        #                     'TP': tp,
        #                     'FN': fn,
        #                     'FP': fp,
        #                     'TN': tn})
        return tp, fn, fp, tn
    
class ActionBasicModule(nn.Module):
    def __init__(self, device="cpu", net=None, classes=7):
        super().__init__()
        self.classes = classes
        self.device = device
        self.model = net
        #self.model.blocks[6].proj = nn.Linear(self.model.blocks[6].proj.in_features, self.classes, bias=True)
        self.model = self.model.to(self.device)
        

    def forward(self, x, label=None, loss_mode="smoothin", smoothing=0.0):
        pred = self.model(x)
        if label is not None:
            if loss_mode == "smoothing":
                lossFunc = LabelSmoothingLoss(self.classes, smoothing=smoothing).to(self.device)
            else:
                lossFunc = nn.CrossEntropyLoss().to(self.device)
            label = label.to(self.device)    
            loss = lossFunc(pred, label)
            return pred, loss
        return pred
    
def inference_action_rec_model_time(train_network="ResNetLSTM", mode="image", cfg=None):
    workPath = os.path.join(cfg.save_path)
    if not os.path.exists(workPath): 
        os.mkdir(workPath)

    videoFolderVal = sorted(glob2.glob(cfg.pathVal + "*"))
    validVideo = []
    for i in range(len(videoFolderVal)):
        validVideo.append(videoFolderVal[i])
    
    # data_transformation = transforms.Compose([
    #                                       transforms.ToTensor(),
    #                                       transforms.Resize((cfg.cropped_box_size,cfg.cropped_box_size)),
    #                                       transforms.Normalize(cfg.mean, cfg.std)
    # ])
    
    data_transformation = albumentations.Compose([
        albumentations.Resize(cfg.cropped_box_size , cfg.cropped_box_size), 
        albumentations.Normalize(cfg.mean, cfg.std),
        albumentations.pytorch.transforms.ToTensorV2()
        ])
    
    if train_network == 'SlowFast':    
        validDataset = ActionDataset(validVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=False, transform=data_transformation, mode=mode, slowfast_alpha=cfg.slowfast_alpha)
        validLoader = DataLoader(validDataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

        ###NET
        net = pyvideo.slowfast.slowfast_16x8_r101_50_50()
        #modelPath = "SLOWFAST_16x8_R101_50_50.pyth"
        #net.load_state_dict(torch.load(modelPath)["model_state"])
        sample_inputA = torch.zeros(1,3, 8,224,224)
        sample_inputB = torch.zeros(1,3, 32 ,224,224)
        sample_input = [sample_inputA, sample_inputB]
        x = [i[...] for i in sample_input]
        print(pytorch_model_summary.summary(net,(x), show_input=True, show_hierarchical=False))
        # flops, params = profile(net, inputs=(x, ),verbose=True)
        # print(flops)
        # print(params)
        #print(torchsummary.summary(net,((1,3,8,224,224),(1,3,32,224,224)), device='cuda'))
        model = ActionBasicModule(cfg.device, net=net, classes = cfg.classes)
        
        #model.load_state_dict(torch.load(workPath + f"/modeltype_{train_network}_{mode}_lastEpoch.pth"))
        model = model.to(cfg.device)
        model.eval()
    elif train_network == 'ResNetLSTM':    
        validDataset = ActionDatasetLSTM(validVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=False, transform=data_transformation, mode=mode)
        validLoader = DataLoader(validDataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)
        
        ###ResnetLSTM
        net = ResNetLSTM(BasicBlock, [2, 2, 2, 2], num_classes = cfg.classes, lstm_hidden_layer = 512, lstm_sequence_number = cfg.num_frames)
        
        x = torch.zeros(1,cfg.num_frames,3,224,224)
        print(pytorch_model_summary.summary(net,(x), show_input=True))
        
        model = ActionBasicModule(cfg.device, net=net, classes = cfg.classes)
        model.load_state_dict(torch.load(cfg.model_weight))
        model = model.to(cfg.device)
        model.eval()
    elif train_network == 'ResNetAttention':
        validDataset = ActionDatasetAttention(validVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=False, transform=data_transformation, mode=mode, img_size = cfg.cropped_box_size)
        validLoader = DataLoader(validDataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
        
        ###ResnetLSTM
        net = ResNetAttention(device = cfg.device, num_class = cfg.classes, num_layers = 1, dim = 128, hidden_dim = 128, num_heads=8, dropout_prob=0.1, max_length=cfg.num_frames, key_point=34)
        
        x_box = torch.zeros(2,cfg.num_frames,3,224,224).to(cfg.device)
        x_key = torch.zeros(2,cfg.num_frames,34).to(cfg.device)
        x = [x_box, x_key]
        print(pytorch_model_summary.summary(net,(x), show_input=True))
        
        model = ActionBasicModule(cfg.device, net=net, classes = cfg.classes)
        model.load_state_dict(torch.load(cfg.model_weight))
        model = model.to(cfg.device)
        model.eval()    
    elif train_network == 'ResNetAttentionVisual':

        validLoader = LoadImages(cfg.predPath)
        
        ###ResnetLSTM
        net = ResNetAttentionVisual(device = cfg.device, num_class = cfg.classes, num_layers = 1, dim = 128, hidden_dim = 128, num_heads=8, dropout_prob=0.1, max_length=cfg.num_frames)
        model = ActionBasicModule(cfg.device, net=net, classes = cfg.classes)

    if not os.path.exists("submission"):
        os.mkdir("submission")
        
    print("------------INFERENCE------------")    
    predicted_label = []
    gt_label = []
    action_name = []
    total_elapsed_time_inf= 0
    for i, d in tqdm(enumerate(validLoader)):
        with torch.no_grad():
            data, label, _name_action, pos2d = d
            if train_network == 'SlowFast':
                x = [i.to(cfg.device)[...] for i in data]
            elif train_network == 'ResNetAttention': 
                x_box = data.to(cfg.device, dtype=torch.float32)
                x_key = pos2d.to(cfg.device, dtype=torch.float32)
                x = [x_box, x_key]
            else: 
                x = data.to(cfg.device, dtype=torch.float32)
            
            ### Model Inference time
            start_time_inf = datetime.now()
            
            predict = model(x, label=None)
            
            end_time_inf = datetime.now()
            elapsed_time_inf = end_time_inf - start_time_inf
            total_elapsed_time_inf += elapsed_time_inf.total_seconds()
            
            #predict = torch.softmax(predict, dim=-1)
            #predict = predict.mean(predict, dim=0)
            ### Extreme 1
            index = predict.argmax(dim=-1).cpu().numpy()
            predicted_label.append(int(index))
            gt_label.append(int(label.numpy()))
            action_name.append(_name_action)
            #logits[i] = index #1.
    
    save_csv = True
    if save_csv:
        csvFilename = './val_csv.csv'
        row_data = zip(action_name, gt_label, predicted_label)
        
        with open(csvFilename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # Writing the header
            csvwriter.writerow(['GT Action name and folder', 'GT', 'Pred'])
            
            # Writing the data
            csvwriter.writerows(row_data)

        print(f"Data successfully saved to {csvFilename}")
                
    inf_time_per_action = total_elapsed_time_inf / len(validLoader)
    
    acc_val = accuracy_score(gt_label, predicted_label)
        
    #_labels = ['Go', 'No_signal', 'Slow', 'Stop_front', 'Stop_side','Turn_left', 'Turn_right']
    _labels = ['right_to_left', 'left_to_right', 'front_stop', 'rear_stop','left_and_right_stop', 'front_and_rear_stop']
    
    conf_matrix = confusion_matrix(y_true = gt_label, y_pred = predicted_label, labels = None)
    #tn_fp_fn_tp = multilabel_confusion_matrix(y_true = gt_label, y_pred =  predicted_label, labels = None)
    TP, FN, FP, TN = counts_from_confusion(conf_matrix)
    
    conf_plt = plot_confusion_matrix(conf_matrix, _labels)
    conf_plt.savefig(os.path.join(workPath,'{}_inference_confusion_matrix_keti_best.png'.format(train_network)))
    conf_plt.close()

    f1_scores = f1_score(predicted_label, gt_label, average = 'weighted')
                    
    print("val_acc:{:.6f}".format(acc_val)) 
    print("val_f1:{:.6f}".format(f1_scores)) 
    print("tp:{}, fn:{}, fp:{}, tn:{}".format(TP, FN, FP, TN))
    
    print("inf_time_per_action:{:.6f}".format(inf_time_per_action)) 
    print("inf_time_fps:{:.6f}".format(len(validLoader) / total_elapsed_time_inf))
    
if __name__ == "__main__":
    
    opt = {
        "model_name": 'ResNetAttention',
        "batch_size": 4,
        "num_workers": 2,
        "lr": 5e-5,
        "max_epochs": 100,
        "warmup_ratio": 0.2,
        "print_step": 100,
        "save_path": "model_weights",
        "model_weight": './ckp/modeltype_ResNetAttention_image_best.pth',
        "device": "cuda",
        "classes": 6, 
        "mean": [0.0, 0.0, 0.0],
        "std": [1, 1, 1],
        "num_frames": 60,
        "sampling_rate": 1,
        "frames_per_second": 32,
        "slowfast_alpha": 4,
        "num_clips": 10,
        #"num_clips": 32,
        "num_crops": 3,
        "cropped_box_size": 224,
        
        #"pathTrain": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_train/",
        #"pathVal": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_val_60/",  
        
        # Over 60 frames k17
        "pathTrain": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_train50_k17/",
        "pathVal": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_val50_k17/",
    }
    args = Namespace(**opt)
    args.save_path = 'runs/'+args.model_name+'lr'+str(args.lr)+'nf'+str(args.num_frames)
    inference_action_rec_model_time(train_network=args.model_name, mode="image", cfg=args)