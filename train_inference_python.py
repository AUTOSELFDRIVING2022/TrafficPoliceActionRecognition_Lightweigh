from tracemalloc import start
import torch
import random
import numpy as np
import glob2
import os
import pandas as pd
from argparse import Namespace
from tqdm.auto import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader

#for model params and flops
import pytorch_model_summary
from thop import profile 

import albumentations
import albumentations.pytorch

from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import set_seed
from torch.optim import AdamW

import pytorchvideo.models.hub as pyvideo
#get_ipython().system('nvidia-smi')
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
from source.dataset.datasetsRGB import ActionDataset, ActionTestDataset, ActionDatasetLSTM
from source.dataset.datasetsAIHUB import ActionDatasetAttention
from source.losses import LabelSmoothingLoss
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, accuracy_score, multilabel_confusion_matrix
from source.plotcm import plot_confusion_matrix

from source.resnet18LSTM import ResNetLSTM, BasicBlock
from source.ResNetAttention import ResNetAttention, ResNetAttentionVisual
from datetime import datetime

import cv2
import csv

labelEncode = {'Go':0, 'No_signal':1, 'Slow':2, 'Stop_front':3, 'Stop_side':4,'Turn_left':5, 'Turn_right':6}
#labelEncode = {'right_to_left':0, 'left_to_right':1, 'front_stop':2, 'rear_stop':3, 'left_and_right_stop':4,'front_and_rear_stop':5}

random_seed = 42
set_seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
os.environ["PYTHONHASHSEED"] = str(random_seed)

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def counts_from_confusion(confusion):
    counts_list = []

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

        counts_list.append({'Class': i, 'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn})
        
    return counts_list

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

def train_action_rec_model(train_network="ResNetLSTM", mode="image", cfg=None):
    trainVideo = []
    validVideo = []
    
    workPath = os.path.join(cfg.save_path)
    if not os.path.exists(workPath): 
        os.mkdir(workPath)
    videoFolderTrain = sorted(glob2.glob(cfg.pathTrain + "*"))
    videoFolderVal = sorted(glob2.glob(cfg.pathVal + "*"))

    for i in range(len(videoFolderTrain)):
        trainVideo.append(videoFolderTrain[i])
        
    for i in range(len(videoFolderVal)):
        validVideo.append(videoFolderVal[i])

    albumentations_train_transform = albumentations.Compose([
        albumentations.Resize(cfg.cropped_box_size , cfg.cropped_box_size), 
        albumentations.RandomBrightnessContrast(p=0.2),
        albumentations.GaussNoise(p=0.2),
        #albumentations.MotionBlur(p=0.2),
        #albumentations.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.1),
        albumentations.Normalize(cfg.mean, cfg.std),
        albumentations.pytorch.transforms.ToTensorV2()
    ])

    albumentations_val_transform = albumentations.Compose([
        albumentations.Resize(cfg.cropped_box_size , cfg.cropped_box_size), 
        albumentations.Normalize(cfg.mean, cfg.std),
        albumentations.pytorch.transforms.ToTensorV2()
    ])
    if train_network == 'SlowFast':
        trainDataset = ActionDataset(trainVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=True, transform=albumentations_train_transform, mode=mode, slowfast_alpha=cfg.slowfast_alpha)
        trainLoader = DataLoader(trainDataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, pin_memory=True)
        
        validDataset = ActionDataset(validVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=False, transform=albumentations_val_transform, mode=mode, slowfast_alpha=cfg.slowfast_alpha)
        validLoader = DataLoader(validDataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)

        ###NET
        net = pyvideo.slowfast.slowfast_16x8_r101_50_50(model_num_class=cfg.classes)
        #modelPath = "SLOWFAST_16x8_R101_50_50.pyth"
        #net.load_state_dict(torch.load(modelPath)["model_state"])
        model = ActionBasicModule(cfg.device, net=net, classes = cfg.classes)

    elif train_network == 'ResNetLSTM':
        trainDataset = ActionDatasetLSTM(trainVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=True, transform=albumentations_train_transform, mode=mode)
        trainLoader = DataLoader(trainDataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, pin_memory=True)
        
        validDataset = ActionDatasetLSTM(validVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=False, transform=albumentations_val_transform, mode=mode)
        validLoader = DataLoader(validDataset, batch_size=1, shuffle=False)
        
        ###ResnetLSTM
        net = ResNetLSTM(BasicBlock, [2, 2, 2, 2], num_classes = cfg.classes, lstm_hidden_layer = 512, lstm_sequence_number = cfg.num_frames)
        model = ActionBasicModule(cfg.device, net=net, classes = cfg.classes)

    elif train_network == 'ResNetAttention':
        trainDataset = ActionDatasetAttention(trainVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=True, transform=albumentations_train_transform, mode=mode, img_size = cfg.cropped_box_size)
        trainLoader = DataLoader(trainDataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, pin_memory=True)
        
        validDataset = ActionDatasetAttention(validVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=False, transform=albumentations_val_transform, mode=mode, img_size = cfg.cropped_box_size)
        validLoader = DataLoader(validDataset, batch_size=1, shuffle=False)
        
        ###ResnetLSTM
        net = ResNetAttention(device = cfg.device, num_class = cfg.classes, num_layers = 1, dim = 128, hidden_dim = 128, num_heads=8, dropout_prob=0.1, max_length=cfg.num_frames, key_point=34)
        
        x_box = torch.zeros(2,cfg.num_frames,3,cfg.cropped_box_size,cfg.cropped_box_size).to(cfg.device)
        x_key = torch.zeros(2,cfg.num_frames,34).to(cfg.device)
        x = [x_box, x_key]
        print(pytorch_model_summary.summary(net,(x), show_input=True))
        
        model = ActionBasicModule(cfg.device, net=net, classes = cfg.classes)
        #model.load_state_dict(torch.load(workPath + f"/modeltype_{train_network}_{mode}_lastEpoch.pth"))
        model = model.to(cfg.device)
        model.eval()      
        
    elif train_network == 'ResNetAttentionVisual':
        trainDataset = ActionDatasetLSTM(trainVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=True, transform=albumentations_train_transform, mode=mode, db_type=cfg.db_type)
        trainLoader = DataLoader(trainDataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, pin_memory=True)
        
        validDataset = ActionDatasetLSTM(validVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=False, transform=albumentations_val_transform, mode=mode, db_type=cfg.db_type)
        validLoader = DataLoader(validDataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
        
        ###ResnetLSTM
        net = ResNetAttentionVisual(device = cfg.device, num_class = cfg.classes, num_layers = 1, dim = 128, hidden_dim = 128, num_heads=8, dropout_prob=0.1, max_length=cfg.num_frames)
        model = ActionBasicModule(cfg.device, net=net, classes = cfg.classes)
        
        if cfg.pretrained_weight != '':
            model.load_state_dict(torch.load(cfg.pretrained_weight))
        #print(pytorch_model_summary.summary(net,(x), show_input=True))
        
    ### Optimizer 
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)

    train_len = len(trainLoader.dataset)
    print(f'train data length {train_len}')
    
    val_len = len(validLoader.dataset)
    print(f'valid data length {val_len}')
    
    num_train_steps = int(train_len / (cfg.batch_size * cfg.num_workers) * cfg.max_epochs)
    #print(f'num_train_steps : {num_train_steps}')
    
    num_warmup_steps = int(num_train_steps * cfg.warmup_ratio)
    #print(f'num_warmup_steps : {num_warmup_steps}')
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=2, num_training_steps=cfg.max_epochs)

    best_acc = 0.0
    nC = 0  
    for epoch in range(cfg.max_epochs):
        total_loss = 0
        model.train()
        print("------------TRAIN------------")
        for i, d in enumerate(tqdm(trainLoader)):  
            data, label, fName, pos2d = d
            
            if train_network == 'SlowFast':
                x = [i.to(cfg.device)[...] for i in data]
            elif train_network == 'ResNetAttention': 
                x_box = data.to(cfg.device, dtype=torch.float32)
                x_key = pos2d.to(cfg.device, dtype=torch.float32)
                x = [x_box, x_key]
            else: 
                x = data.to(cfg.device, dtype=torch.float32)
    
            save_train_img = False
            if save_train_img and epoch < 2: 

                for _idx, boxImg in enumerate(x[0]):
                    _path = os.path.join(workPath,'train',str(epoch),str(fName[_idx]))  # img.jpg
                    create_directory_if_not_exists(_path)
                    for _idx_seq, __boxImg in enumerate(boxImg):
                        _fname = str(fName[_idx])+'_l'+str(label[_idx]) + '_index' + str(_idx_seq) + '.jpg'

                        save_path = os.path.join(_path,_fname)
                        boxImgCpu = __boxImg.squeeze(0).permute(1,2,0).cpu().numpy()*255

                        cv2.imwrite(save_path, cv2.cvtColor(boxImgCpu.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    nC = nC + 1

            optimizer.zero_grad()
            #output, loss = model(x, label, loss_mode="smoothing")
            output, loss = model(x, label, loss_mode="cross_entropy")
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        print("EPOCH:", epoch)
        print("train_loss:{:.6f}".format(total_loss/len(trainLoader)))
        
        print("------------VALID------------")
        model.eval()
        predicted_label = []
        gt_label = []
        action_name = []
        nC = 0   
        for i, (data, label, gt_fName, pos2d) in tqdm(enumerate(validLoader)):
            with torch.no_grad():  
                if train_network == 'SlowFast':
                    x = [i.to(cfg.device)[...] for i in data]
                elif train_network == 'ResNetAttention': 
                    x_box = data.to(cfg.device, dtype=torch.float32)
                    x_key = pos2d.to(cfg.device, dtype=torch.float32)
                    x = [x_box, x_key]
                else: 
                    x = data.to(cfg.device, dtype=torch.float32)
                
                predict = model(x, label=None)            
    
            ### Extreme 1
            index = predict.argmax(dim=-1).cpu().numpy()
            #index = predict.argmax(dim=-1)
            predicted_label.append(index)
            gt_label.append(label.numpy())
            action_name.append(gt_fName)
            
            save_val_img = False
            if save_val_img: 
                            
                _path = os.path.join(workPath,'val',str(epoch),str(gt_fName[0]))  # img.jpg
                create_directory_if_not_exists(_path)
                
                #for _idx, boxImg in enumerate(x[0]):
                for _idx, boxImg in enumerate(x):
                    for _idx_seq, __boxImg in enumerate(boxImg):
                        fname = str(gt_fName[_idx])+'_prLabel'+str(index) + '_seq_'+str(_idx_seq)+'.jpg'
                        #save_path = str(workPath / str(nC) / fname)
                        save_path = os.path.join(_path,fname)
                        boxImgCpu = __boxImg.squeeze(0).permute(1,2,0).cpu().numpy()*255
                        
                        cv2.imwrite(save_path, cv2.cvtColor(boxImgCpu.astype(np.uint8), cv2.COLOR_RGB2BGR))
                nC = nC + 1
                
        acc_val = accuracy_score(gt_label, predicted_label)
        
        save_csv = False
        if save_csv and save_val_img:
            csvFilename = os.path.join(workPath,'val',str(epoch),'val_csv.csv')  # img.jpg
            row_data = zip(action_name, gt_label, predicted_label)
            
            with open(csvFilename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # Writing the header
                csvwriter.writerow(['GT Action name and folder', 'GT', 'Pred'])
                # Writing the data
                csvwriter.writerows(row_data)

            print(f"Data successfully saved to {csvFilename}")
        
        #_labels = ['Go', 'No_signal', 'Slow', 'Stop_front', 'Stop_side','Turn_left', 'Turn_right']
        #_labels = ['right_to_left', 'left_to_right', 'front_stop', 'rear_stop','left_and_right_stop', 'front_and_rear_stop']
        _labels = ['no_signal_hand','right_to_left', 'left_to_right', 'front_stop', 'rear_stop','left_and_right_stop', 'front_and_rear_stop', 'Go', 'Turn_left', 'Turn_right','Stop_front', 'Stop_side','No_signal', 'Slow']
        
        conf_matrix = confusion_matrix(y_true = gt_label, y_pred = predicted_label, labels = None)
        #tn_fp_fn_tp = multilabel_confusion_matrix(y_true = gt_label, y_pred =  predicted_label, labels = None)
        #TP, FN, FP, TN = counts_from_confusion(conf_matrix)
        
        tp_fn_fp_tn = counts_from_confusion(conf_matrix)
        TP, FN, FP, TN = sum([x['TP'] for x in tp_fn_fp_tn]), sum([x['FN'] for x in tp_fn_fp_tn]), sum([x['FP'] for x in tp_fn_fp_tn]), sum([x['TN'] for x in tp_fn_fp_tn])

        f1_scores = f1_score(predicted_label, gt_label, average = 'weighted')
                       
        del predicted_label, gt_label, action_name
        
        print("val_acc:{:.6f}".format(acc_val))
        print("val_f1:{:.6f}".format(f1_scores))
        print("tp:{}, fn:{}, fp:{}, tn:{}".format(TP, FN, FP, TN))
        
        if best_acc < acc_val:
            print("acc increased {} to {}".format(best_acc, acc_val))
            best_acc = acc_val
            torch.save(model.state_dict(),
                    workPath + f"/modeltype_{train_network}_{mode}_{cfg.type}_best.pth")

            conf_plt = plot_confusion_matrix(conf_matrix, _labels)
            conf_plt.savefig(os.path.join(workPath,'{}_confusion_matrix_keti_best.png'.format(train_network)))
            conf_plt.close()

        torch.save(model.state_dict(),
                    workPath + f"/modeltype_{train_network}_{mode}_{cfg.type}_lastEpoch.pth")

if __name__ == "__main__":
    opt = {
        "model_name": 'ResNetAttentionVisual', #'ResNetAttention', #'ResNetAttentionVisual',
        "type": "wand",
        "db_type": "gist_aihub", #"aihub"
        "batch_size": 4,
        "num_workers": 8,
        "lr": 5e-4,
        "max_epochs": 100,
        "warmup_ratio": 0.2,
        "print_step": 100,
        "save_path": "",
        "device": "cuda",
        #"classes": 6, #aihub
        "classes": 15, #gist
        "mean": [0.0, 0.0, 0.0],
        "std": [1, 1, 1],
        "num_frames": 60,
        "sampling_rate": 1,
        "slowfast_alpha": 8,
        "num_clips": 10,
        "num_crops": 3,
        "cropped_box_size": 224,
        #"pretrained_weight": './ckp/modeltype_ResNetAttentionVisual_image_best240730.pth',
        "pretrained_weight": '',
        #"pathTrain": "/dataset/TrafficPoliceData_GIST/cropped_train2_g/",
        #"pathVal": "/dataset/TrafficPoliceData_GIST/cropped_val2_g/",
        
        # "pathTrain": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/train/",
        # "pathVal": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/val/",
        
        # Orginal
        # "pathTrain": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_train/",
        # "pathVal": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_val/",
        
        # Over 60 frames
        #"pathTrain": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_train_60/",
        #"pathVal": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_val_60/",
        
        # Over 60 frames k17
        #"pathTrain": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_hand_train10_k17/",
        #"pathVal": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_hand_val10_k17/",
        
        # Over 60 frames k17 crop 50 wand
        #"pathTrain": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_wand_train50_k17/",
        #"pathVal": "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_wand_val50_k17/",
        
        # Over 60 frame Gist Wand
        #"pathTrain": "/dataset/Gist/train/",
        #"pathVal": "/dataset/Gist/val/",
        
        "pathTrain": "/dataset/Gist_aihub_AC/train/",
        "pathVal": "/dataset/Gist_aihub_AC/val/",
    }
    args = Namespace(**opt)
    args.save_path = 'runs/'+args.model_name+'lr'+str(args.lr)+'nf'+str(args.num_frames)+'type_'+(args.type)+'db_'+args.db_type
    train_action_rec_model(train_network=args.model_name, mode="image", cfg=args)