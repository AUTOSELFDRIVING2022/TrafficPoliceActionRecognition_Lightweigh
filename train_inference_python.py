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

from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import set_seed

import pytorchvideo.models.hub as pyvideo
#get_ipython().system('nvidia-smi')
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
from source.dataset.datasets import ActionDataset, ActionTestDataset, ActionDatasetLSTM
from source.losses import LabelSmoothingLoss
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, accuracy_score
from source.plotcm import plot_confusion_matrix

from source.resnet18LSTM import ResNetLSTM, BasicBlock
from datetime import datetime

labelEncode = {'Go':0, 'No_signal':1, 'Slow':2, 'Stop_front':3, 'Stop_side':4,'Turn_left':5, 'Turn_right':6}

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

class ActionBasicModule(nn.Module):
    def __init__(self, device="cpu", net=None, classes=7):
        super().__init__()
        self.classes = classes
        self.device = device
        self.model = net
        #self.model.blocks[6].proj = nn.Linear(self.model.blocks[6].proj.in_features, self.classes, bias=True)
        self.model = self.model.to(self.device)
        

    def forward(self, x, label=None, loss_mode="smoothin", smoothing=0.0):
        x = self.model(x)
        if label is not None:
            if loss_mode == "smoothing":
                lossFunc = LabelSmoothingLoss(self.classes, smoothing=smoothing).to(self.device)
            else:
                lossFunc = nn.CrossEntropyLoss().to(self.device)
            label = label.to(self.device)    
            loss = lossFunc(x, label)
            return x, loss
        return x

def train_action_rec_model(train_network="ResNetLSTM", mode="image", cfg=None):
    trainVideo = []
    validVideo = []
    
    workPath = os.path.join(cfg.save_path, train_network)
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
        trainLoader = DataLoader(trainDataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
        
        validDataset = ActionDataset(validVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=False, transform=albumentations_val_transform, mode=mode, slowfast_alpha=cfg.slowfast_alpha)
        validLoader = DataLoader(validDataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

        ###NET
        net = pyvideo.slowfast.slowfast_16x8_r101_50_50()
        #modelPath = "SLOWFAST_16x8_R101_50_50.pyth"
        #net.load_state_dict(torch.load(modelPath)["model_state"])
        model = ActionBasicModule(cfg.device, net=net, classes = cfg.classes)
    
    elif train_network == 'ResNetLSTM':
        trainDataset = ActionDatasetLSTM(trainVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=True, transform=albumentations_train_transform, mode=mode)
        trainLoader = DataLoader(trainDataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
        
        validDataset = ActionDatasetLSTM(validVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=False, transform=albumentations_val_transform, mode=mode)
        validLoader = DataLoader(validDataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)
        
        ###ResnetLSTM
        net = ResNetLSTM(BasicBlock, [2, 2, 2, 2], num_classes = cfg.classes, lstm_hidden_layer = 512, lstm_sequence_number = 32)
        model = ActionBasicModule(cfg.device, net=net, classes = cfg.classes)

    ### Optimizer 
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)


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
        num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    best_acc = 0.0
    for epoch in range(cfg.max_epochs):
        total_loss = 0
        model.train()
        print("------------TRAIN------------")
        for i, d in enumerate(tqdm(trainLoader)):  
            data, label, _ = d
            if train_network == 'SlowFast':
                x = [i.to(cfg.device)[...] for i in data]
            else: 
                x = data.to(cfg.device)
            x_in = x
            optimizer.zero_grad()
            output, loss = model(x_in, label, loss_mode="smoothing")
            total_loss += loss 
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            #if i % args.print_step == 0:
            #    print("step:", i)
            #    print("loss:{:.2f}".format(loss.item()))
        print("EPOCH:", epoch)
        print("train_loss:{:.6f}".format(total_loss/len(trainLoader)))   
        
        model.eval()
        predicted_label = []
        gt_label = []
        for i, d in tqdm(enumerate(validLoader)):
            with torch.no_grad():
                data, label, _ = d
                if train_network == 'SlowFast':
                    x = [i.to(cfg.device)[...] for i in data]
                else: 
                    x = data.to(cfg.device)
                predict = model(x, label=None)
                predict = torch.softmax(predict, dim=-1)
                #predict = predict.mean(predict, dim=0)
                ### Extreme 1
                index = predict.argmax(dim=-1).cpu().numpy()
                predicted_label.append(index)
                gt_label.append(label.numpy())
                #logits[i] = index #1.
        acc_val = accuracy_score(gt_label, predicted_label)
        
        _labels = ['Go', 'No_signal', 'Slow', 'Stop_front', 'Stop_side','Turn_left', 'Turn_right']
        conf_matrix = confusion_matrix(y_true = gt_label, y_pred = predicted_label, labels = None)
        f1_scores = f1_score(predicted_label, gt_label, average = 'weighted')
                       
        print("val_acc:{:.6f}".format(acc_val)) 
        print("val_f1:{:.6f}".format(f1_scores)) 
        
        if best_acc < acc_val:
            print("acc increased {} to {}".format(best_acc, acc_val))
            best_acc = acc_val
            torch.save(model.state_dict(),
                    workPath + f"/modeltype_{train_network}_{mode}_best.pth")
            
            conf_plt = plot_confusion_matrix(conf_matrix, _labels)
            conf_plt.savefig(os.path.join(workPath,'{}_confusion_matrix_keti_best.png'.format(train_network)))
            conf_plt.close()
            
        torch.save(model.state_dict(),
                    workPath + f"/modeltype_{train_network}_{mode}_lastEpoch.pth")

def inference_action_rec_model(train_network="ResNetLSTM", mode="image", cfg=None):
    workPath = os.path.join(cfg.save_path, train_network)
    if not os.path.exists(workPath): 
        os.mkdir(workPath)

    videoFolderVal = sorted(glob2.glob(cfg.pathVal + "*"))
    validVideo = []
    for i in range(len(videoFolderVal)):
        validVideo.append(videoFolderVal[i])
    
    albumentations_val_transform = albumentations.Compose([
        albumentations.Resize(cfg.cropped_box_size , cfg.cropped_box_size), 
        albumentations.Normalize(cfg.mean, cfg.std),
        albumentations.pytorch.transforms.ToTensorV2()
    ])
    
    
    if train_network == 'SlowFast':    
        validDataset = ActionDataset(validVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=False, transform=albumentations_val_transform, mode=mode, slowfast_alpha=cfg.slowfast_alpha)
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
        
        model.load_state_dict(torch.load(workPath + f"/modeltype_{train_network}_{mode}_lastEpoch.pth"))
        model = model.to(cfg.device)
        model.eval()

    
    elif train_network == 'ResNetLSTM':    
        validDataset = ActionDatasetLSTM(validVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=False, transform=albumentations_val_transform, mode=mode)
        validLoader = DataLoader(validDataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)
        
        ###ResnetLSTM
        net = ResNetLSTM(BasicBlock, [2, 2, 2, 2], num_classes = cfg.classes, lstm_hidden_layer = 512, lstm_sequence_number = 32)
        
        x = torch.zeros(1,32,3,224,224)
        print(pytorch_model_summary.summary(net,(x), show_input=True))
        
        model = ActionBasicModule(cfg.device, net=net, classes = cfg.classes)
        model.load_state_dict(torch.load(workPath + f"/modeltype_{train_network}_{mode}_lastEpoch.pth"))
        model = model.to(cfg.device)
        model.eval()

    if not os.path.exists("submission"):
        os.mkdir("submission")
        
    print("------------INFERENCE------------")    
    predicted_label = []
    gt_label = []
    action_name = []
    total_elapsed_time_inf= 0
    for i, d in tqdm(enumerate(validLoader)):
        with torch.no_grad():
            data, label, _name_action = d
            if train_network == 'SlowFast':
                x = [i.to(cfg.device)[...] for i in data]
            else: 
                x = data.to(cfg.device)
            
            ### Model Inference time
            start_time_inf = datetime.now()
            
            predict = model(x, label=None)
            
            end_time_inf = datetime.now()
            elapsed_time_inf = end_time_inf - start_time_inf
            total_elapsed_time_inf += elapsed_time_inf.total_seconds()
            
            predict = torch.softmax(predict, dim=-1)
            #predict = predict.mean(predict, dim=0)
            ### Extreme 1
            index = predict.argmax(dim=-1).cpu().numpy()
            predicted_label.append(int(index))
            gt_label.append(int(label.numpy()))
            action_name.append(_name_action)
            #logits[i] = index #1.
    
    acc_val = accuracy_score(gt_label, predicted_label)
    _labels = ['Go', 'No_signal', 'Slow', 'Stop_front', 'Stop_side','Turn_left', 'Turn_right']
    conf_matrix = confusion_matrix(y_true = gt_label, y_pred = predicted_label, labels = None)
    #tn, fp, fn, tp = confusion_matrix(y_true = gt_label, y_pred = predicted_label, labels = None).ravel()
    
    f1_scores = f1_score(predicted_label, gt_label, average = 'weighted')
    metric = precision_recall_fscore_support(predicted_label, gt_label, average = 'weighted')
    
                    
    print("val_acc:{:.6f}".format(acc_val)) 
    print("val_f1:{:.6f}".format(f1_scores))   
    
    inf_time_per_action = total_elapsed_time_inf / len(validLoader)
    print("inf_time_per_action:{:.6f}".format(inf_time_per_action)) 
      
    conf_plt = plot_confusion_matrix(conf_matrix, _labels)
    conf_plt.savefig(os.path.join(workPath,'{}_inference_confusion_matrix_keti_best.png'.format(train_network)))
    conf_plt.close()
            
    submission = pd.read_csv("inference_result.csv")
    submission['file_path'] = action_name
    submission['GT'] = gt_label
    submission['Pred'] = predicted_label
    submission.to_csv(f"valid_result.csv", index=False)
    
if __name__ == "__main__":
    
    opt = {
        "batch_size": 4,
        "num_workers": 2,
        "lr": 5e-5,
        "max_epochs": 10,
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
    train_action_rec_model(train_network="SlowFast", mode="image", cfg=args)
    inference_action_rec_model(train_network="SlowFast", mode="image", cfg=args)
    
    train_action_rec_model(train_network="ResNetLSTM", mode="image", cfg=args)
    inference_action_rec_model(train_network="ResNetLSTM", mode="image", cfg=args)