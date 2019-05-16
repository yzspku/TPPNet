import os
import torch
import torchvision
import time
import os,sys
from torchvision import transforms
import torch, torch.utils
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import bisect
from torch import nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import math
from utility import *
from config import opt
from tqdm import tqdm
from cqt_loader import *
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool



out_dir = 'changed_npy/covers80_cqt_npy_time07/'
print(out_dir)
def CQT(args):
    try:
        in_path, out_path = args
        data, sr = librosa.load(in_path)
    
        data = librosa.effects.pitch_shift(data, sr, n_steps=8)
        #data = librosa.effects.time_stretch(data, 1.3)
            
        cqt = np.abs(librosa.cqt(y=data, sr=sr))
        mean_size = 20
        height, length = cqt.shape
        new_cqt = np.zeros((height,int(length/mean_size)),dtype=np.float64)
        for i in range(int(length/mean_size)):
            new_cqt[:,i] = cqt[:,i*mean_size:(i+1)*mean_size].mean(axis=1)
        np.save(out_path, new_cqt)
        #print(out_path.split('/')[-1], new_cqt.shape)
        #print(new_cens.shape)
    except :
        print('wa', in_path)
        

with open('data/covers80/coversongs/covers32k/list1.list') as f:
    list1 = [line.strip() for line in f]
with open('data/covers80/coversongs/covers32k/list2.list') as f:
    list2 = [line.strip() for line in f]
in_dir = 'data/covers80/coversongs/covers32k/'

'''    
if not os.path.exists(out_dir):
        os.mkdir(out_dir)
  
params =[]
for ii, (path1, path2) in enumerate(zip(list1, list2)):
    in_path1 = in_dir+path1+'.mp3'
    in_path2 = in_dir+path2+'.mp3'
    out_path1 = out_dir+str(ii)+'_0.npy'
    out_path2 = out_dir+str(ii)+'_1.npy'
    params.append((in_path1, out_path1))
    params.append((in_path2, out_path2))

print('begin')
pool = Pool(30)
pool.map(CQT, params)
pool.close()
pool.join()
'''
import models
opt.model = ''
opt.load_model_path = 'best.pth'
model = getattr(models, opt.model)()
model.load(opt.load_model_path)
model.to(torch.device('cuda'))
model.eval()
class CQTT(Dataset):
    def __init__(self, mode='songs80', out_length=None):
        self.mode=mode
        if mode == 'modify': 
            self.indir = out_dir
            filepath = 'data/songs80_list.txt'
        elif mode == 'songs80': 
            self.indir = 'data/covers80_cqt_npy/'
            filepath = 'data/songs80_list.txt'
        with open(filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]
        self.out_length = out_length
    def __getitem__(self, index):
        
        transform_test = transforms.Compose([
            lambda x : x.T,
            lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
            lambda x : cut_data_front(x, self.out_length),
            lambda x : torch.Tensor(x),
            lambda x : x.permute(1,0).unsqueeze(0),
        ])
        filename = self.file_list[index].strip()
        set_id, version_id = filename.split('.')[0].split('_')
        set_id, version_id = int(set_id), int(version_id)
        in_path = self.indir+filename+'.npy'
        data = np.load(in_path) # from 12xN to Nx12
        data = transform_test(data)
        return data, int(set_id)
    def __len__(self):
        return len(self.file_list)


@torch.no_grad()
def val_slow(model, dataloader1, dataloader2, epoch):
    model.eval()
    labels, features = None, None
    for ii, (data, label) in enumerate(dataloader1):
        input = data.to(torch.device('cuda'))
        #print(input.shape)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels,label))
        else:
            features = feature
            labels = label
    features1 = norm(features)
    
    labels, features = None, None
    for ii, (data, label) in enumerate(dataloader2):
        input = data.to(torch.device('cuda'))
        #print(input.shape)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels,label))
        else:
            features = feature
            labels = label
    features2 = norm(features)
    
    dis2d = -np.matmul(features1, features2.T) # [-1,1] Because normalized, so mutmul is equal to ED
    np.save('dis80.npy',dis2d)
    np.save('label80.npy',labels)
    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels,[100, 350])
    else :
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    print(MAP, top10, rank1 )
    model.train()
    return MAP

val_data1 = CQTT('modify', out_length=None)
val_data2 = CQTT('songs80', out_length=None)
dataloader1 = DataLoader(val_data1, 1, shuffle=False,num_workers=1)
dataloader2 = DataLoader(val_data2, 1, shuffle=False,num_workers=1)

val_slow(model, dataloader1, dataloader2, 0)

print(out_dir)