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
from hpcp_loader import *
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

def CQT(args):
    try:
        in_path, out_path ,Q = args
        data, sr = librosa.load(in_path)
        if Q:
            #data = librosa.effects.pitch_shift(data, sr, n_steps=-8)
            data = librosa.effects.time_stretch(data, 1.2)
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
        
in_dir = 'data/you350/audio/'
out_dir = 'changed_npy/you350_cqt_npy_time13/'
print(out_dir)
if not os.path.exists(out_dir):
        os.mkdir(out_dir)
params =[]
for ii, (root, dirs, files) in enumerate(os.walk(in_dir)):         
    if ii < 5000 and len(files):
        for file in files:
            in_path = os.path.join(root,file)
            set_id = int(file.split('_')[0])
            ver_id = int(file.split('_')[-1].split('.')[0])
            #print(set_id,ver_id)
            out_path = out_dir + file.split('.')[0] + '.npy'
            Query = False
            if ver_id < 5:
                Query = True
            params.append((in_path, out_path, Query))

'''

print('begin')
pool = Pool(30)
pool.map(CQT, params)
pool.close()
pool.join()
'''

class CQTT(Dataset):
    def __init__(self, mode='songs350', out_length=None):
        self.mode=mode
        if mode == 'songs350': 
            self.indir = out_dir
            filepath='data/you350_list.txt'
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
def val_slow(model, dataloader, epoch):
    model.eval()
    total, correct = 0, 0
    labels, features = None, None

    for ii, (data, label) in enumerate(dataloader):
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
    features = norm(features)
    #dis2d = get_dis2d4(features)
    dis2d = -np.matmul(features, features.T) # [-1,1] Because normalized, so mutmul is equal to ED
    np.save('dis80.npy',dis2d)
    np.save('label80.npy',labels)
    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels,[100, 350])
    else :
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    print(epoch, MAP, top10, rank1 )
    model.train()
    return MAP

import models
opt.model = 'CQTTPPNet'
opt.load_model_path = 'best.pth'
model = getattr(models, opt.model)()
model.load(opt.load_model_path)
model.to(torch.device('cuda'))
model.eval()

val_data350 = CQTT('songs350', out_length=None)
val_dataloader350 = DataLoader(val_data350, 1, shuffle=False,num_workers=1)
val_slow(model, val_dataloader350, 0)

print(out_dir)