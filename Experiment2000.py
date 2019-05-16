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

in_dir = 'data/songs2000/'
out_dir = 'changed_npy/2000_cqt_npy_time11/'
if not os.path.exists(out_dir):
        os.mkdir(out_dir)
print(out_dir)
def CQT(args):
    try:
        num, in_path, out_path = args
        data, sr = librosa.load(in_path)
        #data = librosa.effects.pitch_shift(data, sr, n_steps=-12)
        data = librosa.effects.time_stretch(data, 1.1)
        cqt = np.abs(librosa.cqt(y=data, sr=sr))
        mean_size = 20
        height, length = cqt.shape
        new_cqt = np.zeros((height,int(length/mean_size)),dtype=np.float64)
        for i in range(int(length/mean_size)):
            new_cqt[:,i] = cqt[:,i*mean_size:(i+1)*mean_size].mean(axis=1)
        np.save(out_path, new_cqt)
        #print(num, out_path, new_cqt.shape)
        #print(new_cens.shape)
    except :
        print('wa', in_path)

params =[]
num = 0
for ii, (root, dirs, files) in enumerate(os.walk(in_dir)):         
    for file in files:
        in_path = os.path.join(root,file)
        out_path = out_dir + file.split('.')[0] + '.npy'
        if not os.path.exists(out_path):
            params.append((num, in_path, out_path))
        num+=1
print(len(params))
print('begin')
pool = Pool(30)
pool.map(CQT, params)
pool.close()
pool.join()
print(out_dir)


filepath = 'hpcp/songs2000_list.txt'
with open(filepath, 'r') as fp:
    file_list = [line.rstrip() for line in fp]
not_exist = []
for file in file_list:
    path = out_dir + file.rstrip() + '.npy'
    if os.path.isfile(path) is False:
        not_exist.append(file)
        
print(not_exist)        
if not_exist==[]:
    import models
    #os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    opt.model = 'CQTTPPNet'
    opt.load_model_path = 'best.pth'
    model = getattr(models, opt.model)()
    model.load(opt.load_model_path)
    model.to(torch.device('cuda'))
    model.eval()
    print('')
    class CQTT(Dataset):
        def __init__(self, mode='test', out_length=None,out_dir=out_dir):
            self.mode=mode
            if mode == 'modify': 
                self.indir = out_dir
                filepath = 'data/songs2000_list.txt'
            elif mode == 'test': 
                self.indir = 'data/songs2000_cqt_npy/'
                filepath = 'data/songs2000_list.txt'
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
    def val_slow(model, dataloader1, dataloader2):
        model.eval()
        labels, features = None, None
        for ii, (data, label) in tqdm(enumerate(dataloader1)):
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
        for ii, (data, label) in tqdm(enumerate(dataloader2)):
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

    val_data1 = CQTT('modify', out_length=None,out_dir=out_dir)
    val_data2 = CQTT('test', out_length=None,out_dir=out_dir)
    dataloader1 = DataLoader(val_data1, 1, shuffle=False,num_workers=1)
    dataloader2 = DataLoader(val_data2, 1, shuffle=False,num_workers=1)
    val_slow(model, dataloader1, dataloader2)
