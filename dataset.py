#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import read_truths_args, read_truths
from image import *

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, crop=False, jitter=0.3, hue=0.1, saturation=1.5, exposure=1.5,
                 transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4,condition=False):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       # print('Go to list data')
       # print('Condition = ',condition)
       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers

       self.crop = crop
       self.jitter = jitter
       self.hue = hue
       self.saturation = saturation
       self.exposure = exposure

       self.condition = condition
       self.init_shape = shape
       self.cur_shape = shape

    def __len__(self):
        return self.nSamples

    def get_different_scale(self):
        if self.seen < 50*self.batch_size:
            wh = 13*32                          # 416
        elif self.seen < 2000*self.batch_size:
            wh = (random.randint(0,3) + 13)*32  # 416, 480
        elif self.seen < 8000*self.batch_size:
            wh = (random.randint(0,5) + 12)*32  # 384, ..., 544
        elif self.seen < 10000*self.batch_size:
            wh = (random.randint(0,7) + 11)*32  # 352, ..., 576
        else: # self.seen < 20000*self.batch_size:
            wh = (random.randint(0,9) + 10)*32  # 320, ..., 608
        # print('new width and height: %d x %d ' % (wh, wh))
        return (wh, wh)

    def get_different_scale_my(self):
        init_width, init_height = self.init_shape
        init_width = init_width//32
        init_height = init_height // 32

        if self.seen < 100*self.batch_size:
            return self.init_shape
        elif self.seen < 8000*self.batch_size:
            rand_scale = random.randint(0,3)
            init_width = init_width-2 + rand_scale
            init_height = init_height-2 + rand_scale
        elif self.seen < 12000*self.batch_size:
            rand_scale = random.randint(0, 5)
            init_width = init_width - 3 + rand_scale
            init_height = init_height - 3 + rand_scale
        elif self.seen < 16000*self.batch_size:
            rand_scale = random.randint(0, 7)
            init_width = init_width - 4 + rand_scale
            init_height = init_height - 4 + rand_scale
        else:
            rand_scale = random.randint(0, 9)
            init_width = init_width - 5 + rand_scale
            init_height = init_height - 5 + rand_scale
        wid = init_width * 32
        hei = init_height * 32
        return (wid, hei)

    def __getitem__(self, index):
        # print('get item')
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        img_id = os.path.basename(imgpath).split('.')[0]

        if self.train:
            # print(index)
            if (self.seen % (self.batch_size*100)) == 0: # in paper, every 10 batches, but we did every 64 images
                self.shape = self.get_different_scale_my()
                # self.shape = self.get_different_scale()
                # print('Image size: ', self.shape)
                # self.shape = self.get_different_scale()
            img, label = load_data_detection(imgpath, self.shape, self.crop, self.jitter, self.hue, self.saturation, self.exposure)
            label = torch.from_numpy(label)
        else:
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img, org_w, org_h = letterbox_image(img, self.shape[0], self.shape[1]), img.width, img.height
    
            # labpath = imgpath.replace('images', 'labels').replace('images', 'Annotations').replace('.jpg', '.txt').replace('.png','.txt')
            labpath = imgpath.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png','.txt').replace('.tif','.txt')
            label = torch.zeros(50*5)
            #if os.path.getsize(labpath):
            #tmp = torch.from_numpy(np.loadtxt(labpath))
            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
            except Exception:
                tmp = torch.zeros(1,5)
            #tmp = torch.from_numpy(read_truths(labpath))
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            #print('labpath = %s , tsz = %d' % (labpath, tsz))
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers

        # this is for day and night time.
        if self.condition:
            set_label = 0 if int(img_id[4]) < 3 else 1
        else:
            set_label = 0

        if self.train:
            if self.condition:
                return (img, (label, set_label))
            else:
                # print('end function get item')
                return (img, label)
        else:
            return (img, label, org_w, org_h)

