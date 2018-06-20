# coding=utf-8

import scipy.io as sio
import numpy as np
import random
import os,shutil

base_dir = "/home/zj/Documents/deepfile/BIT-Vehicle/document/"
def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print "%s not exist!"%(srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print "move %s -> %s"%( srcfile,dstfile)

### python 读取.mat文件 ###
load_fn = base_dir + 'VehicleInfo.mat'
load_data = sio.loadmat(load_fn)
print (type(load_data), load_data.keys())

data = load_data['VehicleInfo']

test_index = random.sample(range(data.size), 1500) 


for i in range(data.size):
    item = data[i]

    name = item['name'][0][0]

    #vehicles_type: Bus, Microbus, Minivan, Sedan, SUV, and Truck
    vehicles = item['vehicles']
    height = item['height'][0][0][0]
    width = item['width'][0][0][0]
    left = vehicles[0][0][0][0][0][0]
    top = vehicles[0][0][0][1][0][0]
    right = vehicles[0][0][0][2][0][0]
    bottom = vehicles[0][0][0][3][0][0]
    vehicles_type = vehicles[0][0][0][4][0]

    if(vehicles_type == 'Bus'): vehicles_type = 0
    elif(vehicles_type == 'Microbus'): vehicles_type = 1
    elif(vehicles_type == 'Minivan'): vehicles_type = 2
    elif(vehicles_type == 'Sedan'): vehicles_type = 3
    elif(vehicles_type == 'SUV'): vehicles_type = 4
    elif(vehicles_type == 'Truck'): vehicles_type = 5

    str = '%s %s %s %s %s %s %s %s' %(name, height, width, left, top, right, bottom, vehicles_type)
    print (str)

    if(i in test_index):
        with open('val_full.txt', 'a+') as f:
            f.write(str + '\n')
            
        mymovefile(base_dir + "dataset/" + name, base_dir + "image/val/")
    else:
        with open('train_full.txt', 'a+') as f:
            f.write(str + '\n')
            
        mymovefile(base_dir + "dataset/" + name, base_dir + "image/train/")

            
print ('done--')
