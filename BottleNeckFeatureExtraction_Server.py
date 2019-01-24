#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 23:23:00 2018

@author: qingguo
"""
import numpy as np
import scipy.misc
import torch
from torch import nn
from torch.autograd import Variable
import time


import torchvision.models as models

import torchvision.transforms as transforms


from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from PIL import Image

spark = SparkSession.builder.appName('X14').getOrCreate()

model = models.resnet50()
model = nn.Sequential(*list(model.children())[:-1])
model.load_state_dict(torch.load('resnet50.pth'))
model.eval()

classes = ('Hernia', 'Pneumonia', 
               'Fibrosis', 'Edema', 'Emphysema', 'Cardiomegaly',  
               'Pleural_Thickening',  'Consolidation',  'Pneumothorax',  
               'Mass',  'Nodule',  'Atelectasis', 'Effusion', 'Infiltration')
def convert_to_binary_array(str_label,n=14):
    if 'No Finding' in str_label:
        return [0]*n
    return [int(l in str_label) for l in classes]
    
def bottleneck_feature_extration(spark, \
                                 model, \
                                 path_to_meta="Data_Entry_2017.csv", \
                                 images_folder = "images", \
                                 target_size=224, \
                                 seed = 2018, \
                                 split_ratio = [7,1,2]):
    data_entry = spark.read.csv(path_to_meta,inferSchema=True,header=True)
    image_label_pid = data_entry.select(col("Image Index").alias("index"),col("Finding Labels").alias("label"),col("Patient ID").alias("pid"),col("Patient Age").alias("age"),col("Patient Gender").alias("gender") )
    
    image_labelArray = image_label_pid.filter(image_label_pid.age < 123).rdd.map(lambda x:(images_folder + "/" + x[0],np.array([x[3]/100] + [1 if x[4] == 'F' else 0] + convert_to_binary_array(x[1]))))
    train,valid,test = image_labelArray.randomSplit(split_ratio, seed)
    train.cache()
    valid.cache()
    test.cache()
    train.map(lambda x: x[0]+','+"".join(list(map(str,x[1].tolist())))).saveAsTextFile("outputs/train_dataset_info"+time.strftime("%Y%m%d-%H%M%S"))
    valid.map(lambda x: x[0]+','+"".join(list(map(str,x[1].tolist())))).saveAsTextFile("outputs/valid_dataset_info"+time.strftime("%Y%m%d-%H%M%S"))
    test.map(lambda x: x[0]+','+"".join(list(map(str,x[1].tolist())))).saveAsTextFile("outputs/test_dataset_info"+time.strftime("%Y%m%d-%H%M%S"))
        
    
    train_transform = transforms.Compose([
        transforms.Scale(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    train.map(lambda x: ",".join(list(map(str,np.append(model(Variable(train_transform(Image.open(x[0]).convert('RGB')).unsqueeze(0))).data.cpu().numpy().ravel(),x[1]).tolist())))).saveAsTextFile("outputs/train_dataset"+time.strftime("%Y%m%d-%H%M%S"))
    valid.map(lambda x: ",".join(list(map(str,np.append(model(Variable(test_transform(Image.open(x[0]).convert('RGB')).unsqueeze(0))).data.cpu().numpy().ravel(),x[1]).tolist())))).saveAsTextFile("outputs/valid_dataset"+time.strftime("%Y%m%d-%H%M%S"))
    test.map(lambda x: ",".join(list(map(str,np.append(model(Variable(test_transform(Image.open(x[0]).convert('RGB')).unsqueeze(0))).data.cpu().numpy().ravel(),x[1]).tolist())))).saveAsTextFile("outputs/test_dataset"+time.strftime("%Y%m%d-%H%M%S"))
            
    return 
    
bottleneck_feature_extration(spark, model,path_to_meta="input/Data_Entry_2017.csv",images_folder = "images")
