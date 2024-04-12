#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip list


# In[5]:


#未加 from skimage import feature as ft
import os
import cv2
from  tqdm  import  tqdm 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import random

from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
import xgboost
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[6]:


# 图像均衡化
def equalize(image):
    # 分割B,G,R （cv2读取图像的格式即为[B,G,R]，与matplotlib的[R,G,B]不同）
    b,g,r = cv2.split(image)
    # 依次均衡化
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    # 结合成一个图像
    equ_img = cv2.merge((b,g,r))
    
    return equ_img


# In[8]:


#以1_Pre_test-h02060.jpg为例
image=cv2.imread('h02060.jpg')


# In[9]:


equalize(image)


# In[14]:


# 提取图片中绿色（叶子）的部分 ✖️  二值化、过滤滤波
def extractGreen(image):
    # 绿色范围
    lower_green = np.array([0, 0, 0], dtype="uint8")  # 颜色下限
    upper_green = np.array([255, 255, 255], dtype="uint8")  # 颜色上限
    
    # 高斯滤波
    img_blur = cv2.GaussianBlur(image, (3, 3), 0)
    img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    # 根据阈值找到对应颜色，二值化
    mask = cv2.inRange(img_blur, lower_green, upper_green)
    
    # 掩膜函数
    output = cv2.bitwise_and(image, image, mask=mask)

    return output


# In[15]:


extractGreen(image)


# In[16]:


# 提取图像的SIFT特征
def sift_feature(image_list):
    feature_sift_list = []  # SIFT特征向量列表
    
    sift = cv2.xfeatures2d.SIFT_create()
    
    for i in tqdm(range(len(image_list))):
        # 转为灰度图
        image = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2GRAY)
        # 获取SIFT特征，kp为关键点信息，des为关键点特征矩阵形式
        kp, des = sift.detectAndCompute(image, None)
        feature_sift_list.append(des)
        
    return feature_sift_list


# In[ ]:




