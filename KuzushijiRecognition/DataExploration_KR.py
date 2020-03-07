# -*- coding: utf-8 -*-
# @Time    : 2020/2/18 14:38
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import ImageFont

_fontsize = 50
_font = ImageFont.truetype('../data/Kuzushiji/font/NotoSansCJKjp-Regular.otf', _fontsize, encoding='utf-8')
_unicode_map = {codepoint: char for codepoint, char in pd.read_csv('../data/Kuzushiji/unicode_translation.csv').values}
_unicode2labels = dict(zip(_unicode_map.keys(), range(len(_unicode_map.keys()))))
_unicode2words = dict(zip(range(len(_unicode_map.keys())), _unicode_map.values()))
# 字典库里有4781个单词
# print(len(_unicode_map))


df_train = pd.read_csv('../data/Kuzushiji/train_1.csv')
# label里面东西的顺序是：label，x，y，w，h
labels = df_train['labels']
id = df_train['image_id']
img_size = df_train['img_size']
# 在训练集中出现过的字，可重复
wordsmap = []
# 训练集中出现的字的字符编码对应一个数字
labelsmap = []
# 训练集中出现的字符编码但是字典中没有的
keyerr = []
# 训练集里面的bbox的w和h
mask_wh_map = np.ndarray(shape=[labels.shape[0], 2])
for idx in range(len(labels)):
    label = labels[idx].split(' ')
    words = len(label)//5
    # 由于有点操作小失误，所以下面一句需要麻烦点，把str转化成我们需要的数字元素
    scale = np.max([int(img_size[idx].split('(')[1].split(',')[0]), int(img_size[idx].split('(')[1].split(',')[1].split(')')[0])])
    for idx_word in range(words):
        wordsmap.append(label[idx_word*5])
        mask_wh_map[idx] = int(label[3+idx_word*5])/scale, int(label[4+idx_word*5])/scale
        try:
            labelsmap.append(_unicode2labels[label[idx_word*5]])
        except KeyError:
            keyerr.append(label[idx_word*5])
# 训练集中出现过的字，去掉重复的
words_unique = np.unique(wordsmap)
# 训练集中不重复的字只有4212个，离字典还缺500多个
print(len(words_unique))

# 通过hist可以看出有的字符使用次数好多，有的很少，相差悬殊
plt.hist(labelsmap, bins=50)
plt.show()

# 看训练集中存在的label，但是字典却不存在
print(np.unique(keyerr))

# 通过scatter查看bbox中w和h的情况
plt.scatter(mask_wh_map[:, 0], mask_wh_map[:, 1])
plt.show()

# 利用kmeans聚类合适的anchor size
# 原本是归一化的，现在要用yolov3使用的图片尺寸416
mask_wh_map = mask_wh_map * 416
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=12, random_state=9)
y_pred = clf.fit_predict(mask_wh_map)
plt.scatter(mask_wh_map[:, 0], mask_wh_map[:, 1], c=y_pred)
plt.show()
# 为YOLOv3设计的9个新的anchor
print(clf.cluster_centers_)
'''
配置文件里的anchor是先w后h，我们这里也是，所以不用改了
[[10.37558548 27.05650205]
 [10.9081477  49.8007071 ]
 [ 9.89446756 16.41377075]
 [43.01325541 56.21417704]
 [13.40842797 22.1770397 ]
 [ 4.33335393 31.81971901]
 [ 4.04522966 20.78887661]
 [10.45961958 95.53746011]
 [19.43751318 33.53579316]]
'''
