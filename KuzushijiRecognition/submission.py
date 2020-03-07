# -*- coding: utf-8 -*-
# @Time    : 2020/3/7 10:29
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import time
import datetime
import argparse
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader

import VisTools as vis

_unicode_map = {codepoint: char for codepoint, char in pd.read_csv('../data/Kuzushiji/unicode_translation.csv').values}
_unicode2labels = dict(zip(_unicode_map.keys(), range(len(_unicode_map.keys()))))
_labels2unicode = dict(zip(range(len(_unicode_map.keys())), _unicode_map.keys()))
_unicode2words = dict(zip(range(len(_unicode_map.keys())), _unicode_map.values()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data/sample
    parser.add_argument("--image_folder", type=str, default="../data/Kuzushiji/img/test_images", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model def file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_6.pth", help="path to weight file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # 创建dataframe
    columns = {'labels', 'image_id'}
    index = range(len(dataloader))
    submission_df = pd.DataFrame(index=index, columns=columns)

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_size, img_paths, ori_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = ori_imgs.type(Tensor).to(device)
        ori_max_size = np.maximum(img_size[0].numpy()[0], img_size[1].numpy()[0])
        scale = ori_max_size/opt.img_size
        padx = np.maximum(img_size[1].numpy()[0]-img_size[0].numpy()[0], 0)/2
        pady = np.maximum(img_size[0].numpy()[0]-img_size[1].numpy()[0], 0)/2
        # Get detections
        with torch.no_grad():
            torch.cuda.empty_cache()
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        image_id = img_paths[0].split('\\')[1].split('.')[0]
        try:
            label = detections[0][:, -1].to(torch.float64).cpu().numpy()
            masks = detections[0][:, :4].to(torch.float64).cpu().numpy()
            # 现在masks是(x y x y),且经过pad和scale to 416，我现在要把他变成x_mid,y_mid,w,h,且还原原始大小
            temp = masks
            masks = masks * scale
            temp[:, 0] = ((masks[:, 0] + masks[:, 2]) / 2) - padx
            temp[:, 1] = (masks[:, 1] + masks[:, 3]) / 2 - pady
            temp[:, 2] = masks[:, 0] - masks[:, 2]
            temp[:, 3] = masks[:, 1] - masks[:, 3]
            masks = temp
            labels = ''
            for idx in range(len(label)):
                labels += _labels2unicode[label[idx]]
                labels += ' '
                labels += str(int(masks[idx, 0]))
                labels += ' '
                labels += str(int(masks[idx, 1]))
                labels += ' '
            submission_df['image_id'][batch_i] = image_id
            submission_df['labels'][batch_i] = labels
        except TypeError:
            submission_df['image_id'][batch_i] = image_id

    submission_df.to_csv('submission.csv', index=None)

















