# -*- coding: utf-8 -*-
# @Time    : 2020/2/16 18:59
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import VisTools as vis
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
from PIL import Image, ImageFont
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=(size, size), mode="nearest").squeeze(0)
    return image


class KuzushijiDataset(Dataset):
    # TODO:今日任务，把这个类给
    def __init__(self, img_path, mode="train", multiscale=True):
        self.img_path = img_path
        self.input_path = '../data/Kuzushiji'
        self.img_paths = list(glob.glob(f"{self.img_path}/*.jpg"))
        self.unicode_trans = pd.read_csv(path.join(self.input_path, 'unicode_translation.csv'))
        self.unicode_map = {codepoint: char for codepoint, char in self.unicode_trans.values}
        self.unicode2labels = dict(zip(self.unicode_map.keys(), range(len(self.unicode_map.keys()))))
        self.label_length = 5
        self.transform = transforms.ToTensor()
        self.img_size = 416
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        if mode == "train":
            self.mode = "train"
            self.mask = pd.read_csv(path.join(self.input_path, 'train.csv'))
        else:
            self.mode = "test"

    def get_label_and_mask(self, image_id):
        # assert type(image_id) == str
        keyerr = ['U+4FA1', 'U+5039', 'U+515A', 'U+5E81', 'U+770C', 'U+7A83']
        split_labels = self.mask[self.mask["image_id"] == image_id]["labels"].str.split(" ").values[0]
        word_num = len(split_labels) // self.label_length
        masks_ = np.zeros((word_num, 4))
        labels = np.zeros(word_num)
        for idx in range(word_num):
            start_idx = idx * self.label_length
            if split_labels[start_idx] not in keyerr:
                # 把编码装换成日语label
                labels[idx] = self.unicode2labels[split_labels[start_idx]]
                # bbox位置
                masks_[idx] = split_labels[start_idx + 1:start_idx + self.label_length]
        # 注意我们的label是没有归一化的
        return labels, masks_

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        img_path = self.img_paths[index]
        # PIL.Image.open打开的图像是RGB，而cv2.imread打开的是BRG
        # 经过ToTensor之后被归一化了
        image = self.transform(Image.open(img_path))

        # Pad to square resolution,且在这个里面把image转化为tensor了
        image, pad = pad_to_square(image, 0)
        padded_h, padded_w = image.shape[1:]

        if self.mode == "train":
            # label是类别，mask是bbox位置
            labels_, masks_ = self.get_label_and_mask(img_path.split('images\\')[1].split('.')[0])
            # mask:(x左上角，y左上角，W,H)
            # 给mask加上pad之后的变化,并且把坐标变成(X左上，X右下，Y左上，Y右下)
            x1 = masks_[:, 0] + pad[0]
            x2 = masks_[:, 0] + masks_[:, 2] + pad[1]
            y1 = masks_[:, 1] + pad[2]
            y2 = masks_[:, 1] + masks_[:, 3] + pad[3]
            # 归一化坐标,且变成（X_mid,Y_mid,W,H）
            masks_[:, 0] = ((x1 + x2) / 2) / padded_w
            masks_[:, 1] = ((y1 + y2) / 2) / padded_h
            masks_[:, 2] *= 1 / padded_w
            masks_[:, 3] *= 1 / padded_h

            # 为了和YOLOV3代码看齐，这里把返回值整合成target
            targets = np.zeros((len(masks_), 6))
            targets[:, 2:] = masks_
            targets[:, 1] = labels_
            return image, self.transform(targets)
        else:
            return image

    def collate_fn(self, batch):
        # 通过解压获得的target是个tuple，tuple里面含有batch个元素，每个元素都是tensor，shape=（1，num of bbox， 6）
        imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        # for i, boxes in enumerate(targets):
        #     boxes[:, 0] = i
        for i in range(len(targets)):
            targets[i][0][:, 0] = i
        # 我们知道每张图片的labels个数不一样，现在通过cat函数把他们在dim=1上堆叠起来成一个数组
        # 合并完的shape是(1, num of bbox, 6)
        targets = torch.cat(targets, 1)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = np.random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return imgs, targets

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.img_paths)


if __name__ == '__main__':
    # 设置路径
    input_path = '../data/Kuzushiji'
    font_path = 'font/NotoSansCJKjp-Regular.otf'
    train_image_path = path.join(input_path, 'img', "train_images")
    test_image_path = path.join(input_path, 'img', "test_images")

    # 设置编码
    fontsize = 50
    font = ImageFont.truetype(path.join(input_path, font_path), fontsize, encoding='utf-8')

    # 加载数据
    df_train = pd.read_csv('../data/Kuzushiji/train.csv')

    # Use the torch dataloader to iterate through the dataset
    k_train = KuzushijiDataset(img_path=train_image_path)
    # print(len(k_train))
    # loader = DataLoader(k_train, batch_size=1, shuffle=False, num_workers=0)
    loader = DataLoader(
        k_train,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=k_train.collate_fn,
    )
    # 显示图片
    # dataiter = iter(loader)
    # images, targets = dataiter.next()
    for batch_i, (images, targets) in enumerate(loader):
        images = images
        targets = targets
        break
    label = targets[..., 1].numpy()
    masks = targets[..., 2:].numpy()
    # images是torch类型的，torch.Size([1, 3, 3874, 2404]),并且归一化了
    # 需要转换成numpy，不过注意torch类型由于都存在batch，所以取[0]
    np_img = images.numpy()[0]
    viz = vis.visualize_training_data_pytorch(np_img, label, masks)
    # viz = vis.visualize_training_data_pytorch(np_img, np.array([[label[0, 1]]]), np.array([[masks[0, 1]]]))
    plt.figure(figsize=(15, 15))
    plt.imshow(viz, interpolation='lanczos')
    plt.show()
