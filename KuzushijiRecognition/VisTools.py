# -*- coding: utf-8 -*-
# @Time    : 2020/2/16 17:27
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# This function takes in a filename of an image, and the labels in the string format given in train.csv, and returns an
# image containing the bounding boxes and characters annotated
def visualize_training_data(image_fn_, labels_):
    # Convert annotation string to array
    labels_ = np.array(labels_.split(' ')).reshape(-1, 5)

    # Read image
    imsource = Image.open(image_fn_).convert('RGBA')
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)
    bbox_draw = ImageDraw.Draw(
        bbox_canvas)  # Separate canvases for boxes and chars so a box doesn't cut off a character
    char_draw = ImageDraw.Draw(char_canvas)

    for codepoint, x, y, w, h in labels_:
        x, y, w, h = int(x), int(y), int(w), int(h)
        char = _unicode_map[codepoint]  # Convert codepoint to actual unicode character

        # Draw bounding box around character, and unicode character next to it
        bbox_draw.rectangle((x, y, x + w, y + h), fill=(255, 255, 255, 0), outline=(255, 0, 0, 255))
        char_draw.text((x + w + _fontsize / 4, y + h / 2 - _fontsize), char, fill=(0, 0, 255, 255), font=_font)

    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    imsource = imsource.convert("RGB")  # Remove alpha for saving in jpg format.
    return np.asarray(imsource)


# This function takes in a filename of an image, and the labels in the string format given in a submission csv,
# and returns an image with the characters and predictions annotated.
def visualize_predictions(image_fn_, labels_):
    # Convert annotation string to array
    labels_ = np.array(labels_.split(' ')).reshape(-1, 3)

    # Read image
    imsource = Image.open(image_fn_).convert('RGBA')
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)
    bbox_draw = ImageDraw.Draw(
        bbox_canvas)  # Separate canvases for boxes and chars so a box doesn't cut off a character
    char_draw = ImageDraw.Draw(char_canvas)

    for codepoint, x, y in labels_:
        x, y = int(x), int(y)
        char = _unicode_map[codepoint]  # Convert codepoint to actual unicode character

        # Draw bounding box around character, and unicode character next to it
        bbox_draw.rectangle((x - 10, y - 10, x + 10, y + 10), fill=(255, 0, 0, 255))
        char_draw.text((x + 25, y - _fontsize * (3 / 4)), char, fill=(255, 0, 0, 255), font=_font)

    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    imsource = imsource.convert("RGB")  # Remove alpha for saving in jpg format.
    return np.asarray(imsource)


# This function takes in a filename of an image, and the labels in the string format given in train.csv,
# and returns an image containing the bounding boxes and characters annotated
def visualize_training_data_pytorch(image_fn_, label, masks):
    # 把(n_c,W,H)转换为(W,H,n_c),不然PIL无法处理
    image_fn_ = image_fn_.transpose((1, 2, 0))
    # A：Alpha通道一般用做透明度参数
    imsource = Image.fromarray(np.uint8(image_fn_ * 255)).convert('RGBA')
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas)  # Separate canvases for boxes and chars so box doesn't cut off a character
    char_draw = ImageDraw.Draw(char_canvas)
    img_w, img_h, _ = image_fn_.shape
    for idx in range(masks.shape[1]):
        # dataset里面的是(X_mid,Y_mid,W,H),且归一化了
        x, y, w, h = masks[0][idx]
        # 矫正成(X_左上，Y_左上，W，H)，且不归一化
        x = (x - w/2) * img_w
        y = (y - h/2) * img_h
        w = w * img_w
        h = h * img_h
        # 矫正完毕
        x, y, w, h = int(x), int(y), int(w), int(h)
        char = list(_unicode_map.values())[int(label[0][idx])]

        # Draw bounding box around character, and unicode character next to it
        bbox_draw.rectangle((x, y, x+w, y+h),
                            fill=(255, 255, 255, 0),
                            outline=(255, 0, 0, 255))
        char_draw.text((x + w + _fontsize / 4, y + h / 2 - _fontsize), char, fill=(0, 0, 255, 255), font=_font)
    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    imsource = imsource.convert("RGB")  # Remove alpha for saving in jpg format.
    # array仍会copy出一个副本，占用新的内存，但asarray不会。
    return np.asarray(imsource)


# 设置编码
_fontsize = 9
_font = ImageFont.truetype('../data/Kuzushiji/font/NotoSansCJKjp-Regular.otf', _fontsize, encoding='utf-8')
_unicode_map = {codepoint: char for codepoint, char in
                pd.read_csv('../data/Kuzushiji/unicode_translation.csv').values}
if __name__ == '__main__':

    # 加载数据
    df_train = pd.read_csv('../data/Kuzushiji/train.csv')

    # 看train数据集的图片+label
    np.random.seed(1337)
    for i in range(2):
        img, labels = df_train.values[np.random.randint(len(df_train))]
        viz = visualize_training_data('../data/Kuzushiji/img/train_images/{}.jpg'.format(img), labels)
        plt.figure(figsize=(15, 15))
        plt.title(img)
        plt.imshow(viz, interpolation='lanczos')
        plt.show()

    # 预测test数据
    image_fn = r'../data/Kuzushiji/img/test_images/test_0a9b81ce.jpg'
    # Prediction string in submission file format
    pred_string = 'U+306F 1231 1465 U+304C 275 1652 U+3044 1495 1218 U+306F 436 1200 U+304C 800 2000 U+3044 1000 300'
    viz = visualize_predictions(image_fn, pred_string)
    plt.figure(figsize=(15, 15))
    plt.imshow(viz, interpolation='lanczos')
    plt.show()
