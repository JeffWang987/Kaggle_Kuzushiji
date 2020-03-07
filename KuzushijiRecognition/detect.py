from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import time
import datetime
import argparse


import torch
from torch.utils.data import DataLoader

import VisTools as vis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/sample", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model def file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_6.pth", help="path to weight file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.6, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = input_imgs.type(Tensor)

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            # idx = np.where((detections[0, :, 3].numpy() > 1) & (detections[0, :, 3].numpy() < 80) &
            #                (detections[0, :, 2].numpy() > 1) & (detections[0, :, 2].numpy() < 80)
            #                )
            # detections = detections[0, idx, :]
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    label = detections[0][:, -1].to(torch.float64).numpy()
    label = label[np.newaxis, ...]
    masks = detections[0][:, :4].to(torch.float64).numpy()

    # 现在masks是(xyxy),且没有归一化，我现在要把他变成x_mid,y_mid,w,h,且归一化，为了能用下面的的显示函数
    temp = masks
    temp[:, 0] = (masks[:, 0] + masks[:, 2])/2
    temp[:, 1] = (masks[:, 1] + masks[:, 3])/2
    temp[:, 2] = masks[:, 0] - masks[:, 2]
    temp[:, 3] = masks[:, 1] - masks[:, 3]
    masks = temp[np.newaxis, ...]

    # images是torch类型的，torch.Size([1, 3, 3874, 2404]),并且归一化了
    # 需要转换成numpy，不过注意torch类型由于都存在batch，所以取[0]
    np_img = input_imgs.detach().to('cpu').numpy()[0]
    # viz = vis.visualize_training_data_pytorch(np_img, np.array([[label[0, 0]]]), np.array([[masks[0,0]]])/np_img.shape[1])
    viz = vis.visualize_training_data_pytorch(np_img, label, masks/np_img.shape[1])
    plt.figure(figsize=(15, 15))
    plt.imshow(viz, interpolation='lanczos')
    plt.show()

