import cv2
import torch
import numpy as np
import os
from torchvision import transforms as T
from PIL import Image
import pandas as pd
from medpy import metric


def calculte_metric_percase(pre, gt):
    jc = metric.binary.jc(pre, gt)
    dice = metric.binary.dc(pre, gt)
    recall = metric.binary.recall(pre, gt)
    hd = metric.binary.hd95(pre, gt)
    return jc, dice, recall, hd

def getMeric(SR, GT):
    TP = (SR + GT == 2).astype(np.float32)
    TN = (SR + GT == 0).astype(np.float32)
    FP = (SR + (1 - GT) == 2).astype(np.float32)
    FN = ((1 - SR) + GT == 2).astype(np.float32)

    IOU = float(np.sum(TP)) / (float(np.sum(TP + FP + FN)) + 1e-6)

    DSC = float(2*np.sum(TP)) / (float(np.sum(TP + FP + TP + FN) + 1e-6))
    recall = float(np.sum(TP)) / (float(np.sum(TP + FN)) + 1e-6)
    return IOU, DSC, recall

def bootstrap_ci(data, statistic=np.mean, alpha=0.05, num_samples=5000):
    n = len(data)
    rng = np.random.RandomState(47)
    samples = rng.choice(data, size=(num_samples, n), replace=True)
    stat = np.sort(statistic(samples, axis=1))
    lower = stat[int(alpha / 2 * num_samples)]
    upper = stat[int((1 - alpha / 2) * num_samples)]
    return lower, upper

pre_mask_path = r"result/pre-mask"
true_mask_path = r"dataset/gt"
pre_list = [f for f in os.listdir(pre_mask_path) if f.endswith('.png')]
true_list = [f for f in os.listdir(true_mask_path) if f.endswith('.png')]

print(f"pre_list:{pre_list}")
print(f"true_list:{true_list}")
iou_list = []
dsc_list = []
recall_list = []
hd95_list = []
img_iou = pd.DataFrame(columns=["img_name", "pre_gt_iou"])
img_iou.to_csv("./img256_iou.csv", index=False)

for i, mask in enumerate(pre_list):
    print(mask)
    pre_path = os.path.join(pre_mask_path, mask)
    true_path = os.path.join(true_mask_path, mask)

    pre_mask = Image.open(pre_path).convert("1")
    true_mask = Image.open(true_path).convert("1")

    Transform_PD = T.Compose([T.ToTensor()])
    pre_mask = Transform_PD(pre_mask)
    pre_array = (torch.squeeze(pre_mask)).data.cpu().numpy()
    print(f'np.unique(pre_array):{np.unique(pre_array)}')

    Transform_GT = T.Compose([T.ToTensor()])
    true_mask = Transform_GT(true_mask)
    gt_array = (torch.squeeze(true_mask)).data.cpu().numpy()
    print(f'np.unique(gt_array):{np.unique(gt_array)}')

    iou, dsc, recall, hd95 = calculte_metric_percase(pre_array, gt_array)
    iou_list.append(iou)
    dsc_list.append(dsc)
    recall_list.append(recall)
    hd95_list.append(hd95)
    print(f"iou:{iou}")
    print(f"dsc:{dsc}")
    print(f"recall:{recall}")
    print(f"hd95:{hd95}")
    list = [mask, iou]
    data = pd.DataFrame([list])
    data.to_csv("./img256_iou.csv", mode="a", header=False, index=False)

iou_final = np.mean(iou_list)
low_iou,up_iou = bootstrap_ci(iou_list)

dsc_final = np.mean(dsc_list)
low_dsc,up_dsc = bootstrap_ci(dsc_list) 

recall_final = np.mean(recall_list)
low_re,up_re = bootstrap_ci(recall_list)

hd95_final = np.mean(hd95_list)
low_hd,up_hd = bootstrap_ci(hd95_list)

print(f"iou_final:{iou_final}","[",low_iou,up_iou,"]")
print(f"dsc_final:{dsc_final}","[",low_dsc,up_dsc,"]")
print(f"recall_final:{recall_final}","[",low_re,up_re,"]")
print(f"hd95_final:{hd95_final}","[",low_hd,up_hd,"]")
