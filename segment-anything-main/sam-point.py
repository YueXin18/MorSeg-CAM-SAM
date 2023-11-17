import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import random

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, point_1024,label, H, W):
    point_torch = torch.as_tensor(point_1024, dtype=torch.float, device=img_embed.device)
    label_torch = torch.as_tensor(label, dtype=torch.int, device=img_embed.device)

    point_torch, label_torch = point_torch[None, :, :], label_torch[None, :]
    points_torch = (point_torch, label_torch)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=points_torch,
        boxes=None,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

parser = argparse.ArgumentParser(
    description="run inference on testing set based on SAM"
)

parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="../model/sam_vit_h_4b8939.pth",
    help="path to the trained model",
)
args = parser.parse_args()

device = args.device
medsam_model = sam_model_registry["vit_h"](checkpoint=args.checkpoint)
medsam_model = medsam_model.to(device)
medsam_model.eval()

dataset_path = "../dataset/test"
mask_path = "../result/MorSeg+LCAM"
save_path = "../result/MorSeg+LCAM+SAM-p10"
plt_save_path = "../result/plt-MorSeg+LCAM+SAM-p10"
if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(plt_save_path):
    os.makedirs(plt_save_path)

data_names = os.listdir(dataset_path)
mask_names = [img for img in os.listdir(mask_path) if os.path.splitext(img)[1].lower() == '.png']
for i, img in enumerate(data_names):
    print(f'now is: {img}')
    img_path = os.path.join(dataset_path,img)
    img_save_path = os.path.join(save_path,img)

    mask_img_path = os.path.join(mask_path,img)
    img_save_path = os.path.join(save_path,img)

    pseudo_label_mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
    if pseudo_label_mask.dtype != np.uint8:
        pseudo_label_mask = pseudo_label_mask.astype(np.uint8)
    contours, _ = cv2.findContours(pseudo_label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour[:, 0, :]

    if len(points) >= 10:
        random_indices = random.sample(range(len(points)), 10)
        random_points = points[random_indices]

        for point in random_points:
            x, y = point
            print(f"Random Point: ({x}, {y})")
    else:
        print("Less than 10 points in the outline!")
        random_indices = random.sample(range(len(points)), len(points))
        random_points = points[random_indices]

    img_np = io.imread(img_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape

    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    point_np = np.array(random_points)
    label = [1] * len(random_points)
    input_label = np.array(label)

    point_1024 = point_np / np.array([W, H]) * 1024
    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

    medsam_seg = medsam_inference(medsam_model, image_embedding, point_1024,input_label, H, W)#jjj

    medsam_seg_mask = (medsam_seg * 255).astype(np.uint8)
    medsam_seg_mask = medsam_seg_mask.astype(np.uint8)

    medsam_image = Image.fromarray(medsam_seg_mask)
    medsam_image.save(join(save_path, os.path.basename(img_path)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_3c)
    show_points(point_np, input_label, ax[0])
    ax[0].set_title("Input Image and Point")
    ax[1].imshow(img_3c)
    show_mask(medsam_seg, ax[1])
    show_points(point_np, input_label, ax[1])
    ax[1].set_title("SAM Segmentation")
    plt.show()
    plt.savefig(os.path.join(plt_save_path,os.path.basename(img_path)))





