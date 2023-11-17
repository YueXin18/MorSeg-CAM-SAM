import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import time

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


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
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
save_path = "../result/MorSeg+LCAM+SAM-box"
plt_save_path = "../result/plt-MorSeg+LCAM+SAM-box"
if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(plt_save_path):
    os.makedirs(plt_save_path)

data_names = os.listdir(dataset_path)
mask_names = os.listdir(mask_path)

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
    if len(contours) != 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
    else:
        x = 0
        y = 0
        w = 0
        h = 0

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

    box=[]
    box.append(x)
    box.append(y)
    box.append(x+w)
    box.append(y+h)

    box_np = np.array([box])
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

    medsam_seg_mask = (medsam_seg * 255).astype(np.uint8)
    medsam_seg_mask = medsam_seg_mask.astype(np.uint8)

    medsam_image = Image.fromarray(medsam_seg_mask)
    medsam_image.save(join(save_path, os.path.basename(img_path)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_3c)
    show_box(box_np[0], ax[0])
    ax[0].set_title("Input Image and Bounding Box")
    ax[1].imshow(img_3c)
    show_mask(medsam_seg, ax[1])
    show_box(box_np[0], ax[1])
    ax[1].set_title("SAM Segmentation")
    plt.show()
    plt.savefig(os.path.join(plt_save_path,os.path.basename(img_path)))




