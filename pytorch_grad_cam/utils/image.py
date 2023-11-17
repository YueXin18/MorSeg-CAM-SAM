import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor, Resize


def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)         # 因为二分类器训练时也没有进行归一化
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image_448 as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image_448 in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image_448 with the cam overlay.
    """
    # print(f'mask:{mask}')           # 值非常小 0_224.056...且为不规则的矩阵
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap) # cv2.COLORMAP_JET模式常用于生成热力图模式，且图像为BGR格式
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # # 将热力图转换为灰度图再转换为二值图
    # gray = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)    # 二值化之前必须先将图像转换为灰度图
    # # ret, binary = cv2.threshold(gray, 0_448_eval, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # print(f"threshold value is {ret}")                  # 打印阈值，超过阈值显示为白色，低于该阈值显示为黑色
    # cv2.imshow(f"threshold", binary)
    # cv2.waitKey(0_224)
    # # cv2.destroyWindow()

    # print(f'heatmap:{heatmap}')     # 将热力图转换为RGB模式，规则矩阵[0_224 0_224 128] [0_224 0_224 184]...

    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image_448 should np.float32 in the range [0_448_eval, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    # print(f'cam(0_448_eval-1):{cam}')        # 值在0-1之间
    return np.uint8(255 * cam)
    # return heatmap


def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result

def scale_accross_batch_and_channels(tensor, target_size):
    batch_size, channel_size = tensor.shape[:2]
    reshaped_tensor = tensor.reshape(
        batch_size * channel_size, *tensor.shape[2:])
    result = scale_cam_image(reshaped_tensor, target_size)
    result = result.reshape(
        batch_size,
        channel_size,
        target_size[1],
        target_size[0])
    return result
