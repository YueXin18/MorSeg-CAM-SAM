import argparse
import cv2
import os
import numpy as np
import torch
from torchvision import models
from torch.nn import DataParallel
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./dataset/test',
        help='Input image path')
    parser.add_argument(
        '--result-path',
        type=str,
        default='./cam',
        help='Output image path'
    )
    parser.add_argument(
        '--grayscale-path',
        type=str,
        default='./cam_grayscale',
        help='Output image path'
    )
    parser.add_argument(
        '--heatmap-path',
        type=str,
        default='./cam_heatmap',
        help='Output image path'
    )
    parser.add_argument(
        '--binary-path',
        type=str,
        default='./cam_binary',
        help='Output mask path'
    )
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='layercam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

def resnext101_test(**kwargs):
    model = models.resnet101(pretrained=False, num_classes=2, **kwargs)
    checkpoint = torch.load('model/resNet101_256_lr=0.00005.pth', map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def resnext50_test(**kwargs):
    model = models.resnet101(pretrained=False, num_classes=2, **kwargs)
    checkpoint = torch.load('model/resNet101_256_lr=0.00005.pth', map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def resnet50_test(**kwargs):
    model = models.resnet101(pretrained=False, num_classes=2, **kwargs)
    checkpoint = torch.load('model/resNet101_256_lr=0.00005.pth', map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def resnet101_test(**kwargs):
    model = models.resnet101(pretrained=False, num_classes=2, **kwargs)
    checkpoint = torch.load('model/resNet101_256_lr=0.00005.pth', map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def densenet121_test(**kwargs):
    model = models.DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, num_classes=2, **kwargs)
    checkpoint = torch.load('./model/densenet121_256_normal.pth', map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])
    return model


def resnet152_test(**kwargs):
    model = models.resnet152(pretrained=False, num_classes=2, **kwargs)
    checkpoint = torch.load('./model/resnet152_224.pth', map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])
    return model


"""def res2net101_v1b_test(**kwargs):
    model = res2net_v1b.res2net101_v1b(pretrained=False, num_classes=2, **kwargs)
    checkpoint = torch.load('./model/res2net101_v1b_224.pth', map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])
    return model"""

def find_max_region(binary):
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    print(f"len(contours):{len(contours)}")

    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    max_idx = np.argmax(area)
    max_area = cv2.contourArea(contours[max_idx])
    print(f"area_list:{area}")
    print(f"max_idx:{max_idx}")
    print(f"max_area:{max_area}")

    for j in range(len(contours)):
        if j != max_idx:
            cv2.fillPoly(binary, [contours[j]], (0, 0, 0))

    cv2.fillPoly(binary, [contours[max_idx]], (255, 255, 255))

    return binary


if __name__ == '__main__':
    """ python cam.py -image_448-path <path_to_image>"""

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    model = densenet121_test()
    target_layer_model = [model.features[-1]]
    cate_4 = os.listdir(args.image_path)
    for cate in cate_4:
        print(f'now is: {cate}')
        cate_path = os.path.join(args.image_path, cate)
        res_path = os.path.join(args.result_path, cate)
        bin_path = os.path.join(args.binary_path, cate)
        grayscale_path = os.path.join(args.grayscale_path, cate)

        if not os.path.exists(res_path):
            os.makedirs(res_path)
        if not os.path.exists(bin_path):
            os.makedirs(bin_path)
        if not os.path.exists(grayscale_path):
            os.makedirs(grayscale_path)

        image_list = os.listdir(cate_path)
        image_path = [os.path.join(cate_path, image) for image in image_list]

        for i, img_path in enumerate(image_path):
            img_name0 = img_path.split('/')[-1]
            img_name = os.path.splitext(img_name0)[0]
            print(f"img_name:{img_name}")
            
            rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
            rgb_img = np.ascontiguousarray(rgb_img)
            rgb_img = np.float32(cv2.resize(rgb_img, (256, 256), interpolation=cv2.INTER_AREA)) / 255
            
            input_tensor = preprocess_image(rgb_img,
                                            mean=[0.32825083, 0.32824174, 0.32820907],
                                            std=[0.19632086, 0.1963208, 0.19629629],)
            targets = [ClassifierOutputTarget(int(cate))]
            cam_algorithm = methods[args.method]

            with cam_algorithm(model=model,
                                target_layers=target_layer_model,
                                use_cuda=args.use_cuda) as cam:

                cam.batch_size = 32
                grayscale_cam = cam(input_tensor=input_tensor,
                                    targets=targets,
                                    aug_smooth=args.aug_smooth,
                                    eigen_smooth=args.eigen_smooth)

                grayscale_cam = grayscale_cam[0, :]
                gray_mask = np.uint8(255 * grayscale_cam)

                ret, binary = cv2.threshold(gray_mask, 210, 255, cv2.THRESH_BINARY)
                pre_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                print(f"len(contours):{len(pre_contours)}")
                for i in range(len(pre_contours)):
                    if cv2.contourArea(pre_contours[i]) <= 150:
                        cv2.fillPoly(binary, [pre_contours[i]], (0, 0, 0))

                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            save_cam_path = os.path.join(res_path, img_name + '.png')
            save_bin_path = os.path.join(bin_path, img_name + '.png')
            save_grayscale_path = os.path.join(grayscale_path, img_name + '.png')

            print(f'save_cam_path:{save_cam_path}')
            print(f'save_bin_path:{save_bin_path}')
            print(f'save_grayscale_path:{save_grayscale_path}')

            cv2.imwrite(save_cam_path, cam_image)
            cv2.imwrite(save_bin_path, binary)
            cv2.imwrite(save_grayscale_path, gray_mask)
