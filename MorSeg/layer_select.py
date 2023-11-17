import cv2
import numpy as np
import os

def get_img_and_layer():
    img_path = r"./threshold-result"
    layer_mask_path = r"./layer-mask"
    merge_path = r"./layer-select"

    if not os.path.exists(merge_path):
        os.makedirs(merge_path)

    img_path_list = [os.path.join(img_path, image) for image in os.listdir(img_path)]

    for img256_path in img_path_list:
        image_name = img256_path.split("\\")[-1]
        print(image_name)

        img_path = os.path.join(img_path, image_name)
        layer_path = os.path.join(layer_mask_path, image_name)
        image = cv2.imread(img_path, 0)
        layer = cv2.imread(layer_path, 0)

        print(f"image.unique:{np.unique(image)}")
        print(f"layer.unique:{np.unique(layer)}")

        image_layer = cv2.resize(layer, (256, 256), interpolation=cv2.INTER_AREA)

        merge = cv2.bitwise_and(image, image_layer)
        ret, merge_img = cv2.threshold(merge, 127, 255, cv2.THRESH_BINARY)
        merge_contours, _ = cv2.findContours(merge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"len(merge_contours):{len(merge_contours)}")
        for contour in merge_contours:
            if cv2.contourArea(contour) <= 105:
                cv2.fillPoly(merge_img, [contour], (0, 0, 0))

        save_merge_path = os.path.join(merge_path, image_name)
        cv2.imwrite(save_merge_path, merge_img)


if __name__ == "__main__":
    get_img_and_layer()
