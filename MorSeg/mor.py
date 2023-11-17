import cv2
import os
import numpy as np
import math

if __name__ == "__main__":
    path = r'./result/Layer_select'
    save_path = r"./result/Layer_select-MOR"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_list = os.listdir(path)

    for image in img_list:
        print(f"image:{image}")
        img_path = os.path.join(path, image)
        img_save = os.path.join(save_path, image)
        img = cv2.imread(img_path, 0)
        img_new = img.copy()

        print(f"image.unique:{np.unique(img_new)}")
        ret, binary_img = cv2.threshold(img_new, 127, 255, cv2.THRESH_BINARY)
        img_contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print(f"len(merge_contours):{len(img_contours)}")
        if len(img_contours) > 1:
            for contour in img_contours:
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                long = np.max(rect[1])
                short = np.min(rect[1])
                rate = long/short
                print(f"long:{long}")
                print(f"short:{short}")
                print(f"rate:{rate}")

                if rate >= 2.5:
                    cv2.fillPoly(binary_img, [contour], (0, 0, 0))

        cv2.imwrite(img_save, binary_img)
