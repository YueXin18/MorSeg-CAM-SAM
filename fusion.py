import numpy as np
import cv2
import os

def select_best(cam_pre, tra_pre):
    size = 256
    max_intersection = 0.0
    max_tra_contour = np.zeros((size, size), np.uint8)
    tra_contours, hierarchy = cv2.findContours(tra_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(tra_contours)):
        black_img = np.zeros((size, size), np.uint8)
        cv2.fillPoly(black_img, [tra_contours[i]], (255, 255, 255))

        merge = cv2.bitwise_and(cam_pre, black_img)
        pre_contours, hierarchy = cv2.findContours(merge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"cam and tra merge len(pre_contours):{len(pre_contours)}")
        for i in range(len(pre_contours)):
            area = cv2.contourArea(pre_contours[i])
            if area > max_intersection:
                max_intersection = area
                max_tra_contour = black_img
                print(f"now max_intersection is:{max_intersection}")

    print(f"np.unique.max_tra_contour:{np.unique(max_tra_contour)}")
    if cv2.countNonZero(max_tra_contour) == 0:
        max_tra_contour = cam_pre

    return max_tra_contour


def merge_tra_and_cam():
    mask_cam = r"./cam-result"
    mask_mor = r"./mor-result"
    merge_mask_path = r"./mor-cam"

    if not os.path.exists(merge_mask_path):
        os.makedirs(merge_mask_path)

    img_path_list = [os.path.join(mask_cam, image) for image in os.listdir(mask_cam)]

    for img256_path in img_path_list:
        image_name = img256_path.split("\\")[-1]
        print(image_name)

        img_tra_path = os.path.join(mask_mor, image_name)

        image_256 = cv2.imread(img256_path, 0)
        image_tra = cv2.imread(img_tra_path, 0)

        image_tra = cv2.resize(image_tra, (256, 256), interpolation=cv2.INTER_AREA)
        ret, binary = cv2.threshold(image_tra, 127, 255, cv2.THRESH_BINARY)
        pre_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"image_tra's len(contours):{len(pre_contours)}")
        for contour in pre_contours:
            cv2.fillPoly(binary, [contour], (255, 255, 255))

        pre_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(pre_contours)):
            if cv2.contourArea(pre_contours[i]) <= 100:
                cv2.fillPoly(binary, [pre_contours[i]], (0, 0, 0))

        print(f"image_256.unique:{np.unique(image_256)}")
        print(f"image_tra.unique:{np.unique(binary)}")

        merge = select_best(image_256, binary)
        print(f"merge.unique:{np.unique(merge)}")
        ret, merge_img = cv2.threshold(merge, 127, 255, cv2.THRESH_BINARY)
        merge_contours, _ = cv2.findContours(merge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"len(merge_contours):{len(merge_contours)}")
        for contour in merge_contours:
            cv2.fillPoly(merge_img, [contour], (255, 255, 255))

        merge_contours, _ = cv2.findContours(merge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"len(merge_contours):{len(merge_contours)}")
        for contour in merge_contours:
            if cv2.contourArea(contour) <= 110:
                cv2.fillPoly(merge_img, [contour], (0, 0, 0))

        save_merge_path = os.path.join(merge_mask_path, image_name)
        cv2.imwrite(save_merge_path, merge_img)

if __name__ == "__main__":
    merge_tra_and_cam()
