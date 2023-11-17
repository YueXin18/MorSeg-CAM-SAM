import cv2
import numpy as np
import os

def main():

    image_folder = r"./result"
    output_folder = r"./post_result"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            output_path = os.path.join(output_folder, filename)

            img = cv2.imread(os.path.join(image_folder, filename), 0)
            mask = 255 - img

            marker = np.zeros_like(img)
            marker[0, :] = 255
            marker[-1, :] = 255
            marker[:, 0] = 255
            marker[:, -1] = 255

            SE = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(3, 3))
            count = 0
            while True:
                count += 1
                marker_pre = marker
                dilation = cv2.dilate(marker, kernel=SE)
                marker = np.min((dilation, mask), axis=0)

                if (marker_pre == marker).all():
                    break

            dst = 255 - marker
            cv2.imwrite(output_path, dst)
    print("Finished!")


if __name__ == "__main__":
    main()