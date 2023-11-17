import os
import cv2
import matplotlib.pyplot as plt


image_path = r"../data"
save_path = r"../result/mid"
save_hist_path = r"../result/e_hist"
save_ahist_path = r"../result/e_ahist"
save_lhist_path = r"../result/e_lhist"

image_list = os.listdir(image_path)
print(f"pre_list:{image_path}")


for i, image in enumerate(image_list):
    img_path = os.path.join(image_path, image)
    src = cv2.imread(img_path)
    img = src.copy()
    img_median = cv2.medianBlur(img, 5)
    cv2.imwrite(os.path.join(save_path, image), img_median)

    img_gray = cv2.cvtColor(img_median, cv2.COLOR_BGR2GRAY)
    hist0 = cv2.calcHist([img_gray], [0], None, [256], [0, 255])
    plt.plot(hist0, label="Histogram of grayscale maps", linestyle='--', color='g')
    plt.savefig(os.path.join(save_hist_path, image))

    img_all_aug = cv2.equalizeHist(img_gray)
    cv2.imwrite(os.path.join(save_ahist_path, image), img_all_aug)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    img_local_aug = clahe.apply(img_gray)
    cv2.imwrite(os.path.join(save_lhist_path, image), img_local_aug)
