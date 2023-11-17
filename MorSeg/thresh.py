import os
import cv2
import preWithFunction as pre
import segWithFunction as seg
import numpy as np
if __name__ == "__main__":
    base_path = r'../result/kmeans'
    files = os.listdir(base_path)
    files.sort(key=lambda x: int(x.split('.')[0]))
    nums = 1
    for path in files:
        full_path = os.path.join(base_path, path)
        name = os.path.basename(full_path)

        threshold0 = seg.threshold(full_path)
        unique_values = np.unique(threshold0)
        print("Unique pixel values in threshold0:", unique_values)
        inverted_binary_image = 255 - threshold0
        opfile = r'../result/threshold150'
        image_path = opfile + '/' + name
        if (os.path.isdir(opfile) == False):
            os.makedirs(opfile)
        cv2.imwrite(image_path, inverted_binary_image)

        print(name)
        nums += 1
    print('Finish!')