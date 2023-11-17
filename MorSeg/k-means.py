import os
import numpy as np
import cv2
from sklearn.cluster import KMeans

input_folder = r'../result/ACE'
output_folder = r'../result/kmeans'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

num_clusters = 2

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        pixels = img.reshape((-1, 3))
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        segmented_img = centers[labels].reshape((h, w, 3)).astype(np.uint8)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, segmented_img)
        print(filename)

print("Conversion completed.")
