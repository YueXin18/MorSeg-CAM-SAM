import os
import cv2
import preWithFunction as pre
import segWithFunction as seg

if __name__ == "__main__":
    base_path =r'../result/ACE'
    files = os.listdir(base_path)
    files.sort(key=lambda x: int(x.split('.')[0]))
    nums = 1
    for path in files:
        full_path = os.path.join(base_path, path)
        name = os.path.basename(full_path)

        cluster0 = seg.clustering(full_path)
        opfile = r'../result/cluster-2'
        image_path = opfile + '/' + name
        if (os.path.isdir(opfile) == False):
            os.makedirs(opfile)
        cv2.imwrite(image_path, cluster0)

        print(name)
        nums += 1
    print('Finish!')