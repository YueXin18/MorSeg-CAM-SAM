import os
import cv2
import preWithFunction as pre
import segWithFunction as seg

if __name__ == "__main__":
    base_path = r"../data"
    files = os.listdir(base_path)
    nums = 1
    for path in files:
        full_path = os.path.join(base_path, path)
        name = os.path.basename(full_path)

        pretreatment = pre.pretreatmentWithACE(full_path)
        opfile = r'../result/ACE'
        image_path = opfile + '/' + name
        if (os.path.isdir(opfile) == False):
            os.makedirs(opfile)
        cv2.imwrite(image_path, pretreatment)

        print(name)
        nums += 1
    print('Finish!')