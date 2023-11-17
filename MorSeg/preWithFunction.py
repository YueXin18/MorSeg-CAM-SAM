import cv2
import numpy as np
import math


def stretchImage(data, s=0.005, bins=2000):
    ht = np.histogram(data, bins);
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0;
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)

g_para = {}

def getPara(radius=5):
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h ** 2 + w ** 2)
    m /= m.sum()
    g_para[radius] = m
    return m


def automaticColorEqualization(I, ratio=4, radius=300):
    para = getPara(radius)
    height, width = I.shape
    zh, zw = [0]*radius + list(range(height)) + [height-1]*radius, [0]*radius + list(range(width)) + [width -1]*radius
    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
    return res


def automaticColorEqualization_Fast(I, ratio, radius):
    height, width = I.shape[:2]
    if min(height, width) <= 2:
        return np.zeros(I.shape) + 0.5
    Rs = cv2.resize(I, ((width+1)//2, (height+1)//2))
    Rf = automaticColorEqualization_Fast(Rs, ratio, radius)
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))

    return Rf + automaticColorEqualization(I, ratio, radius) - automaticColorEqualization(Rs, ratio, radius)


def automaticColorEqualization_RGB(img, ratio=3, radius=3):
    res = np.zeros(img.shape)
    for k in range(3):
        res[:, :, k] = stretchImage(automaticColorEqualization_Fast(img[:, :, k], ratio, radius))
    return res

def show_img(win_name,img,wait_time=0,img_ratio=0.5,is_show=True):
    if is_show is not True:
        return
    rows = img.shape[0]
    cols = img.shape[1]
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL )
    cv2.resizeWindow(win_name,(int(cols*img_ratio),int(rows*img_ratio)))
    cv2.imshow(win_name,img)
    if wait_time >= 0:
        cv2.waitKey(wait_time)

def pretreatmentWithACE(readPath):
    img = automaticColorEqualization_RGB(cv2.imread(readPath) / 448.0) * 448
    return img