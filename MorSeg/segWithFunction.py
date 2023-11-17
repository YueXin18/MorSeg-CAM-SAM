import cv2
import numpy as np
import matplotlib.pyplot as plt

def threshold(readPath):
    img = cv2.imread(readPath)
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    ret, result = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
    return result

def adaptiveThreshold(readPath):
    img = cv2.imread(readPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return result

def edge(readPath):

    img = cv2.imread(readPath)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
    y = cv2.filter2D(binary, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
    y = cv2.filter2D(binary, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    dst = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)

    x = cv2.Scharr(binary, cv2.CV_32F, 1, 0)
    y = cv2.Scharr(binary, cv2.CV_32F, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Scharr = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    gaussianBlur = cv2.GaussianBlur(binary, (3, 3), 0)
    Canny = cv2.Canny(gaussianBlur, 50, 150)

    gaussianBlur = cv2.GaussianBlur(binary, (3, 3), 0)
    dst = cv2.Laplacian(gaussianBlur, cv2.CV_16S, ksize=3)
    LOG = cv2.convertScaleAbs(dst)
    return Roberts, Prewitt, Sobel,Laplacian, Scharr, Canny, LOG

def contour(readPath):
    img = cv2.imread(readPath)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(grayImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    return img

def clustering(readPath):
    img = cv2.imread(readPath)

    data = img.reshape((-1, 3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    flags = cv2.KMEANS_RANDOM_CENTERS

    compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)

    centers2 = np.uint8(centers2)
    res = centers2[labels2.flatten()]
    dst2 = res.reshape((img.shape))

    dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
    return dst2

def watershed(readPath):
    img = cv2.imread(readPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    return img
