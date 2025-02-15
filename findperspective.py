import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os

def edgeCanny(imageFile):
    img = cv.imread(imageFile, cv.IMREAD_UNCHANGED)
    edges = cv.Canny(img, 400, 500)
    plt.subplots(1, 2, figsize=(12, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(edges)
    plt.title('Edge Image')
    plt.xticks([])
    plt.yticks([])
    plt.suptitle(imageFile)
    plt.show()
    return edges

edgeImgs = []

for file in os.listdir('3D_Map/'):
    if file.endswith('.png'):
        edgeImgs.append(edgeCanny(file))
#%%
import numpy as np
import math

def show_image(img, title="Image"):
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

def HoughLines(cannyImg, d=20):
    cdst = cv.cvtColor(cannyImg, cv.COLOR_GRAY2BGR)
    lines = cv.HoughLines(cannyImg, 10, np.pi / 180, 500, None, 0, 0)
    #print('lines - ', lines)
    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + d*(-b)), int(y0 + d*(a)))
            pt2 = (int(x0 - d*(-b)), int(y0 - d*(a)))
            cv.line(cdst, pt1, pt2, (255,255,255), 1, cv.LINE_AA)
    show_image(cannyImg, "Canny filtered Image")
    show_image(cdst, "Detected Lines (in blue) - Standard Hough Line Transform")
    # cv.imshow("Canny filtered Image", cannyImg)
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)

for edgeImg in edgeImgs:
    HoughLines(edgeImg)