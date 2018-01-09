#!/usr/bin/env python3

import time
import cv2

import sys

# img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
N = 224
wind_row, wind_col = 0, 0

#generating the sliding window
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# print("image shape=",img.shape)

def show_window(img):
    # print(img.shape)
    for winSize in [i for i in range(32, min(img.shape[0], img.shape[1]), 64)]:
    # for winSize in [i for i in range(32+32, 65, 64)]:
        wind_row, wind_col = winSize, winSize

        for(x,y, window) in sliding_window(img, 16, (wind_row,wind_col)):
            if window.shape[0] != wind_row or window.shape[1] != wind_col:
                continue
            # clone = img.copy()
            # clone = cv2.cvtColor(clone,cv2.COLOR_GRAY2RGB)
            cropped = cv2.resize(img[y:y+wind_col, x:x+wind_row], (N, N))

            yield (cropped,x,y,x+wind_row,y+wind_col)
            # cv2.rectangle(clone, (x, y), (x + wind_row, y + wind_col), (0, 255, 0), 2)

            # cv2.imshow("sliding_window", img[y:y+wind_row,x:x+wind_col])
            # cv2.imshow("Window", cropped)
            # cv2.waitKey(1)
            # time.sleep(0.01)

# show_window()
# cv2.waitKey(0)
# cv2.destroyAllWindows()