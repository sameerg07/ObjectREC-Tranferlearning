#!/usr/bin/env python3

import cv2
import os

# print("ok")

N = 64
DIRS = ["./train/", "./val/"]

for fol in DIRS:
    for clas in os.listdir(fol):
        clas = os.path.join(fol, clas)
        for img in os.listdir(clas):
            img = os.path.join(clas, img)
            imgMat = cv2.imread(img)

            imgMat = cv2.resize(imgMat, (N, N))

            # cv2.imshow("image", imgMat);
            # cv2.waitKey(30);

            cv2.imwrite(img, imgMat)
