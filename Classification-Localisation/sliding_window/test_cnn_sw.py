#!/usr/bin/env python3

import numpy
import os
import cv2
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

loaded_model = load_model(sys.argv[1])

# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

loaded_model.summary()

import numpy as np
import slide

ip = cv2.imread(sys.argv[2])
ip = ip/255.0
imgs = slide.show_window(ip)

# print(next(imgs))

# exit()

mx = 0
mx_clone = ip.copy()
clone = ip.copy()
import time
ax,ay,awx,awy = 100000,100000,0,0
for img,x,y,wx,wy in imgs:
    # img = cv2.imread(sys.argv[1], 0)
    print(img.shape,x,y,wx,wy)

    # preds = loaded_model.predict(np.expand_dims(np.expand_dims(img, axis=0), axis=3))[0]
    preds = loaded_model.predict(np.expand_dims(img, axis=0))[0]
    # input(sum(preds))
    cmx = np.max(preds)
    print(cmx)
    # if cmx > mx:
        # ax,ay,awx,awy = x,y,wx,wy
        # mx = cmx

    if cmx > 0.25:
        print(np.argmax(preds))
        mx_clone = cv2.rectangle(mx_clone, (x, y), (wx, wy), (255, 0, 0), 1)
        if x<ax:
            ax = x
        if y<ay:
            ay = y
        if wx>awx:
            awx = wx
        if wy>awy:
            awy = wy
    c_clone = mx_clone.copy()
    c_clone = cv2.rectangle(c_clone, (x, y), (wx, wy), (0, 255, 0), 2)
    cv2.imshow("Window", c_clone)
    cv2.waitKey(1)
# # mx_clone = cv2.cvtColor(ip.copy(),cv2.COLOR_GRAY2RGB)
# mx_clone = ip.copy()
clone = cv2.rectangle(clone, (ax, ay), (awx, awy), (0, 255, 0), 2)
cv2.imshow("Window", clone)
cv2.waitKey(0)
cv2.destroyAllWindows()
