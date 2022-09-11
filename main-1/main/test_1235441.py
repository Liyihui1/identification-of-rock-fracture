import os
import cv2 as cv

image_path = 'data/xunlian/train_seg'

for file in os.listdir(image_path):
    name = file.split(sep='_')
    newname = os.path.join(name.split('_')[0] + '.png')
    scr = cv.imread(image_path + file)
    cv.imwrite(newname, scr)
