import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import os
import shutil

corlor = np.array([[0, 0, 0], [255, 0, 0]])
# corlor = np.array([[0, 0, 0],
#                                  [128, 0, 0],
#                                  [0, 128, 0],
#                                  [128, 128, 0],
#                                  [0, 0, 128],
#                                  [128, 0, 128],
#                                  [0, 128, 128],
#                                  [128, 128, 128],
#                                  [64, 0, 0],
#                                  [192, 0, 0],
#                                  [64, 128, 0],
#                                  [192, 128, 0],
#                                  [64, 0, 128],
#                                  [192, 0, 128],
#                                  [64, 128, 128],
#                                  [192, 128, 128],
#                                  [0, 64, 0],
#                                  [128, 64, 0],
#                                  [0, 192, 0],
#                                  [128, 192, 0],
#                                  [0, 64, 128]
#                                  ], dtype='uint8')
print(corlor.shape)
# print(corlor[3][1])
tpath = "data/xunlian/train/train_lie"
vpath = "data/xunlian/imagesyingyong2"
ts = "data/xunlian/train_seg"
vs = "data/xunlian/imagesyingyongseg"
train_labels = os.listdir(tpath)
print(train_labels)
val_labels = os.listdir(vpath)
print(train_labels)
count1 = 0
count2 = 0
for label in train_labels:
    labelpath = os.path.join(tpath, label)
    # print(labelpath)
    annotation = cv2.imread(labelpath)  # 根据自己的实际路径更改路径名
    # print(annotation.shape)
    temp = np.zeros([annotation.shape[0], annotation.shape[1]])
    annotation = annotation[:, :, [2, 0, 1]]
    for i in range(annotation.shape[0]):
        for j in range(annotation.shape[1]):
            for k in range(corlor.shape[0]):
                if (annotation[i][j] == corlor[k]).all():
                    temp[i][j] = k
    # labelgray = Image.fromarray(temp)
    # labelgray.save(os.path.join(vs, label))
    newlabel = os.path.join(ts, label.split("_")[0] + ".png")
    cv2.imwrite(newlabel, temp)
    count1 += 1
    print("已处理%s张训练标签" % count1)
for label in val_labels:
    labelpath = os.path.join(vpath, label)
    # annotation = np.array(Image.open(labelpath))
    annotation = cv2.imread(labelpath)  # 根据自己的实际路径更改路径名
    # 根据自己的实际路径更改路径名
    temp = annotation[:, :, 0]
    for i in range(annotation.shape[0]):
        for j in range(annotation.shape[1]):
            for k in range(0, corlor.shape[0]):
                if (annotation[i][j] == corlor[k]).all():
                    temp[i][j] = k
    # labelgray = Image.fromarray(temp)
    # labelgray.save(os.path.join(vs, label))
    newlabel = label.split("_")[0] + "_" + label.split("_")[1] + ".png"
    cv2.imwrite(os.path.join(vs, newlabel), temp)
    count2 += 1
    print("已处理%s张验证集标签" % count2)

# seglist = os.listdir(ts)
# for seg in seglist:
#     path = os.path.join(ts, seg)
#     newpath = os.path.join(ts, seg.split('_')[0]+'_'+seg.split('_')[1] + '.png')
#     img = cv2.imread(path, 0)
#     cv2.imwrite(newpath, img)
#     os.remove(path)
