
import cv2
import os
tpath = "data/xunlian/train/train_yuan"
ts = "data/xunlian/train/train_yuan"
train_labels = os.listdir(tpath)
count1 = 0
for label in train_labels:
    labelpath = os.path.join(tpath, label)
    # print(labelpath)
    annotation = cv2.imread(labelpath)# 根据自己的实际路径更改路径名
    newlabel = label.split(".")[0] + ".jpg"
    os.remove(labelpath)
    cv2.imwrite(os.path.join(ts, newlabel),  annotation)
    count1 +=1
    print("已处理%s张训练标签"%count1)