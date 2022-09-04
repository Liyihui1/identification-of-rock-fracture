import cv2
import numpy as np
import torch
import os
import shutil
# print(torch.cuda.current_device())
imgpath = 'data/New1/Val new seg/2082.png'
img = cv2.imread(imgpath,0)
print(np.unique(img))
print(img.shape)
print(img)
# segpath = 'data/bio/val_seg'
# seglist = os.listdir(segpath)
# for seg in seglist:
#     segimg1 = os.path.join(segpath, seg.split('.')[0] + '.png')
#     # print(segimg)
#     # if os.path.exists(segimg):
#     #     os.remove(segimg)
#     img = cv2.imread(segimg1)
#     # # img[img > 0] = 1
#     # # print(np.unique(img))
#     segimg2 = os.path.join(segpath, seg.split('_')[0] + '.png')
#     cv2.imwrite(segimg2, img)
#     print(segimg2)
#     os.remove(segimg1)

