from networcks.deeplab.model.deeplab import DeepLab
from networcks.linknet import DinkNet50
from networcks.linknet import LinkNet50
from networcks.linknet import DinkNet34
from networcks.linknet import MDinkNet50
from networcks.linknet import MDinkNet_xception
from networcks.linknet import DinkNet_xception
from networcks.unet import Unet_resnet
from networcks.fcn import Resnet_FCN8s
from networcks.segnet import SegNet
import torch
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
import cv2
import numpy as np
import os
import time

testpath = 'data/xunlian/imagesyingyong'
predimgs = 'predimgs/'
modelloadpath = 'weights/liexi/DinkNet34_weights_42_0.5057716637145223.pth'
n_class = 1
imgsize = 1024
modelname = 'DinkNet34'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_transforms = transforms.Compose([
    transforms.ToTensor(),  # -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
])
#model = eval(modelname)(backbone='xception', output_stride=16, num_classes=n_class).to(device)#deeplab
model = eval(modelname)(num_classes=n_class).to(device)  # else
model.load_state_dict(torch.load(modelloadpath))
model.eval()
torch.no_grad()
imglist = os.listdir(testpath)
print('开始')
if os.path.exists(predimgs + modelname) is not True:
    os.mkdir(predimgs + modelname)

times = []
for i in imglist:
    imgpath = os.path.join(testpath, i)
    predpath = os.path.join(predimgs + modelname + '/', i.split('.')[0] + '.png')
    img = cv2.imread(imgpath, -1)
    (w, h, c) = img.shape
    # print(img.shape)
    img = cv2.resize(img, (imgsize, imgsize))
    img = x_transforms(img).cuda()
    img = img.unsqueeze(0)
    # print(img.shape)
    T1 = time.time()
    pred = model(img)
    T2 = time.time()
    pred = pred.detach().cpu()
    pred = torch.where(pred > 0.5, torch.tensor(255), torch.tensor(0))
    # print(pred.shape)
    pred = pred.squeeze(0)
    pred = pred.squeeze(0)
    # print(pred.shape)
    pred = pred.numpy()
    # pred = cv2.resize(pred, (w, h))
    T = T2 - T1
    times.append(T)
    cv2.imwrite(predpath, pred)

seglist = os.listdir(predimgs + modelname + '/')
for i in seglist:
    path = os.path.join(predimgs + modelname + '/', i)
    imgpath = os.path.join(testpath, i.split('.')[0] + '.jpg')
    img = cv2.imread(imgpath, -1)
    (w, h, c) = img.shape
    seg = cv2.imread(path, 0)
    seg = cv2.resize(seg,(h, w))
    cv2.imwrite(path, seg)

time_sum = 0
for i in times:
    time_sum += i
print('fps:%f'%(1.0/(time_sum/len(times))))
print('完成')
