import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from losses import CrossEntropy
from losses import DiceLoss
from dataset.mydataset import myDataset
from matric import segmetric
# import torch.nn.functional as F
from networcks.deeplab.model.deeplab import DeepLab
from networcks.linknet import DinkNet50
from networcks.linknet import DinkNet34
from networcks.linknet import MDinkNet50
from networcks.linknet import MDinkNet_xception
from networcks.linknet import DinkNet_xception
from networcks.AttDlinknet34 import AttDinkNet34
from networcks.linknet import LinkNet50
from networcks.unet import Unet_resnet
from networcks.fcn import Resnet_FCN8s
from networcks.segnet import SegNet
from networcks.MSTSOD import MSST
from networcks.SWT import SwinIR


#config参数设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#train
trainimg = 'data/xunlian/train/train_yuan'
trainseg = 'data/xunlian/train_seg'
valimg = 'data/xunlian/val/val_yuan'
valseg = 'data/xunlian/val_seg'
epochs=50
batchsize = 2
val_batchsize = 1
learningrate = 0.0005
width = 224
height = 224
weightdecay = 1e-8
n_channels = 3
n_class = 1
modelname = 'MSST'
# modelname = 'DinkNet50'
#test
savepath = ''
# 是否使用cuda
x_transforms = transforms.Compose([
    transforms.ToTensor(),  # -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
])


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:0')

def train_model(model, criterion, optimizer, train_dataload, val_dataload, num_epochs=40):
    train_loss_record = []
    train_Iou_record = []
    val_loss_record = []
    val_AP_record = []
    val_R_record = []
    val_ACC_record = []
    val_F1_record = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        dt_size = len(train_dataload)
        val_size = len(val_dataload)
        epoch_loss = 0
        val_epoch_loss = 0
        val_epoch_AP = 0
        val_epoch_R = 0
        val_epoch_ACC = 0
        val_epoch_F1 = 0
        step = 0
        model = model.to(device0)
        metric = segmetric(n_class, device0)
        model.train()
        for batch in train_dataload:
            step += 1
            batch_imgs = batch['image']
            batch_masks = batch['mask']
            inputs = batch_imgs.to(device=device0, dtype=torch.float32)
            labels = batch_masks.to(device=device0, dtype=torch.float32)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            if modelname == 'DeepLab': outputs = torch.sigmoid(outputs)
            outputs = outputs.squeeze(1)
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            AP = metric.binary_percision(outputs, labels)
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f, train_precision:%0.3f" % (step, dt_size, loss.item(), AP))#多分类时AP要加.item()
        print("epoch %d loss:%0.3f" % (epoch+1, (epoch_loss/dt_size)))
        train_loss_record.append((epoch_loss/dt_size))
        model = model.to(device1)
        metric = segmetric(n_class, device1)
        with torch.no_grad():
            model.eval()
            for batch in val_dataload:
                batch_imgs = batch['image']
                batch_masks = batch['mask']
                inputs = batch_imgs.to(device=device1, dtype=torch.float32)
                labels = batch_masks.to(device=device1, dtype=torch.float32)
                outputs = model(inputs)
                if modelname == 'DeepLab': outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)
                outputs = outputs.squeeze(1)
                val_AP = metric.binary_percision(outputs, labels)
                val_R = metric.binary_recall(outputs, labels)
                val_ACC = metric.binary_ACC(outputs, labels)
                val_F1 = metric.F1(outputs, labels)
                val_epoch_loss += loss.item()
                val_epoch_AP += val_AP#多分类时AP要加.item()
                val_epoch_R += val_R
                val_epoch_ACC += val_ACC
                val_epoch_F1 += val_F1
        print("epoch %d val_loss:%0.3f" % (epoch+1, (val_epoch_loss/val_size)))
        print("val_AP:", (val_epoch_AP/val_size))
        print("val_R:", (val_epoch_R / val_size))
        print("val_ACC:", (val_epoch_ACC / val_size))
        print("val_F1:", (val_epoch_F1 / val_size))
        val_loss_record.append((val_epoch_loss/val_size))
        val_AP_record.append((val_epoch_AP/val_size))
        val_R_record.append((val_epoch_R / val_size))
        val_ACC_record.append((val_epoch_ACC / val_size))
        val_F1_record.append((val_epoch_F1 / val_size))
        torch.save(model.state_dict(), 'weights/liexi/'+ modelname +'_weights_%d_%s.pth' % (epoch, (val_epoch_AP/val_size)))
    print("trainloss:", train_loss_record)
    print("valloss:", val_loss_record)
    print("valAP:", val_AP_record)
    print("valR:", val_R_record)
    print("valACC:", val_ACC_record)
    print("valF1:", val_F1_record)
    return model

def train():
    # model = eval(modelname)(backbone='xception', output_stride=16, num_classes=n_class)#deeplab
    # model = eval(modelname)(num_classes=n_class)
    model = eval(modelname)(img_size=224, window_size=8, in_channels=3, num_out_ch=1, hidden_dims = [32, 64, 128])
    # model = torch.nn.DataParallel(model)
    # model = model.to(device)# else
    # criterion = torch.nn.BCELoss()
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=learningrate, momentum=0.9, weight_decay=weightdecay)
    train = myDataset(trainimg, trainseg,  width, height, n_class,transform=x_transforms, target_transform=False)
    val = myDataset(valimg, valseg,  width, height, n_class, transform=x_transforms, target_transform=False)
    train_loader = DataLoader(train,
                              batch_size=batchsize,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                             drop_last=True)
    val_loader = DataLoader(val,
                            batch_size=val_batchsize,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True)
    train_model(model, criterion, optimizer, train_loader, val_loader, epochs)
    return 0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('开始训练')
    train()


