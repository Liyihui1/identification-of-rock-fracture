import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from losses import CrossEntropy
from losses import DiceLoss
from dataset.mydataset import myDataset
from matric import segmetric
from networcks.deeplab.model.deeplab import DeepLab
from networcks.linknet import DinkNet50

#config参数设置
#train
trainimg = 'data/voc2012/train'
trainseg = 'data/voc2012/train_seg'
valimg = 'data/voc2012/val'
valseg = 'data/voc2012/val_seg'
epochs = 40
batchsize = 2
val_batchsize = 2
learningrate = 0.0001
width = 512
height = 512
weightdecay = 1e-8
n_channels = 3
n_class = 21
modelname = 'DeepLab'
# modelname = 'DinkNet50'
#test
savepath = ''
# 是否使用cuda
x_transforms = transforms.Compose([
    transforms.ToTensor(),  # -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
])

# mask只需要转换为tensor
y_transforms = True
metric = segmetric(n_class)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_model(model, criterion, optimizer, train_dataload, val_dataload, num_epochs=40):
    train_loss_record = []
    train_Iou_record = []
    val_loss_record = []
    val_AP_record = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        dt_size = len(train_dataload)
        val_size = len(val_dataload)
        epoch_loss = 0
        val_epoch_loss = 0
        val_epoch_AP = 0
        step = 0
        for batch in train_dataload:
            step += 1
            batch_imgs = batch['image']
            batch_masks = batch['mask']
            inputs = batch_imgs.to(device=device, dtype=torch.float32)
            labels = batch_masks.to(device=device, dtype=torch.float32)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            AP = metric.average_precision(outputs, labels)
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f, train_precision:%0.3f" % (step, dt_size, loss.item(), AP))#多分类时AP要加.item()
        print("epoch %d loss:%0.3f" % (epoch+1, (epoch_loss/dt_size)))
        train_loss_record.append((epoch_loss/dt_size))
        for batch in val_dataload:
            batch_imgs = batch['image']
            batch_masks = batch['mask']
            inputs = batch_imgs.to(device=device, dtype=torch.float32)
            labels = batch_masks.to(device=device, dtype=torch.float32)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_AP = metric.average_precision(outputs, labels)
            val_epoch_loss += loss.item()
            val_epoch_AP += val_AP#多分类时AP要加.item()
        print("epoch %d val_loss:%0.3f" % (epoch+1, (val_epoch_loss/val_size)))
        print("val_AP:", (val_epoch_AP/val_size))
        val_loss_record.append((val_epoch_loss/val_size))
        val_AP_record.append((val_epoch_AP/val_size))
        # torch.save(model.state_dict(), 'weights/liexi/weights_%d_%s.pth' % (epoch, (val_clsiou/val_size)))
    print("trainloss:", train_loss_record)
    print("valloss:", val_loss_record)
    print("valAP:", val_AP_record)
    return model

def train():
    model = eval(modelname)(backbone='resnet', output_stride=16, num_classes=n_class).to(device)#deeplab
    # model = eval(modelname)(num_classes=n_class).to(device)  #
    criterion = CrossEntropy()
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=learningrate, momentum=0.9, weight_decay=weightdecay)
    train = myDataset(trainimg, trainseg,  width, height, n_class,transform=x_transforms, target_transform=y_transforms)
    val = myDataset(valimg, valseg,  width, height, n_class, transform=x_transforms, target_transform=y_transforms)
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