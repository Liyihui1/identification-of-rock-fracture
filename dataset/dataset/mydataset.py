import torch.utils.data as data
# import PIL.Image as Image
import cv2
import os
import torch
import torchvision.transforms.functional as tf
from torchvision import transforms
import random
import numpy as np

def make_dataset(imgroot ,segroot):
    imgs = []
    #print()
    print(len(os.listdir(segroot)))
    assert len(os.listdir(imgroot)) == len(os.listdir(segroot))
    # n = len(os.listdir(imgroot)) // 2
    for i in os.listdir(imgroot):
        img = os.path.join(imgroot, i.split('.')[0] + '.jpg')#注意图片格式
        mask = os.path.join(segroot, i.split('.')[0] + '.png')
        # img = os.path.join(imgroot, i.split('.')[0] + '.png')
        # mask = os.path.join(segroot, i.split('.')[0] +'_mask' +  '.png')
        imgs.append((img, mask))
    return imgs


class myDataset(data.Dataset):
    def __init__(self, imgroot ,segroot, width, height, n_class, transform=None, target_transform = True):
        imgs = make_dataset(imgroot,segroot)
        self.imgs = imgs

        self.transform = transform
        self.target_transform = target_transform
        self.w = width
        self.h = height
        self.classes = n_class

    def rotate(self, image, mask, angle=None):
        if random.random() > 0.5:
            if angle == None:
                angle = transforms.RandomRotation.get_params([-180, 180])  # -180~180随机选一个角度旋转
            if isinstance(angle, list):
                angle = random.choice(angle)
            center = (self.w // 2, self.h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (self.w, self.h))
            mask = cv2.warpAffine(mask, M, (self.w, self.h))
            # image = tf.to_tensor(image)
            # mask = tf.to_tensor(mask)
        return image, mask

    def flip(self, image, mask):  # 水平翻转和垂直翻转
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        # image = tf.to_tensor(image)
        # mask = tf.to_tensor(mask)
        return image, mask

    def randomResizeCrop(self, image, mask, scale=(0.3, 1.0),
                             ratio=(1, 1)):  # scale表示随机crop出来的图片会在的0.3倍至1倍之间，ratio表示长宽比
        img = np.array(image)
        h_image, w_image = img.shape
        resize_size = h_image
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        image = tf.resized_crop(image, i, j, h, w, resize_size)
        mask = tf.resized_crop(mask, i, j, h, w, resize_size)
        # image = tf.to_tensor(image)
        # mask = tf.to_tensor(mask)
        return image, mask

    def adjustContrast(self, image, mask):
        factor = transforms.RandomRotation.get_params([0, 10])  # 这里调增广后的数据的对比度
        image = tf.adjust_contrast(image, factor)
        # mask = tf.adjust_contrast(mask,factor)
        # image = tf.to_tensor(image)
        # mask = tf.to_tensor(mask)
        return image, mask

    def adjustBrightness(self, image, mask):
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据亮度
        image = tf.adjust_brightness(image, factor)
        # mask = tf.adjust_contrast(mask, factor)
        # image = tf.to_tensor(image)
        # mask = tf.to_tensor(mask)
        return image, mask

    def centerCrop(self, image, mask, size=None):  # 中心裁剪
        if size == None: size = image.size  # 若不设定size，则是原图。
        image = tf.center_crop(image, size)
        mask = tf.center_crop(mask, size)
        # image = tf.to_tensor(image)
        # mask = tf.to_tensor(mask)
        return image, mask

    def adjustSaturation(self, image, mask):  # 调整饱和度
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据亮度
        image = tf.adjust_saturation(image, factor)
        # mask = tf.adjust_saturation(mask, factor)
        # image = tf.to_tensor(image)
        # mask = tf.to_tensor(mask)
        return image, mask

    def seg_onhot(self, seg):
        seg = np.array(seg)
        seg_labels = np.eye(self.classes)[seg]
        return tf.to_tensor(seg_labels)

    def twoclass(self, seg):
        # print(np.unique(seg))
        seg = np.array(seg)
        return torch.tensor(seg)

    def __getitem__(self, index):

        x_path, y_path = self.imgs[index]

        img_x = cv2.imread(x_path, -1)
        # print(img_x.shape)
        img_y = cv2.imread(y_path, 0)
        img_x = cv2.resize(img_x, (self.w, self.h))

        img_y = cv2.resize(img_y, (self.w, self.h))
        img_x, img_y = self.rotate(img_x, img_y)
        img_x, img_y = self.flip(img_x, img_y)
        # img_x, img_y = self.randomResizeCrop(img_x, img_y)
        # img_x, img_y = self.centerCrop(img_x, img_y)
        # img_x, img_y = self.adjustSaturation(img_x, img_y)
        # img_x, img_y = self.adjustContrast(img_x, img_y)
        # img_x, img_y = self.adjustBrightness(img_x, img_y)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform == False:
            img_y = self.twoclass(img_y)
        elif self.target_transform == True:
            img_y = self.seg_onhot(img_y)
        return {'image': img_x, 'mask': img_y}

    def __len__(self):
        return len(self.imgs)