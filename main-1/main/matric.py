import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class segmetric(object):
    def __init__(self, n_class = 21, device = 'cuda:0'):
        self.num_classes = n_class
        self.device = device

    def process(self, input, target):
        if self.num_classes == 1:
            input = torch.where(input>0.5, torch.ones_like(input).to(self.device), torch.zeros_like(input).to(self.device))
            target = target.squeeze(1)
        else:
            input = torch.argmax(input, dim=1)
            target = torch.argmax(target, dim=1)
        return input, target

    def Iou(self, input, target):
        input, target = self.process(input, target)
        cls_ious = []
        for cls in range(self.num_classes):
            pred_id = input == cls
            target_id = target == cls
            intersection = pred_id[target_id].sum().item()
            union = target_id[target_id].sum().item() + pred_id[pred_id].sum().item() - intersection
            if target_id[target_id].sum().item() == 0:
                cls_iou = 1
            else:
                cls_iou = intersection/union
            cls_ious.append(cls_iou)
        # print(cls_ious)
        return cls_ious

    def mean_iou(self, input, target):
        cls_ious = self.Iou(input, target)
        return np.mean(np.array(cls_ious))

    def precision(self, input, target):
        input, target = self.process(input, target)
        cls_precisions = []
        for cls in range(self.num_classes):
            pred_id = input == cls
            target_id = target == cls
            tp = pred_id[target_id].sum().item()
            P = pred_id[pred_id].sum().item() #fp+tp = P
            if P == 0:
                precision = 1
            else:
                precision = tp/P
            cls_precisions.append(precision)
        return cls_precisions

    def average_precision(self, input, target):
        cls_precisions = self.precision(input, target)
        return np.mean(np.array(cls_precisions))

    def binary(self, input, target):
        input, target = self.process(input, target)
        # print(target.unique())
        # print(input.unique())
        pred_id1 = input == 1
        target_id1 = target == 1
        pred_id2 = input == 0
        target_id2 = target == 0
        tp = pred_id1[target_id1].sum().item()
        P = pred_id1[pred_id1].sum().item()  # fp+tp = P
        tn = pred_id2[target_id2].sum().item()
        F = pred_id2[pred_id2].sum().item() # fn + tn = F
        fn = F - tn
        fp = P - tp
        return tp, fp, tn, fn
    def binary_percision(self, input, target):
        tp, fp, tn, fn = self.binary(input, target)
        if fp+ tp == 0:
            bp = torch.tensor(0)
        else:
            bp = tp / (fp + tp)
        return bp

    def binary_ACC(self, input, target):
        tp, fp, tn, fn = self.binary(input, target)
        if fp + tp + tn + fn == 0:
            bp = torch.tensor(0)
        else:
            bp = (tp + tn)/ (fp + tp + tn + fn)
        return bp

    def binary_recall(self, input, target):
        tp, fp, tn, fn = self.binary(input, target)
        if fp+ tp == 0:
            bp = torch.tensor(0)
        else:
            bp = tp / (fn + tp)
        return bp

    def F1(self, input, target):
        P = self.binary_percision(input, target)
        R = self.binary_recall(input, target)
        if P+R ==0:
            f1 = 0
        else:
            f1 = 2*P*R/(P+R)
        return f1
    # def binary_percision(self, input, target):
    #     input, target = self.process(input, target)
    #     pred_id = input == 1
    #     target_id = target == 1
    #     tp = pred_id[target_id].sum().item()
    #     P = pred_id[pred_id].sum().item()  # fp+tp = P
    #     bp = tp/P
    #     return bp