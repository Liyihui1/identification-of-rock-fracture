import torch

from networcks.linknet import DinkNet34
modelname = 'DinkNet34'
modelloadpath = 'weights/liexi/DinkNet34_weights_17_0.8315882855349298.pth'
net = eval(modelname)(num_classes=1)
state_dict = torch.load(modelloadpath)
net.load_state_dict(state_dict)
torch.save(net.state_dict(), modelloadpath, _use_new_zipfile_serialization = False)
