import os
import random
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]= "2"

import torch
import torch.nn as nn
import albumentations
import torch.optim as optim
from albumentations.pytorch import ToTensorV2, ToTensor

from C2C.models.resnet import *
from C2C import train
from C2C.loss import KLDLoss
from C2C.eval_model import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)

torch.backends.cudnn.deterministic=True

CSV_PATH = '/workspace/C2C/data/patch_data_1000.csv'
df = pd.read_csv(CSV_PATH)

# Initialize Model
model_ft = WSIClassifier(6, bn_track_running_stats=True)
model_ft = model_ft.to(device)

# Data Transforms
data_transforms = albumentations.Compose([
    ToTensor()
    ])    

# Cross Entropy Loss 
criterion_ce = nn.CrossEntropyLoss()
criterion_kld = KLDLoss()
criterion_dic = {'CE': criterion_ce, 'KLD': criterion_kld}

# Observe that all parameters are being optimized
optimizer = optim.Adam(model_ft.parameters(), lr=1e-4)

model_ft = train.train_model(model_ft, 
                             criterion_dic, 
                             optimizer, 
                             df, 
                             data_transforms=data_transforms,
                             alpha=1, 
                             beta=0.01, 
                             gamma=0.01, 
                             num_epochs=50, 
                             fpath='checkpoint.pt',
                             topk=False,
                             wandb_monitor=False)