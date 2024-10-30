import sys
import numpy as np
import pickle as pkl
import pandas as pd

import torch
from torch.utils.data import DataLoader
cuda = torch.cuda.is_available()
import vae as vae
from sklearn.preprocessing import MinMaxScaler
torch.set_printoptions(sci_mode = False)
from torch.utils.tensorboard import SummaryWriter
#%% Load data and scaler
df = #READ DATASET

ndis = 15 # the number of disease features
scaler = MinMaxScaler()
data_train_scaled = df.copy()
data_train_scaled.iloc[:,ndis:] = scaler.fit_transform(data_train_scaled.iloc[:,ndis:])



#%% Train Model:
path = "Example\\Path\\tensorboard" #Replace with path to tensorboard
writer = SummaryWriter(log_dir=path)
bnorm = False
concat = False
torch.manual_seed(260395) ## set seeds for reproducability
np.random.seed(260395) ## set seeds for reproducability
train_dataloader = DataLoader(torch.from_numpy(data_train_scaled.values).float(),
                          batch_size = 256, 
                          shuffle = True,
                          num_workers = 0,
                          pin_memory = cuda)

# Here the full dataset is used as validation - during e.g. cross-validation experiments this would be an independent validation set
val_dataloader = DataLoader(torch.from_numpy(data_train_scaled.values).float(),
                        batch_size = 256,
                        num_workers = 0,
                        pin_memory = cuda)

epochs  = 2000
net = vae.VAE(input_size = [15,3], first_dims = [15,3],
                                 hidden_layers = [8], latent_features = 2, bnorm = bnorm, concat = concat)

if cuda:
    net = net.cuda()

save_path = "Example\\path\\for\\model.pth" ## replace with path to checkpoint of model

args = {'epochs': epochs,
        'lr': 0.001,
        'model': net,
        'trainloader': train_dataloader,
        'testloader': val_dataloader,
        'cuda': cuda,
        'w_a': 4,
        'w_aux': 1,
        'writer': writer,
        'save_path': save_path,
	'ndis': ndis
        }

output = vae.train_new(args)

writer.flush()

##save output from training
filename = 'Example\\filename\\for\\saving\\training\\output.pkl'
file = open(filename, 'wb')
pkl.dump(output, file)
file.close()



       
