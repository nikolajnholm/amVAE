import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import torch

cuda = False

import helpers.utility_functions as uf
import helpers.cluster_funcs as cf

torch.set_printoptions(sci_mode = False)

#%%
args = {'input_size': [15, 3],
        'first_dims': [15, 3],
        'hidden_layers': [8],
        'latent_features': 2,
        'bnorm': False,
        'cuda': cuda,
        'concat':  False,
        'save_path': 'path\\of\\saved\\amVAE\\model.pth'
        }
#%%

df = pd.read_pickle(r'dataset\\file\\location')

train_data  = df.copy()

retdf = uf.load_model_compute_val_lf_global(args, train_data)
#%% features for clustering (depends on latent dimensionality, here 2)
x = retdf[['mu1','mu2']]
x = np.array(x)

# run clustering for multiple k's
ks = range(3,36)
directory = "dir\\path\\for\\clustering\\results"
np.random.seed(260395)
best_mod, best_wcss = cf.run_multiple_report_best_new(ks, x, directory, max_counter =  25)
