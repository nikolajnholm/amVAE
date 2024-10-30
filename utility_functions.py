import numpy as np
import torch
from torch.utils.data import DataLoader
import vae as vae
from sklearn.preprocessing import MinMaxScaler
torch.set_printoptions(sci_mode = False)


def load_model_compute_val_lf_global(args, df):
    input_size = args['input_size']
    first_dims = args['first_dims']
    hidden_layers = args['hidden_layers']
    latent_features = args['latent_features']
    bnorm = args['bnorm']
    cuda = args['cuda']
    concat  = args["concat"]
    ndis = input_size[0]
    providedScaler = False
    if 'scaler' in args:
        scaler = args['scaler']
        providedScaler = True
    else:
        scaler = MinMaxScaler()
    
    
    ## load model, compute 
    
    net = vae.VAE(input_size = input_size, first_dims = first_dims, hidden_layers = hidden_layers, latent_features = latent_features, bnorm = bnorm, concat = concat)
    checkpoint = torch.load(args['save_path'])
    
    state_dict =  checkpoint
        
    net.load_state_dict(state_dict)

    data_train_scaled = df.copy()
    if providedScaler:
        data_train_scaled.iloc[:,ndis:] = scaler.transform(data_train_scaled.iloc[:,ndis:])
    else:
        data_train_scaled.iloc[:,ndis:] = scaler.fit_transform(data_train_scaled.iloc[:,ndis:])

    val_dataloader = DataLoader(torch.from_numpy(data_train_scaled.values).float(),
                            batch_size = 256,
                            num_workers = 0,
                            pin_memory = cuda)
    
    with torch.no_grad():
        mus = []
        logvars = []
        net.eval()
        for x in val_dataloader:
            if cuda:
                x = x.cuda()
            x_hat_dis, x_hat_time, mu, logvar, z = net(x[:,:15], x[:,15:])
            mu = mu.cpu().numpy()
            mus.append(mu)
            logvar = logvar.cpu().numpy()
            logvars.append(logvar)
    muss = np.vstack(mus)
    logvarss = np.vstack(logvars)
    
    latent_df  = df.copy()
    latent_df['mu1'] =  muss[:,0]
    latent_df['mu2'] =  muss[:,1]
    latent_df['logvar1'] =  logvarss[:,0]
    latent_df['logvar2'] =  logvarss[:,1]
    
    
    return latent_df



        
       