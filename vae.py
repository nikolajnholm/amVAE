import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_size, first_dims, hidden_layers, latent_features=2, bias = True):
        super(Encoder, self).__init__()
        
        # Encoder part
        # Input diseases to first encoding dimension
        self.input_size = input_size
        self.first_dims = first_dims
        
        self.dis_encoder = nn.Linear(input_size[0], first_dims[0], bias = bias)
        self.time_encoder = nn.Linear(input_size[1], first_dims[1], bias = bias)
        
        self.hidden_layers_encoder = nn.ModuleList([nn.Linear(first_dims[0] +
                                                              first_dims[1],
                                                              hidden_layers[0], bias = bias)])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers_encoder.extend([nn.Linear(h1, h2, bias = bias) for h1, h2 in layer_sizes])
        self.encoder_output = nn.Linear(hidden_layers[-1], 2*latent_features, bias = bias)
        
       
    def forward(self, x_dis, x_time):
        x_dis = F.relu(self.dis_encoder(x_dis))
        x_time = F.relu(self.time_encoder(x_time))
        
        x_hid = torch.cat([x_dis, x_time], dim = 1)
        
        for each in self.hidden_layers_encoder:
            x_hid = F.relu(each(x_hid))
            
        x_lat = self.encoder_output(x_hid)
        
        mu, logvar = torch.chunk(x_lat, 2, dim = 1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
        
class Decoder(nn.Module):
    def __init__(self, input_size, first_dims, hidden_layers, latent_features=2, bias = True):
        super(Decoder, self).__init__()
        
        self.first_dims = first_dims
        self.hidden_layers_decoder = nn.ModuleList([nn.Linear(latent_features, hidden_layers[-1], bias = bias)])
        # Add a variable number of more hidden layers
        hl_rev = hidden_layers[::-1]
        layer_sizes = zip(hl_rev[:-1], hl_rev[1:])
        self.hidden_layers_decoder.extend([nn.Linear(h1, h2, bias = bias) for h1, h2 in layer_sizes])
        
        self.hidden_layers_decoder.append(nn.Linear(hl_rev[-1], first_dims[0] + first_dims[1], bias = bias))
        
        self.dis_decoder = nn.Linear(first_dims[0], input_size[0], bias = bias)
        self.time_decoder = nn.Linear(first_dims[1], input_size[1], bias = bias) 
        
        
    def forward(self, z):
        x_temp = z
        # Decode
        for each in self.hidden_layers_decoder:
            x_temp = F.relu(each(x_temp))
        
        x_hat_dis = x_temp[:,:self.first_dims[0]]
        x_hat_time = x_temp[:,self.first_dims[0]:]
        
        x_hat_dis = self.dis_decoder(x_hat_dis)
        x_hat_time = self.time_decoder(x_hat_time)
        
        return x_hat_dis, x_hat_time
        
        
    
    
class EncoderWithBnorm(nn.Module):
    def __init__(self, input_size, first_dims, hidden_layers, latent_features=2, bias = True):
        super(EncoderWithBnorm, self).__init__()
        
        # Encoder part
        # Input diseases to first encoding dimension
        self.input_size = input_size
        self.first_dims = first_dims
        
        self.dis_encoder = nn.Linear(input_size[0], first_dims[0], bias = bias)
        self.time_encoder = nn.Linear(input_size[1], first_dims[1], bias = bias)
        
        self.hidden_layers_encoder = nn.ModuleList([nn.Linear(first_dims[0] +
                                                              first_dims[1],
                                                              hidden_layers[0], bias = bias)])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers_encoder.extend([nn.Linear(h1, h2, bias = bias) for h1, h2 in layer_sizes])
        self.encoder_output = nn.Linear(hidden_layers[-1], 2*latent_features, bias = bias)
        
        
        self.bnorm_list = nn.ModuleList([nn.BatchNorm1d(first_dims[0]+first_dims[1])])
        self.bnorm_list.extend([nn.BatchNorm1d(h) for h in hidden_layers[:-1]])
      
       
    def forward(self, x_dis, x_time):
        x_dis = F.relu(self.dis_encoder(x_dis))
        x_time = F.relu(self.time_encoder(x_time))
        
        x_hid = torch.cat([x_dis, x_time], dim = 1)

        
        for i, each in enumerate(self.hidden_layers_encoder[:-1]):
            x_hid = self.bnorm_list[i](x_hid)
            x_hid = F.relu(each(x_hid))
        
        x_hid = F.relu(self.hidden_layers_encoder[-1](x_hid))
        x_lat = self.encoder_output(x_hid)
        
        mu, logvar = torch.chunk(x_lat, 2, dim = 1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
        
class DecoderWithBnorm(nn.Module):
    def __init__(self, input_size, first_dims, hidden_layers, latent_features=2, bias = True):
        super(DecoderWithBnorm, self).__init__()
        
        self.first_dims = first_dims
        self.hidden_layers_decoder = nn.ModuleList([nn.Linear(latent_features, hidden_layers[-1], bias = bias)])
        # Add a variable number of more hidden layers
        hl_rev = hidden_layers[::-1]
        layer_sizes = zip(hl_rev[:-1], hl_rev[1:])
        self.hidden_layers_decoder.extend([nn.Linear(h1, h2, bias = bias) for h1, h2 in layer_sizes])
        
        self.hidden_layers_decoder.append(nn.Linear(hl_rev[-1], first_dims[0] + first_dims[1], bias = bias))
        
        self.dis_decoder = nn.Linear(first_dims[0], input_size[0], bias = bias)
        self.time_decoder = nn.Linear(first_dims[1], input_size[1], bias = bias) 
        
        self.bnorm_list = nn.ModuleList([nn.BatchNorm1d(hl_rev[0])])
        self.bnorm_list.extend([nn.BatchNorm1d(h) for h in hl_rev[1:]])
        
    def forward(self, z):
        x_temp = z
        # Decode
        x_temp = self.hidden_layers_decoder[0](x_temp)
        
        for i, each in enumerate(self.hidden_layers_decoder[1:]):
            x_temp = self.bnorm_list[i](x_temp)
            x_temp = F.relu(each(x_temp))
        
        x_hat_dis = x_temp[:,:self.first_dims[0]]
        x_hat_time = x_temp[:,self.first_dims[0]:]
        
        x_hat_dis = self.dis_decoder(x_hat_dis)
        x_hat_time = self.time_decoder(x_hat_time)
        
        return x_hat_dis, x_hat_time





class EncoderNoSplit(nn.Module):
    def __init__(self, input_size, first_dims, hidden_layers, latent_features=2, bias = True):
        super(EncoderNoSplit, self).__init__()
        
        # Encoder part
        # Input diseases to first encoding dimension
        self.input_size = input_size
        self.first_dims = first_dims
        
        
        self.input_encoder = nn.Linear(input_size[0] + input_size[1], first_dims[0] + first_dims[1], bias = bias)
        
        self.hidden_layers_encoder = nn.ModuleList([nn.Linear(first_dims[0] +
                                                              first_dims[1],
                                                              hidden_layers[0], bias = bias)])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers_encoder.extend([nn.Linear(h1, h2, bias = bias) for h1, h2 in layer_sizes])
        self.encoder_output = nn.Linear(hidden_layers[-1], 2*latent_features, bias = bias)
        
       
    def forward(self, x_dis, x_time):
        
        x_hid = F.relu(self.input_encoder(torch.cat([x_dis, x_time], dim = 1)))
        
        for each in self.hidden_layers_encoder:
            x_hid = F.relu(each(x_hid))
            
        x_lat = self.encoder_output(x_hid)
        
        mu, logvar = torch.chunk(x_lat, 2, dim = 1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
    
class EncoderNoSplitBnorm(nn.Module):
    def __init__(self, input_size, first_dims, hidden_layers, latent_features=2, bias = True):
        super(EncoderNoSplitBnorm, self).__init__()
        
        # Encoder part
        # Input diseases to first encoding dimension
        self.input_size = input_size
        self.first_dims = first_dims
        
        
        self.input_encoder = nn.Linear(input_size[0] + input_size[1], first_dims[0] + first_dims[1], bias = bias)
        
        self.hidden_layers_encoder = nn.ModuleList([nn.Linear(first_dims[0] +
                                                              first_dims[1],
                                                              hidden_layers[0], bias = bias)])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers_encoder.extend([nn.Linear(h1, h2, bias = bias) for h1, h2 in layer_sizes])
        self.encoder_output = nn.Linear(hidden_layers[-1], 2*latent_features, bias = bias)
        self.bnorm_list = nn.ModuleList([nn.BatchNorm1d(first_dims[0]+first_dims[1])])
        self.bnorm_list.extend([nn.BatchNorm1d(h) for h in hidden_layers[:-1]])
       
    def forward(self, x_dis, x_time):
        
        x_hid = F.relu(self.input_encoder(torch.cat([x_dis, x_time], dim = 1)))
        
        for i, each in enumerate(self.hidden_layers_encoder):
            x_hid = self.bnorm_list[i](x_hid)
            x_hid = F.relu(each(x_hid))
            
        x_lat = self.encoder_output(x_hid)
        
        mu, logvar = torch.chunk(x_lat, 2, dim = 1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar

class DecoderNoSplit(nn.Module):
    def __init__(self, input_size, first_dims, hidden_layers, latent_features=2, bias = True):
        super(DecoderNoSplit, self).__init__()
        
        self.first_dims = first_dims
        self.hidden_layers_decoder = nn.ModuleList([nn.Linear(latent_features, hidden_layers[-1], bias = bias)])
        # Add a variable number of more hidden layers
        hl_rev = hidden_layers[::-1]
        layer_sizes = zip(hl_rev[:-1], hl_rev[1:])
        self.hidden_layers_decoder.extend([nn.Linear(h1, h2, bias = bias) for h1, h2 in layer_sizes])
        
        self.hidden_layers_decoder.append(nn.Linear(hl_rev[-1], first_dims[0] + first_dims[1], bias = bias))
        
        self.dis_decoder = nn.Linear(first_dims[0] + first_dims[1], input_size[0], bias = bias)
        self.time_decoder = nn.Linear(first_dims[0] + first_dims[1], input_size[1], bias = bias)
              
    def forward(self, z):
        x_temp = z
        # Decode
        for each in self.hidden_layers_decoder:
            x_temp = F.relu(each(x_temp))
        
        x_hat_dis = self.dis_decoder(x_temp)
        x_hat_time = self.time_decoder(x_temp)
        
        return x_hat_dis, x_hat_time
    
class DecoderNoSplitBnorm(nn.Module):
    def __init__(self, input_size, first_dims, hidden_layers, latent_features=2, bias = True):
        super(DecoderNoSplitBnorm, self).__init__()
        
        self.first_dims = first_dims
        self.hidden_layers_decoder = nn.ModuleList([nn.Linear(latent_features, hidden_layers[-1], bias = bias)])
        # Add a variable number of more hidden layers
        hl_rev = hidden_layers[::-1]
        layer_sizes = zip(hl_rev[:-1], hl_rev[1:])
        self.hidden_layers_decoder.extend([nn.Linear(h1, h2, bias = bias) for h1, h2 in layer_sizes])
        
        self.hidden_layers_decoder.append(nn.Linear(hl_rev[-1], first_dims[0] + first_dims[1], bias = bias))
        
        self.dis_decoder = nn.Linear(first_dims[0] + first_dims[1], input_size[0], bias = bias)
        self.time_decoder = nn.Linear(first_dims[0] + first_dims[1], input_size[1], bias = bias)
        
        self.bnorm_list = nn.ModuleList([nn.BatchNorm1d(hl_rev[0])])
        self.bnorm_list.extend([nn.BatchNorm1d(h) for h in hl_rev[1:]])
              
    def forward(self, z):
        #print(z.shape)
        x_temp = z
        # Decode
        x_temp = self.hidden_layers_decoder[0](x_temp)
        for i, each in enumerate(self.hidden_layers_decoder[1:]):
            x_temp = self.bnorm_list[i](x_temp)
            x_temp = F.relu(each(x_temp))
        
        x_hat_dis = self.dis_decoder(x_temp)
        x_hat_time = self.time_decoder(x_temp)
        
        return x_hat_dis, x_hat_time
    
class DecoderWithSigmoidNoSplit(nn.Module):
    def __init__(self, input_size, first_dims, hidden_layers, latent_features=2, bias = True):
        super(DecoderWithSigmoidNoSplit, self).__init__()
        
        self.first_dims = first_dims
        self.hidden_layers_decoder = nn.ModuleList([nn.Linear(latent_features, hidden_layers[-1], bias = bias)])
        # Add a variable number of more hidden layers
        hl_rev = hidden_layers[::-1]
        layer_sizes = zip(hl_rev[:-1], hl_rev[1:])
        self.hidden_layers_decoder.extend([nn.Linear(h1, h2, bias = bias) for h1, h2 in layer_sizes])
        
        self.hidden_layers_decoder.append(nn.Linear(hl_rev[-1], first_dims[0] + first_dims[1], bias = bias))
        
        self.dis_decoder = nn.Linear(first_dims[0] + first_dims[1], input_size[0], bias = bias)
        self.time_decoder = nn.Linear(first_dims[0] + first_dims[1], input_size[1], bias = bias)
              
    def forward(self, z):
        x_temp = z
        # Decode
        for each in self.hidden_layers_decoder:
            x_temp = F.relu(each(x_temp))
        
        x_hat_dis = F.sigmoid(self.dis_decoder(x_temp))
        x_hat_time = self.time_decoder(x_temp)
        
        return x_hat_dis, x_hat_time
    
class DecoderWithSigmoid(nn.Module):
    def __init__(self, input_size, first_dims, hidden_layers, latent_features=2, bias = True):
        super(DecoderWithSigmoid, self).__init__()
        
        self.first_dims = first_dims
        self.hidden_layers_decoder = nn.ModuleList([nn.Linear(latent_features, hidden_layers[-1], bias = bias)])
        # Add a variable number of more hidden layers
        hl_rev = hidden_layers[::-1]
        layer_sizes = zip(hl_rev[:-1], hl_rev[1:])
        self.hidden_layers_decoder.extend([nn.Linear(h1, h2, bias = bias) for h1, h2 in layer_sizes])
        
        self.hidden_layers_decoder.append(nn.Linear(hl_rev[-1], first_dims[0] + first_dims[1], bias = bias))
        
        self.dis_decoder = nn.Linear(first_dims[0], input_size[0], bias = bias)
        self.time_decoder = nn.Linear(first_dims[1], input_size[1], bias = bias) 
        
        
    def forward(self, z):
        #print(z.shape)
        x_temp = z
        # Decode
        for each in self.hidden_layers_decoder:
            x_temp = F.relu(each(x_temp))

        x_hat_dis = x_temp[:,:self.first_dims[0]]
        x_hat_time = x_temp[:,self.first_dims[0]:]
        x_hat_dis = F.sigmoid(self.dis_decoder(x_hat_dis))
        x_hat_time = self.time_decoder(x_hat_time)
        
        return x_hat_dis, x_hat_time

class VAE(nn.Module):
    def __init__(self, input_size, first_dims, hidden_layers, latent_features = 2, concat = True, bnorm = False, beta = 1, bias = True):
        super(VAE, self).__init__()
        if concat:
            if bnorm:
                self.encoder = EncoderWithBnorm(input_size, first_dims, hidden_layers, latent_features, bias = bias)
                self.decoder = DecoderWithBnorm(input_size, first_dims, hidden_layers, latent_features, bias = bias)
            else:
                self.encoder = Encoder(input_size, first_dims, hidden_layers, latent_features, bias = bias)
                self.decoder = Decoder(input_size, first_dims, hidden_layers, latent_features, bias = bias)
        else:
            if bnorm:
                self.encoder = EncoderNoSplitBnorm(input_size, first_dims, hidden_layers, latent_features, bias = bias)
                self.decoder = DecoderNoSplitBnorm(input_size, first_dims, hidden_layers, latent_features, bias = bias)
            else:
                self.encoder = EncoderNoSplit(input_size, first_dims, hidden_layers, latent_features, bias = bias)
                self.decoder = DecoderNoSplit(input_size, first_dims, hidden_layers, latent_features, bias = bias)
                
        self.beta = beta
        self.MSELoss = nn.MSELoss(reduce = False)
        self.CELoss = nn.BCEWithLogitsLoss(reduce = False)
             
        
    def forward(self, x_dis, x_time):
        z, mu, logvar = self.encoder(x_dis, x_time)
        x_hat_dis, x_hat_time = self.decoder(z)
        
        return x_hat_dis, x_hat_time, mu, logvar, z
    
    def loss_function(self, dis_output, time_output, target, mu, logvar, dis_dim = 15):
        dis_target = target[:, :dis_dim]
        time_target = target[:, dis_dim:]

        dis_loss = self.CELoss(dis_output, dis_target).sum(-1)
        time_loss = self.MSELoss(time_output, time_target).sum(-1)
        
        reconstruction_loss = dis_loss + time_loss

        kl_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim = 1)
 

        return (reconstruction_loss + self.beta*kl_divergence), reconstruction_loss, kl_divergence, dis_loss, time_loss
     

def auxiliary_loss(time_output):
    overshoot = time_output > 1
    undershoot  = time_output < 0
    undershoot_loss = torch.sum(torch.square(torch.where(undershoot, time_output, torch.zeros_like(time_output))), dim = 1)
    overshoot_loss = torch.sum(torch.square(torch.where(overshoot, time_output - 1.0, torch.zeros_like(time_output))), dim = 1)
    
    
    
    age_start = time_output[:, 0]
    age_stop = time_output[:, 1]
    non_interval_loss = torch.square(torch.max(torch.zeros_like(age_start), age_start - age_stop))
    
    return overshoot_loss, undershoot_loss, non_interval_loss
    
   
def validation_with_aux(model, testloader, cuda, w_a, w_aux, ndis):
    test_loss = 0
    test_recon = 0
    test_kl = 0
    test_dis = 0
    test_time = 0
    test_under = 0
    test_over = 0
    test_non_interval = 0
    test_unw = 0
    
    report_num_samples = 0
    for x in testloader:
        batch_size, _ = x.size()
        report_num_samples += batch_size
        
        if cuda:
            x = x.cuda()
        
        x_hat_dis, x_hat_time, mu, logvar, z = model(x[:,:ndis], x[:,ndis:])
        loss, reconstruction_loss, kl_divergence, dis_loss,  time_loss = model.loss_function(x_hat_dis, x_hat_time, x, mu, logvar)
        overshoot_loss, undershoot_loss, non_interval_loss = auxiliary_loss(x_hat_time)
        
        reconstruction_loss = reconstruction_loss.sum()
        kl_divergence  = kl_divergence.sum()
        dis_loss = dis_loss.sum()
        time_loss = time_loss.sum()
        overshoot_loss =  overshoot_loss.sum()
        undershoot_loss = undershoot_loss.sum()
        non_interval_loss = non_interval_loss.sum()
        
        test_loss += w_a*time_loss.item() + dis_loss.item() + model.beta*kl_divergence.item() + w_aux*(overshoot_loss.item() + undershoot_loss.item() + non_interval_loss.item())
        test_unw += time_loss.item() + dis_loss.item() + kl_divergence.item() + overshoot_loss.item() + undershoot_loss.item() + non_interval_loss.item()
        test_kl += kl_divergence.item()
        test_recon += reconstruction_loss.item()
        test_dis += dis_loss.item()
        test_time += time_loss.item()
        test_under += undershoot_loss.item()
        test_over += overshoot_loss.item()
        test_non_interval += non_interval_loss.item()
        
    
    test_loss = test_loss / report_num_samples
    test_unw = test_unw / report_num_samples
    test_recon = test_recon / report_num_samples
    test_kl = test_kl / report_num_samples
    test_dis  = test_dis / report_num_samples
    test_time = test_time / report_num_samples
    test_under = test_under  / report_num_samples
    test_over  = test_over / report_num_samples
    test_non_interval = test_non_interval / report_num_samples
    
    return test_loss, test_recon, test_kl, test_dis, test_time, test_under, test_over, test_non_interval, test_unw

def train_new(args, print_every = 1200, warm_up = 10, kl_start = 1,  decay_epoch = 10, clip_grad = 5, lr_decay = 0.5, max_decay = 5, load_best_epoch  = 15):
    
    model = args["model"]
    trainloader =  args["trainloader"]
    testloader = args["testloader"]
    cuda = args["cuda"]
    epochs = args["epochs"]
    w_a =  args["w_a"]
    w_aux  = args["w_aux"]
    writer = args["writer"]
    ndis = args["ndis"] 
    
    opt_dict = {"not improved": 0, "lr": args['lr'], 'best_loss': 1e4}
    best_loss = 1e4
    kl_weight = kl_start
    
    if warm_up > 0:
        anneal_rate = (1.0 - kl_start) / (warm_up * len(trainloader))
    else:
        anneal_rate =  0
    
    optimizer = optim.Adam(model.parameters(), lr = opt_dict["lr"])
    decay_cnt = 0
    
    val_loss = []
    val_recon_loss = []
    val_kl = []
    val_dis_loss = []
    val_time_loss = []
    val_over_loss = []
    val_under_loss = []
    val_unw_loss = []
    val_non_interval_loss = []
    n_total_steps = len(trainloader)
    for e in range(epochs):
        
        model.train()
        running_loss = 0
        running_recon = 0
        running_kl = 0
        running_dis = 0
        running_time = 0
        running_over = 0
        running_under = 0
        running_non_interval = 0
        running_loss_unweighed = 0
        report_num_samples = 0
        
        for i, x in enumerate(trainloader):
            
            batch_size, _ = x.size()
            report_num_samples += batch_size
            
            if cuda:
                x = x.cuda()
            
            kl_weight = min(model.beta, kl_weight + anneal_rate)
            
            optimizer.zero_grad()
            
            x_hat_dis, x_hat_time, mu, logvar, z = model(x[:,:ndis], x[:,ndis:])
            
            loss, reconstruction_loss, kl_divergence, dis_loss, time_loss = model.loss_function(x_hat_dis, x_hat_time, x, mu, logvar)
            overshoot_loss, undershoot_loss, non_interval_loss = auxiliary_loss(x_hat_time)
            
            tot_loss = w_a*time_loss + dis_loss + kl_weight*kl_divergence + w_aux*(overshoot_loss + undershoot_loss + non_interval_loss)
            tot_loss = tot_loss.mean(dim=-1)
            tot_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            reconstruction_loss = reconstruction_loss.sum()
            kl_divergence  = kl_divergence.sum()
            dis_loss = dis_loss.sum()
            time_loss = time_loss.sum()
            overshoot_loss  = overshoot_loss.sum()
            undershoot_loss = undershoot_loss.sum()
            non_interval_loss = non_interval_loss.sum()
            
            optimizer.step()
            
            running_loss += w_a*time_loss.item() + dis_loss.item() + model.beta*kl_divergence.item() + w_aux*(overshoot_loss.item() + undershoot_loss.item() + non_interval_loss.item())
            running_loss_unweighed += time_loss.item() + dis_loss.item() + kl_divergence.item() + overshoot_loss.item() + undershoot_loss.item() + non_interval_loss.item()
            running_recon += reconstruction_loss.item()
            running_kl += kl_divergence.item()
            running_dis += dis_loss.item()
            running_time += time_loss.item()
            running_under += undershoot_loss.item()
            running_over += overshoot_loss.item()
            running_non_interval += non_interval_loss.item()         
            if (i+1) % print_every == 0:
                writer.add_scalar('Total Loss/train', running_loss / report_num_samples, e * n_total_steps + i)
                writer.add_scalar('Total Loss/train unweighed', running_loss_unweighed / report_num_samples, e*n_total_steps + i)
                writer.add_scalar('KL-Divergence/train', running_kl / report_num_samples, e * n_total_steps + i)
                writer.add_scalar('Reconstruction Loss/train', running_recon / report_num_samples, e * n_total_steps + i)
                writer.add_scalar('Reconstruction Loss Portfolio/train', running_dis / report_num_samples, e * n_total_steps + i)
                writer.add_scalar('Reconstruction Loss Age/train', running_time / report_num_samples, e * n_total_steps + i)
                writer.add_scalar('Auxiliary Loss/train undershoot', running_under / report_num_samples, e*n_total_steps + i)
                writer.add_scalar('Auxiliary Loss/train overshoot', running_over / report_num_samples, e*n_total_steps + i)
                writer.add_scalar('Auxiliary Loss/train non interval', running_non_interval / report_num_samples, e*n_total_steps + i)
                writer.add_scalar('Beta/train', kl_weight, e*n_total_steps + i)
                writer.add_scalar('Learning rate/train', opt_dict["lr"], e*n_total_steps + i)
                running_loss = 0
                running_recon = 0
                running_kl = 0
                running_dis = 0
                running_time = 0
                running_over = 0
                running_under = 0
                running_non_interval = 0
                report_num_samples = 0
                running_loss_unweighed = 0
       
        # turn off gradients for validation to speed up:
        with torch.no_grad():
            # Model in inference mode, potential dropout off:
            model.eval()
            test_loss, test_recon, test_kl, test_dis, test_time, test_under, test_over, test_non_interval, test_unw = validation_with_aux(model, testloader, cuda, w_a, w_aux, ndis)
        
        val_loss.append(test_loss)
        val_recon_loss.append(test_recon)
        val_kl.append(test_kl)
        val_dis_loss.append(test_dis)
        val_time_loss.append(test_time)
        val_under_loss.append(test_under)
        val_over_loss.append(test_over)
        val_non_interval_loss.append(test_non_interval)
        val_unw_loss.append(test_unw)
        writer.add_scalar("Total Loss/validation", test_loss, e+1)
        writer.add_scalar("Total Loss/validation unweighed", test_unw, e+1)
        writer.add_scalar("KL-Divergence/validation", test_kl, e+1)
        writer.add_scalar("Reconstruction Loss/validation", test_recon, e+1)
        writer.add_scalar('Reconstruction Loss Portfolio/validation', test_dis, e+1)
        writer.add_scalar('Reconstruction Loss Age/validation', test_time, e+1)
        writer.add_scalar('Auxiliary Loss/validation undershoot', test_under, e+1)
        writer.add_scalar('Auxiliary Loss/validation overshoot', test_over, e+1)
        writer.add_scalar('Auxiliary Loss/validation non interval', test_non_interval, e + 1)
        writer.add_scalar('Beta/validation', kl_weight, e + 1)
        writer.add_scalar('Learning rate/validation', opt_dict["lr"], e + 1)
        
        
        if test_loss < best_loss:
            print("update best  loss")
            best_loss = test_loss
            torch.save(model.state_dict(), args['save_path'])
        
        if test_loss > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if  opt_dict["not_improved"] >= decay_epoch and e >= load_best_epoch:
                opt_dict["best_loss"] = test_loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                model.load_state_dict(torch.load(args['save_path']))
                print('new  lr: %f' %  opt_dict["lr"])
                decay_cnt += 1
                optimizer = optim.Adam(model.parameters(), lr = opt_dict["lr"])
        
        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = test_loss
        
        if decay_cnt == max_decay:
            break
        
        model.train()
                

    return {'val_loss': val_loss,
            'val_recon_loss': val_recon_loss,
            'val_kl': val_kl,
            'val_dis_loss': val_dis_loss,
            'val_time_loss': val_time_loss,
            'val_under_loss': val_under_loss,
            'val_over_loss': val_over_loss,
            'val_non_interval_loss': val_non_interval_loss,
            'val_unw_loss': val_unw_loss}
    

