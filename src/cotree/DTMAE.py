import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR



import os
import pandas as pd
from tqdm import tqdm



################################################################################DT-MAE

def relu_block(input_size, output_size, dropout = 0.1):
    return nn.Sequential(nn.Linear(input_size, output_size),
            nn.LayerNorm(output_size),
            nn.LeakyReLU(0.05),
            nn.Dropout(dropout))



class FCLayer(nn.Module):
    def __init__(self, layer_dim, dropout_rate):
        super(FCLayer,self).__init__()
        self.fc_layer = nn.Sequential()
        for i in range(len(layer_dim)-1):
            self.fc_layer.add_module(f"block_{i}",relu_block(layer_dim[i],layer_dim[i+1],dropout_rate))

    def forward(self, x):
        output = self.fc_layer(x)
        return output
    
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_layer, outputsize, dropout_rate):
        super(Encoder,self).__init__()
        self.G_encoder = FCLayer([input_size['G']] + hidden_layer['G'] + [outputsize['G']], dropout_rate)
        self.M_encoder = FCLayer([input_size['M']] + hidden_layer['M'] + [outputsize['M']], dropout_rate)
        self.P_encoder = FCLayer([input_size['P']] + hidden_layer['P'] + [outputsize['P']], dropout_rate)
        self.batch_norm2 = nn.LayerNorm(input_size['M'])
        self.batch_norm3 = nn.LayerNorm(input_size['P'])
    
    def forward(self, G, M, P):
        x1_embed = self.G_encoder(G)
        x2_embed = self.M_encoder(self.batch_norm2(M))
        x3_embed = self.P_encoder(self.batch_norm3(P))

        all_embedding = torch.cat((x1_embed,x2_embed,x3_embed),dim=1)
        return all_embedding, x1_embed, x2_embed, x3_embed

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_layer, outputsize, dropout_rate):
        super(Decoder,self).__init__()
        self.G_decoder = FCLayer([input_size['G']] + hidden_layer['G'] + [outputsize['G']], dropout_rate)
        self.M_decoder = FCLayer([input_size['M']] + hidden_layer['M'] + [outputsize['M']], dropout_rate)
        self.P_decoder = FCLayer([input_size['P']] + hidden_layer['P'] + [outputsize['P']], dropout_rate)
      
    def forward(self, G_embed, M_embed, P_embed):
        G = self.G_decoder(G_embed)
        M = self.M_decoder(M_embed)
        P = self.P_decoder(P_embed)
        return G, M, P

class Predictor(nn.Module):
    def __init__(self, input_size, hidden_layer, outputsize, dropout_rate):
        super(Predictor,self).__init__()
        self.G_decoder = FCLayer([input_size['G']+input_size['M']+input_size['P']] + hidden_layer['G'] + [outputsize['G']], dropout_rate)
        self.M_decoder = FCLayer([input_size['M']+input_size['P']] + hidden_layer['M'] + [outputsize['M']], dropout_rate)
        self.P_decoder = FCLayer([input_size['R']+input_size['M']+input_size['P']] + hidden_layer['P'] + [outputsize['P']], dropout_rate)

    def forward(self, G_curr, M_curr, P_curr, R_curr):        
        G = self.G_decoder(torch.concat((G_curr,M_curr,P_curr),dim = -1))
        M = self.M_decoder(torch.concat((M_curr,P_curr),dim = -1))
        P = self.P_decoder(torch.concat((R_curr,M_curr,P_curr),dim = -1))

        return G, M, P
    

class autoencoder(nn.Module):
    def __init__(self, input_size, hidden_layer_encoder, hidden_layer_decoder, hidden_layer_predictor, latent_dim, outputsize, dropout_rate):
        super(autoencoder,self).__init__()
        self.encoder = Encoder(input_size, hidden_layer_encoder, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, hidden_layer_decoder, outputsize, dropout_rate)
        self.predictor = Predictor(latent_dim, hidden_layer_predictor, input_size, dropout_rate)
    
    def forward(self, G, M, P, R):
        all_embedding, G_embed, M_embed, P_embed = self.encoder(G, M, P)
        G_curr, M_curr, P_curr = self.decoder(G_embed, M_embed, P_embed)
        G_pred, M_pred, P_pred = self.predictor(G_embed, M_embed, P_embed, R)
        recover_vec = torch.concat((G_curr, M_curr, P_curr),dim = -1)
        pred_vec = torch.concat((G_pred, M_pred, P_pred),dim = -1)
        return pred_vec, recover_vec, G_embed, M_embed, P_embed, all_embedding


################################################################################Dataset preparation

class OmicsDataset(Dataset):
    """
    Load data.
    Args:
        data_folder: path of dataset folder. 
        labels: list of subfolders contains trajectories.
    """
    def __init__(self, data_folder, labels):
        self.data_folder = data_folder
        self.labels = labels
        self.names = {}
        self.data = {}
        self.cell_label = []
        if not self.data:
            self.load_datasets()
        self.samples = self.create_pairs()
        

    def load_datasets(self):        
        for label in tqdm(self.labels, desc='Loading datasets'):
            for root, dirs, files in os.walk(os.path.join(self.data_folder, label)):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    csv_files = [file for file in os.listdir(dir_path) if file.endswith('.csv')]
                    if not csv_files:
                        continue
                    for file_name in tqdm(csv_files, desc=f"Processing {label}, Loading files in {dir_name}"):
                        file_path = os.path.join(dir_path, file_name)
                        df = pd.read_csv(file_path, index_col=0)
                        self.names[dir_name] = df.index
                        idx = int(file_path.split('.')[len(file_path.split('.')) - 2])
                        cell_name = label+str(idx)
                        if cell_name in self.data.keys():
                            self.data[cell_name][dir_name] = df
                        else:
                            self.data[cell_name] = {}
                            self.data[cell_name][dir_name] = df
                            self.cell_label.append(label)

    def create_pairs(self):
        samples = []
        i = 0
        for idx, datasets in tqdm(self.data.items(), desc="Preparing samples"):
            dna = torch.tensor(datasets['genomics'].to_numpy().T,dtype=torch.float32)
            rna = torch.tensor(datasets['transcriptomics'].to_numpy().T,dtype=torch.float32)
            protein = torch.tensor(datasets['proteomics'].to_numpy().T,dtype=torch.float32)
            metabolome = torch.tensor(datasets['metabolomics'].to_numpy().T,dtype=torch.float32)
            samples.append((dna,rna,protein,metabolome,idx, self.cell_label[i]))
            i = i+1
        return samples        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.cell_label[idx]
        

class training_dataset():
    """
    Prepare training data.
    Args:
        dataset: samples from OmicsDataset object.
        filter_idx: feature index that are not included in analysis e.g.:
            `{'M': [], 'P': [0,1], 'G': [], 'R': []}`.
    """

    def __init__(self, dataset, filter_idx):
        self.dataset = dataset
        self.scale_meta = 1
        self.scale_protein = 1
        self.scale_dna = 1
        self.scale_rna = 1
        self.filter_idx = filter_idx
        self.keep_ind = {}
        self.samples = self.create_pairs()
        


    def create_pairs(self):
        samples = []
        cell_name = []
        for i in range(len(self.dataset)):
            G_all, R_all, P_all, M_all, cellidx, label = self.dataset[i]
            G = G_all[:-1,:]
            P = P_all[:-1,:]
            R = R_all[:-1,:]
            M = M_all[:-1,:]
            G_pred = G_all[1:,:]
            M_pred = M_all[1:,:]
            P_pred = P_all[1:,:]
            R_pred = R_all[1:,:]
            if i == 0:
                G_tensor = G
                R_tensor = R
                P_tensor = P
                M_tensor = M
                G_pred_tensor = G_pred
                M_pred_tensor = M_pred
                P_pred_tensor = P_pred
                R_pred_tensor = R_pred
                
            else:
                G_tensor = torch.cat((G_tensor, G), 0)
                R_tensor = torch.cat((R_tensor, R), 0)
                P_tensor = torch.cat((P_tensor, P), 0)
                M_tensor = torch.cat((M_tensor, M), 0)
                G_pred_tensor = torch.cat((G_pred_tensor, G_pred), 0)
                M_pred_tensor = torch.cat((M_pred_tensor, M_pred), 0)
                P_pred_tensor = torch.cat((P_pred_tensor, P_pred), 0)
                R_pred_tensor = torch.cat((R_pred_tensor, R_pred), 0)
                cell_name = cell_name + [cellidx]*G_tensor.shape[0]

        def filter_and_scale_features(pred_tensor, curr_tensor, extra_excluded_idx=None, weight = 1):

            zero_variances = torch.var(torch.cat((pred_tensor, curr_tensor), dim=0), dim=0)
            nonzero_mask = zero_variances > 0

            extra_mask = torch.ones_like(nonzero_mask, dtype=torch.bool)
            extra_mask[extra_excluded_idx] = False

            keep_mask = nonzero_mask & extra_mask
            pred_filtered = pred_tensor[:, keep_mask]
            curr_filtered = curr_tensor[:, keep_mask]


            scale_para = torch.clamp(torch.mean(pred_filtered), min=1e-8)*weight
            pred_filtered = pred_filtered / scale_para
            curr_filtered = curr_filtered / scale_para

            return pred_filtered, curr_filtered, keep_mask, scale_para
        
        M_pred_tensor, M_tensor, M_keep_indices, self.scale_meta = filter_and_scale_features(M_pred_tensor, M_tensor, self.filter_idx['M'])
        self.keep_ind['M'] = M_keep_indices

        R_pred_tensor, R_tensor, R_keep_indices, self.scale_rna = filter_and_scale_features(R_pred_tensor, R_tensor, self.filter_idx['R'])
        self.keep_ind['R'] = R_keep_indices

        G_pred_tensor, G_tensor, G_keep_indices, self.scale_dna = filter_and_scale_features(G_pred_tensor, G_tensor, self.filter_idx['G'], weight = 0.5)
        self.keep_ind['G'] = G_keep_indices

        P_pred_tensor, P_tensor, P_keep_indices, self.scale_protein = filter_and_scale_features(P_pred_tensor, P_tensor, self.filter_idx['P'], weight = 0.5)
        self.keep_ind['P'] = P_keep_indices
              
        
        for j in range(G_tensor.shape[0]):
            samples.append((G_tensor[j,:], M_tensor[j,:], P_tensor[j,:], R_tensor[j,:], 
                            torch.cat((G_pred_tensor[j,:],M_pred_tensor[j,:],P_pred_tensor[j,:]),dim = -1),
                            torch.cat((G_tensor[j,:],M_tensor[j,:],P_tensor[j,:]),dim = -1)))
        return samples

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        G, M, P, R, pred_vec, curr_vec = self.samples[idx]
        return G, M, P, R, pred_vec, curr_vec
        

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        G, M, P, R, pred_vec, curr_vec = self.samples[idx]
        return G, M, P, R, pred_vec, curr_vec


def stacked_dataset(data, max_meta, max_protein, max_dna, max_rna, keep_index):
    i = 0
    for idx, datasets in data.items():
        G = torch.tensor(datasets['genomics'].to_numpy().T,dtype=torch.float32)
        R = torch.tensor(datasets['transcriptomics'].to_numpy().T,dtype=torch.float32)
        P = torch.tensor(datasets['proteomics'].to_numpy().T,dtype=torch.float32)
        M = torch.tensor(datasets['metabolomics'].to_numpy().T,dtype=torch.float32)
        if i == 0:
            G_tensor = G
            R_tensor = R
            P_tensor = P
            M_tensor = M
        else:
            G_tensor = torch.cat((G_tensor, G), 0)
            R_tensor = torch.cat((R_tensor, R), 0)
            P_tensor = torch.cat((P_tensor, P), 0)
            M_tensor = torch.cat((M_tensor, M), 0)
        i = i+1
    M_tensor = M_tensor[:,keep_index['M']]
    M_tensor = M_tensor/max_meta
    P_tensor = P_tensor[:,keep_index['P']]
    P_tensor = P_tensor/max_protein
    G_tensor = G_tensor[:,keep_index['G']]
    G_tensor = G_tensor/max_dna
    R_tensor = R_tensor[:,keep_index['R']]
    R_tensor = R_tensor/max_rna
    return G_tensor, M_tensor, P_tensor, R_tensor

################################################################################Training

def seperate_output(output, input_size):
    G = output[:,:input_size["G"]]
    M = output[:,input_size["G"]:(input_size["G"]+input_size["M"])]
    P = output[:,(input_size["G"]+input_size["M"]):]
    R = torch.rand(input_size["R"])
    return G, M, P, R


def train_autoencoder_model(
    data,
    device: torch.device,
    *,
    input_size,
    hidden_layer_encoder,
    hidden_layer_decoder,
    hidden_layer_predictor,
    latent_dim,
    epochs = 200,
    batch_size = 200,
    lr  = 5e-3,
    step_size = 200,
    gamma = 0.5,
    weight_curr ={"G": 2.0, "M": 1.0, "P": 2.0},  # weights for current loss
    weight_pred = {"G": 2.0, "M": 1.0, "P": 2.0},  # weights for future/pred loss
    print_every = 10,
    dropout_rate = 0,
    ratio_train_test = 0.99
    ):

    """
    Train the autoencoder.

    Args:
        data: 
            Output from the `training_dataset` function.

        device: 
            The computing device to use — typically `'cpu'` or `'cuda'` (GPU).

        input_size: 
            Dictionary specifying the input dimensions of each omics layer, e.g.:
            `{'M': 181, 'P': 458, 'R': 455, 'G': 493}`.

        hidden_layer_encoder: 
            Dictionary specifying the encoder hidden layer sizes for each omics layer, e.g.:
            `{'M': [100], 'P': [250], 'R': [250], 'G': [100]}`.

        hidden_layer_decoder: 
            Dictionary specifying the decoder hidden layer sizes for each omics layer, 
            using the same format as `hidden_layer_encoder`.

        hidden_layer_predictor: 
            Dictionary specifying the predictor hidden layer sizes for each omics layer,
            e.g. `{'M': [100], 'P': [250], 'R': [250], 'G': [100]}`.

        latent_dim: 
            Dictionary specifying the latent dimension size for each omics layer, e.g.:
            `{'M': 30, 'P': 50, 'G': 10, 'R': 455}`.

        epochs (int, optional): 
            Number of training epochs. Default is `200`.

        batch_size (int, optional): 
            Mini-batch size for training and validation. Default is `200`.

        lr (float, optional): 
            Learning rate for the optimizer. Default is `0.005`.

        step_size (int, optional): 
            Number of epochs between learning rate decays in the StepLR scheduler. Default is `200`.

        gamma (float, optional): 
            Learning rate decay factor. Default is `0.5`.

        weight_curr (dict[str, float], optional): 
            Weight coefficients for current-step reconstruction losses across modalities.  
            Example: `{'G': 2.0, 'M': 1.0, 'P': 2.0}`. Default is `{'G': 2.0, 'M': 1.0, 'P': 2.0}`.

        weight_pred (dict[str, float], optional): 
            Weight coefficients for predicted-step reconstruction losses across modalities.  
            Example: `{'G': 2.0, 'M': 1.0, 'P': 2.0}`. Default is `{'G': 2.0, 'M': 1.0, 'P': 2.0}`.

        print_every (int, optional): 
            Interval (in epochs) for printing loss information. Default is `10`.

        dropout_rate (float, optional): 
            Dropout probability applied in the model to reduce overfitting. Default is `0.0`.

    Returns:
        model (torch.nn.Module): 
            Trained autoencoder model.
    """


    total_samples = data.__len__()
    train_size = int(ratio_train_test * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    # Model
    model = autoencoder(
        input_size,
        hidden_layer_encoder,
        hidden_layer_decoder,
        hidden_layer_predictor,
        latent_dim,
        input_size,
        dropout_rate,
    ).to(device)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer/scheduler/loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = nn.MSELoss()

    pbar = tqdm(range(epochs), desc="Training", dynamic_ncols=True)


    for epoch in pbar:
        # ---- Training ----
        model.train()
        train_loss_total = 0.0
        for G, M, P, R, pred_target, curr_target in train_loader:
            G = G.to(device); M = M.to(device); P = P.to(device); R = R.to(device)
            pred_target = pred_target.to(device)
            curr_target = curr_target.to(device)

            pred_output, curr_output, *_ = model(G, M, P, R)
            G_pred, M_pred, P_pred, R_pred = seperate_output(pred_output, input_size)
            G_tgt,  M_tgt,  P_tgt,  R_tgt  = seperate_output(pred_target, input_size)
            G_curr, M_curr, P_curr, R_curr = seperate_output(curr_output, input_size)
            G_tgt_c, M_tgt_c, P_tgt_c, R_tgt_c = seperate_output(curr_target, input_size)

            loss_curr = (
                weight_curr.get("G", 1.0) * loss_fn(G_tgt_c, G_curr) +
                weight_curr.get("M", 1.0) * loss_fn(M_tgt_c, M_curr) +
                weight_curr.get("P", 1.0) * loss_fn(P_tgt_c, P_curr)
            )
            loss_pred = (
                weight_pred.get("G", 1.0) * loss_fn(G_tgt, G_pred) +
                weight_pred.get("M", 1.0) * loss_fn(M_tgt, M_pred) +
                weight_pred.get("P", 1.0) * loss_fn(P_tgt, P_pred)
            )
            loss = loss_curr + loss_pred

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        scheduler.step()
        train_loss_avg = train_loss_total / max(1, len(train_loader))

        # ---- Validation ----
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for G, M, P, R, pred_target, curr_target in val_loader:
                G = G.to(device); M = M.to(device); P = P.to(device); R = R.to(device)
                pred_target = pred_target.to(device)
                curr_target = curr_target.to(device)

                pred_output, curr_output, *_ = model(G, M, P, R)
                G_pred, M_pred, P_pred, R_pred = seperate_output(pred_output, input_size)
                G_tgt,  M_tgt,  P_tgt,  R_tgt  = seperate_output(pred_target, input_size)
                G_curr, M_curr, P_curr, R_curr = seperate_output(curr_output, input_size)
                G_tgt_c, M_tgt_c, P_tgt_c, R_tgt_c = seperate_output(curr_target, input_size)

                loss_curr = (
                    weight_curr.get("G", 1.0) * loss_fn(G_tgt_c, G_curr) +
                    weight_curr.get("M", 1.0) * loss_fn(M_tgt_c, M_curr) +
                    weight_curr.get("P", 1.0) * loss_fn(P_tgt_c, P_curr)
                )
                loss_pred = (
                    weight_pred.get("G", 1.0) * loss_fn(G_tgt, G_pred) +
                    weight_pred.get("M", 1.0) * loss_fn(M_tgt, M_pred) +
                    weight_pred.get("P", 1.0) * loss_fn(P_tgt, P_pred)
                )
                val_loss_total += (loss_curr + loss_pred).item()

        val_loss_avg = val_loss_total / max(1, len(val_loader))

        pbar.set_postfix({
            "train_loss": f"{train_loss_avg:.4f}",
            "val_loss": f"{val_loss_avg:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.3e}"
        })

    return model


def RL_output(model, dataset, training_data, keep_index, device):
    G, M, P, R = stacked_dataset(dataset.data, training_data.scale_meta, training_data.scale_protein, training_data.scale_dna, training_data.scale_rna, keep_index) 
    model.eval()
    with torch.no_grad():
        G = G.to(device)
        R = R.to(device)
        P = P.to(device)
        M = M.to(device)
        pred_output, curr_output, G_embed, M_embed, P_embed, latent_embedding = model(G,M,P,R)
    latent_embedding = latent_embedding.to('cpu').numpy()
    M_stack = M.to('cpu').numpy()
    P_stack = P.to('cpu').numpy()
    G_stack = G.to('cpu').numpy()
    output = {"Latent representation": latent_embedding, "Raw metabolite": M_stack, "Raw protein": P_stack, "Raw gene": G_stack}
    return output


def reconstruct_lineage(trajectory_tree, model, latent_dim, device, interval = 3):
    resample_lineage_species = {}
    for leaf in trajectory_tree.keys():
        G_embed, M_embed, P_embed, R = seperate_output(trajectory_tree[leaf], latent_dim)
        G_embed = torch.tensor(G_embed, dtype=torch.float32).to(device)
        M_embed = torch.tensor(M_embed,dtype=torch.float32).to(device)
        P_embed = torch.tensor(P_embed,dtype=torch.float32).to(device)
        model.eval()
        G_curr, M_curr, P_curr = model.decoder(G_embed, M_embed, P_embed)
        lineage_embed = torch.concat((G_curr, M_curr, P_curr), dim=-1)
        curve = lineage_embed.cpu().detach().numpy()
        resample_lineage_species[leaf] = curve[::interval]
    return resample_lineage_species