
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm



class CausalityDataset(Dataset):
    def __init__(self, data_lineage, index_y, index_x, lags):
        T, num_species = data_lineage.shape
        self.X, self.Y, self.Y_future = [], [], []

        for t in range(lags, T):
            x_lag = data_lineage[t-lags:t, index_x].reshape(-1)
            y_lag = data_lineage[t-lags:t, index_y].reshape(-1)
            y_future = data_lineage[t, index_y]

            self.X.append(x_lag)
            self.Y.append(y_lag)
            self.Y_future.append(y_future)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.Y_future = torch.tensor(np.array(self.Y_future), dtype=torch.float32)

    def __len__(self):
        return len(self.Y_future)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Y_future[idx]

class CausalityDetector(nn.Module):
    def __init__(self, input_dim_x, time_lag, dim_y_input, dim_y_output):    
        super(CausalityDetector, self).__init__()

        self.n = time_lag
        self.ydim = dim_y_input
        self.feature_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 5),
                nn.LeakyReLU(),
                nn.Linear(5, 1)
            ) for _ in range(self.n * input_dim_x)
        ])
        self.feature_mlps_y = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 5),
                nn.LeakyReLU(),
                nn.Linear(5, 1)
            ) for _ in range(self.n)
        ])
        self.B = nn.Parameter(
            torch.randn(self.n * dim_y_input + self.n * input_dim_x, dim_y_output)
        )

    def forward(self, input_Y_list, input_X_list):

        projected = []
        for i, mlp in enumerate(self.feature_mlps):
            feature = input_X_list[:, i].view(-1, 1)       
            projected.append(mlp(feature))  
        projected = torch.cat(projected, dim=1)  

        projected_y = [] 
        for i, mlp in enumerate(self.feature_mlps_y):
            feature = input_Y_list[:, i].view(-1, 1)       
            projected_y.append(mlp(feature))  
        projected_y = torch.cat(projected_y, dim=1)
          
        inp = torch.cat([projected_y, projected], dim=1) 


        # linear predictor
        y_hat = torch.matmul(inp, self.B)  
        return y_hat

def train_causality_model(
    data_lineage,
    index_y,
    index_x,
    device,
    time_lag=10,
    l1_lambda=5,
    lr=0.01,
    batch_size=100,
    num_epochs=1500,
    differencing = True):

    """
    Granger causality test to assess whether species X Granger-cause species Y.

    Args:
        data_lineage:
            Species profile array for lineage of interest. The columns correspond to features/species and the
            first dimension is ordered in growth stage.

        index_x:
            Indices of the predictor species (X). 

        index_y:
            Indices of the response species (Y). 

        device:
            Computing device, typically "cpu" or "cuda".

        time_lag:
            Number of past time steps (lags) included in the model.

        l1_lambda:
            Strength of the L1 sparsity penalty applied to the X→Y coefficients.
            Larger values yield sparser solutions.

        lr:
            Optimizer learning rate. Default: 0.01.

        batch_size:
            Mini-batch size for training. Default: 100.

        num_epochs:
            Number of training epochs. Default: 1500.
    Returns:
        B:
            NumPy array of learned X→Y coefficients with shape
            (time_lag * |X|, |Y|). Larger magnitudes indicate stronger predictive
            influence of lagged X on current Y.
    """
    if differencing:
        first_diff = (data_lineage[1:]-data_lineage[:-1])
        input = first_diff[1:]-first_diff[:-1]
    else:
        input = data_lineage


    # dataset
    gc_dataset = CausalityDataset(input, index_y, index_x, time_lag)
    dataloader = DataLoader(gc_dataset, batch_size=batch_size, shuffle=True)

    input_dim_x = len(index_x)
    dim_y_input = len(index_y)
    dim_y_output = len(index_y)

    # model
    model = CausalityDetector(
        input_dim_x=input_dim_x,
        time_lag=time_lag,
        dim_y_input=dim_y_input,
        dim_y_output=dim_y_output
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0
    )
    criterion = nn.MSELoss()

    # training loop
    pbar = tqdm(range(num_epochs), desc="Training", dynamic_ncols=True)
    for epoch in pbar:
        mse_loss = 0
        l1_loss = 0
        num_samples = 0
        model.train()

        for input_X, input_Y, output_Y in dataloader:
            input_X = input_X.to(device)
            input_Y = input_Y.to(device)
            output_Y = output_Y.to(device)

            Y_pred = model(input_Y, input_X)

            B = model.B
            l1_penalty_x = B[(time_lag*dim_y_input):,:].abs().mean()
            l1_penalty_y = B[:(time_lag*dim_y_input),:].abs().mean()
            mse_loss_item = criterion(Y_pred, output_Y)
            loss = mse_loss_item + l1_lambda * l1_penalty_x + l1_penalty_y

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mse_loss += mse_loss_item
            l1_loss += l1_lambda * l1_penalty_x
            num_samples += output_Y.size(0)

        pbar.set_postfix({
            "MSE-Loss": f"{mse_loss/num_epochs:.4f}",
            "L1-Loss": f"{l1_loss/num_samples:.4f}"        
            })
    B = model.B.cpu().detach().numpy()[(time_lag*dim_y_input):,:]

    return B

#################################################################################################plot

def plot_gc_heatmap(
    B_list,
    x_labels=None,
    y_labels=None,
    figsize=(5,3),
    file_path=None):

    '''
    Plot of granger causality.

    Args:

        B_list: Concatenated outputs of train_causality_model.

        x_labels: Species list for predictor.

        y_labels: Species list for target species.

        figsize: Size of output figure.

        file_path: Output path to save the figure.
    '''

    M = np.max(abs(B_list), axis=1)

 
    plt.figure(figsize=figsize)
    sns.heatmap(
        M, cmap="Reds",
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar_kws={"label": "GC strength"}
    )
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=7, rotation=0)
    plt.title("Granger Causality", fontsize=9)
    plt.xlabel("Predictor", fontsize=9)
    plt.ylabel("Target", fontsize=9)
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path, dpi=300)
    plt.show()

