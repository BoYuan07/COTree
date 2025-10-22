import numpy as np
import pandas as pd
import torch
import matplotlib.patches as mpatches
from kneed import KneeLocator
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm


from cotree.DTMAE import seperate_output


##################################################################################### helper functions 
def bifurcation_data(trajectory_tree, seperation_time):
    seperate_data = {}
    include_lineage = list(trajectory_tree.keys())
    for leaf, time in seperation_time.items():
        include_lineage.remove(leaf)
        seperate_data[leaf] = include_lineage.copy()
    return seperate_data


def extract_trajectory(trajectory_tree,
                       growth_stage_matrix,
                       data_raw,
                       flatten_embeddings,
                       clustering_labels,
                       seperation_time,
                       list_truncation=80,
                       inter = 13):
    
    """
    Extract trajectory for driver species detection.

    Args:
        trajectory_tree: Trajectory tree.

        growth_stage_matrix: Growth stage matrix.

        data_raw: Concatenated cell profiles. Accepts NumPy array.

        flatten_embeddings: Concatenated cell embeddings. Accepts NumPy array.

        clustering_labels: Clustering labels of cells.

        seperation_time: Dictionary that includes seperation time for each lineage.
    
    Outputs:
        Inputs for driver species detection.
    """
    
    sorted_bifurcation_pairs = dict(sorted(seperation_time.items(), key=lambda item: item[1]))
    seperate_data = bifurcation_data(trajectory_tree, sorted_bifurcation_pairs)


    def _extract_data_for_lineage(node_index, node_time, lineages, direction, leaf, inter):

        """Helper function to extract trajectory data for a given lineage."""
        raw_data[node_index][direction] = []
        encoded_data[node_index][direction] = []
        weight[node_index][direction] = []
        real_time[node_index][direction] = []

    
        # Extract relevant data for each cell in the lineage
        for lineage in lineages:
            for cell_index, cell_time in enumerate(growth_stage_matrix):
                if clustering_labels[cell_index] != lineage:
                    continue
                
                times = cell_time[:, np.where(leaf == lineage)[0][0]]

                point_time_indices = np.where(times > node_time)[0]
                if len(point_time_indices) == 0:
                    print(f"Warning: No valid time point found for cell {cell_index} in lineage {lineage}")
                    continue

                point_time = point_time_indices[0]
                node_position = flatten_embeddings[cell_index][point_time,:]

                truncation = min(list_truncation, data_raw[cell_index].shape[0])

                raw_data[node_index][direction].append(data_raw[cell_index][point_time:truncation:inter])
                encoded_data[node_index][direction].append(flatten_embeddings[cell_index][point_time:truncation:inter])

                # delta lambda
                weight[node_index][direction].append(
                    times[point_time:truncation:inter]-node_time
                )

                real_time[node_index][direction].append(point_time)

        # Convert lists to arrays
        if raw_data[node_index][direction]:
            raw_data[node_index][direction] = np.concatenate(raw_data[node_index][direction], axis=0)
            encoded_data[node_index][direction] = np.concatenate(encoded_data[node_index][direction], axis=0)
            weight[node_index][direction] = np.concatenate(weight[node_index][direction], axis=0)


    # Initialize output dictionaries
    raw_data, encoded_data, weight, real_time = {}, {}, {}, {}
    leaf = np.array(list(trajectory_tree.keys()))
    for node_index, (lineage1, lineage2) in enumerate(seperate_data.items()):
        raw_data[node_index], encoded_data[node_index] = {}, {}
        weight[node_index], real_time[node_index] = {}, {}
        node_time = seperation_time[lineage1]
        lineage1 = [lineage1]

        _extract_data_for_lineage(node_index, node_time, lineage1, "direction 1", leaf, inter)
        _extract_data_for_lineage(node_index, node_time, lineage2, "direction 2", leaf, inter)

    return raw_data, encoded_data, weight, real_time


def optimize_a_b(model, X_list, Z_list, a, b, w_list, lambda_reg, input_size, device, learning_rate=0.01, 
                 epochs=100, threshold = 0.5):
    
    optimizer = torch.optim.Adam([a, b], lr=learning_rate)
    pbar = tqdm(range(epochs), desc="Training", dynamic_ncols=True)

    for epoch in pbar:
        optimizer.zero_grad()  
        # Initialize loss function
        loss = 0  
        k = b.shape[0]  
        cosine = []

        for j in range(k+1):
           
            X = X_list[j]
            Z = Z_list[j]

            # Truncate weights
            q = w_list[j][:-1]
            q[q>threshold] = 0
            N, d = X.shape 

            # Compute direction vectors
            delta_X = X[1:, :] - X[:-1, :]  
            delta_Z = Z[1:, :] - Z[:-1, :] 

            v = a.unsqueeze(0).expand(N-1, -1) 

            # For non-reference lineages
            if j > 0:
                v = v.clone()
                v += b[j-1,:].unsqueeze(0).expand(N-1, -1)  
                loss += lambda_reg * torch.norm(b[j-1,:],1)


            # Compute projected vector
            proj = torch.sum(delta_X * v, dim=1, keepdim=True)  # Shape: (N-1, 1) 
            transformed_X = X[:-1, :] + proj * v  

            # Compute embeddings for projected vectors
            G, M, P, R = seperate_output(transformed_X, input_size) 
            G = G.to(device)
            M = M.to(device)
            P = P.to(device) 
            all_embedding, x1_embed, x2_embed, x3_embed = model.encoder(G, M, P)
            f_transformed_X = all_embedding.to(Z.device)
        
            # Compute estimated direction
            h = f_transformed_X - Z[:-1, :]  

            # Loss
            numerator = torch.sum(h * delta_Z, dim=1)  
            denom = torch.norm(h, dim=1) * torch.norm(delta_Z, dim=1) + 1e-8 
            valid_mask = (q != 0)
            loss_terms = torch.where(valid_mask, (numerator / denom), torch.tensor(0.0, device=X.device))
            loss -= torch.sum(loss_terms)/torch.sum(q!=0)
            cosine.append(torch.sum(loss_terms)/torch.sum(q!=0))
            
        loss.backward()
        optimizer.step()

        pbar.set_postfix({
            "Cosine (direction 1)": f"{cosine[0]:6f}",
            "Cosine (direction 2)": f"{cosine[1]:6f}",
            "regularization": f"{lambda_reg * torch.norm(b[0,:],1):6f}"
        })

    return a.detach(), b.detach()  # Return optimized values


def init_parameters(X_list, dist_w, threshold = 0.5):
    b_list = []
    filtered_data = []
    for j in range(len(X_list)):
        X1 = X_list[j][:-1,:]
        X2 = X_list[j][1:,:]
        w = dist_w[j][:-1]
        q = w
        diff = X2-X1
        if j == 0:
            a = torch.mean(diff[q <= threshold,:],dim=0)
        else:
            b_list.append(torch.mean(diff[q <= threshold,:],dim=0)-a)
        filtered_data.append(X2[q<=threshold]-X1[q <= threshold])

    a = a.detach().clone().requires_grad_(True)
    b = torch.tensor(np.array(b_list),requires_grad=True, dtype=torch.float)
    return a, b


def driver_species_detection(model,
                            node_ind, 
                            raw_data, 
                            encoded_data, 
                            weight, 
                            input_size,
                            device,
                            sparse_penalty = 0.01, 
                            epoches = 300, 
                            learning_rate = 0.001,
                            threshold = 0.5):

    """
    Driver species detection.

    Args:
        model: DT-MAE model

        node_ind: Node index for driver species detection.

        raw_data: Raw data output from extract_trajectory.

        encoded_data: Latent data output from extract_trajectory.

        weight: Weight output from extract_trajectory.

        input_size: Input size fro gene, metabolite, protein and mRNA. Dictionary.

        sparse_penalty: Parameter for l1 penalty.

        epoches: Number of epoches.

        learning_rate: Learning rate for optimization.

        threshold: Delta growth stage.
    
    Outputs:
        current_b: Differential scores.

        selected_index: Index for differential species.
    """
    
    
    X = [torch.tensor(raw_data[node_ind][i], dtype=torch.float) for i in raw_data[node_ind]]
    Z = [torch.tensor(encoded_data[node_ind][i], dtype=torch.float) for i in raw_data[node_ind]]
    d = X[node_ind].shape[1]
    w_list = [torch.tensor(weight[node_ind][i], dtype=torch.float) for i in weight[node_ind]]
    # Initialization
    a,b = init_parameters(X,w_list, threshold = threshold)
    # Optimize a and b
    optimized_a, optimized_b = optimize_a_b(model, X, Z, a, b, w_list, device= device, input_size = input_size, lambda_reg = sparse_penalty, 
                                            epochs = epoches, learning_rate=learning_rate, threshold = threshold)
    
    # choose optimal tau
    def compute_loss(b_vec, threshold = threshold):
        X2, Z2 = X[1], Z[1]
        q = w_list[1][:-1]
        q[q > threshold] = 0
        N, d = X2.shape

        delta_X = X2[1:, :] - X2[:-1, :]
        delta_Z = Z2[1:, :] - Z2[:-1, :]

        v = a.unsqueeze(0).expand(N-1, -1).clone()
        v += b_vec.unsqueeze(0).expand(N-1, -1)

        proj = torch.sum(delta_X * v, dim=1, keepdim=True)
        transformed_X = X2[:-1, :] + proj * v

        G, M, P, R = seperate_output(transformed_X, input_size)
        G, M, P = G.to(device), M.to(device), P.to(device)
        all_embedding, _, _, _ = model.encoder(G, M, P)
        f_transformed_X = all_embedding.to(Z2.device)

        h = f_transformed_X - Z2[:-1, :]

        numerator = torch.sum(h * delta_Z, dim=1)
        denom = torch.norm(h, dim=1) * torch.norm(delta_Z, dim=1) + 1e-8
        valid_mask =  (q != 0)
        loss_terms = torch.where(valid_mask,
                                 numerator / denom,
                                 torch.tensor(0.0, device=X2.device))
        return -torch.sum(loss_terms) / torch.sum(q != 0)
    
    current_b = b[0].clone().detach()
    baseline_loss = compute_loss(current_b)
    null_loss = compute_loss(torch.zeros_like(current_b))
    taus = np.linspace(torch.min(abs(current_b)),torch.max(abs(current_b)),1000)
    cosine_diff = []
    selected_tau = 0
    for tau in taus:
        trial_b = current_b.clone()   
        idx = torch.where(torch.abs(current_b) < tau)[0]
        trial_b[idx] = 0.0   
        trial_loss = compute_loss(trial_b)
        cosine_diff.append(trial_loss.item() - baseline_loss.item())
        if (trial_loss.item()-baseline_loss.item())/(null_loss.item()-baseline_loss.item()) >= 0.1:
            break
    knee = KneeLocator(taus[:len(cosine_diff)],cosine_diff,curve="convex")
    selected_tau = knee.elbow

    if not selected_tau:
        selected_tau = taus[len(cosine_diff)-1]
    print(f"tau:{selected_tau}")

    selected_index = torch.where(torch.abs(current_b) > selected_tau)[0]
    return current_b, selected_index




#################################################################################### Plot

def plot_species_scores(
    diff_score,
    species,
    lineage,
    diff_species,
    figsize = (6, 3)
):
    """
    Plot differential scores for species.

    Args:
        diff_score: Differential score from driver_species_detection.
        species: Name of all species.
        lineage: First lineage in node of interest.
        diff_species: Index of differential species.
    """
    # Extract and sort values
    b_val = diff_score.detach().cpu().numpy()
    species_sorted = [species[i] for i in diff_species]
    b_sorted = [b_val[i] for i in diff_species]

    # Create and sort DataFrame
    df = pd.DataFrame({
        "Species": species_sorted,
        "Magnitude": b_sorted/max(abs(b_val))
    }).sort_values(by="Magnitude", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Set colors based on sign
    colors = ["#aa0e0ec1" if val < 0 else "#1f77b4" for val in df["Magnitude"]]

    # Plot on given axis
    bars = ax.bar(df["Species"], df["Magnitude"], color=colors)

    # Reference line at 0
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    # Custom legend (placed below axis, not overlapping x-ticks)
    legend_patches = [
        mpatches.Patch(color="#aa0e0ec1", label=f"Faster increase/slower decrease in {lineage}"),
        mpatches.Patch(color="#1f77b4", label="Faster increase/slower decrease in other lineages")
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),  # move further below
        fontsize=9,
        frameon=False,
        ncol=1
    )
 
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(range(len(df["Species"])))
    ax.set_xticklabels(df["Species"], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(ax.get_yticks())
    ax.tick_params(axis="y", labelsize=9)