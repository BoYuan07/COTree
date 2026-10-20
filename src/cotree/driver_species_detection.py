from ._model_api import encode_gmp
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from kneed import KneeLocator
from .DTMAE import seperate_output


##################################################################################### helper functions
def bifurcation_data(trajectory_tree, merged_bifurcation_pairs):
    seperate_data = {}
    include_lineage = list(trajectory_tree.keys())
    for leaf, time in merged_bifurcation_pairs.items():
        include_lineage.remove(leaf)
        if include_lineage:
            seperate_data[leaf] = include_lineage.copy()
    return seperate_data



def extract_trajectory(trajectory_tree,
                       growth_stage_matrix,
                       data_raw,
                       flatten_embeddings,
                       clustering_labels,
                       merged_bifurcation_pairs,
                       list_truncation=80,
                       inter = 13):

    sorted_bifurcation_pairs = dict(sorted(merged_bifurcation_pairs.items(), key=lambda item: item[1]))
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
    raw_data, encoded_data, weight, real_time, lineage_map = {}, {}, {}, {}, {}
    leaf = np.array(list(trajectory_tree.keys()))
    for node_index, (lineage1, lineage2) in enumerate(seperate_data.items()):
        raw_data[node_index], encoded_data[node_index] = {}, {}
        weight[node_index], real_time[node_index] = {}, {}
        node_time = merged_bifurcation_pairs[lineage1]
        lineage_map[node_index] = {"direction 1": lineage1, "direction 2": lineage2}
        dir2_display = ", ".join(lineage2) if isinstance(lineage2, list) else lineage2
        print(f"Node {node_index}: direction 1 = {lineage1}, direction 2 = {dir2_display}")
        lineage1 = [lineage1]

        _extract_data_for_lineage(node_index, node_time, lineage1, "direction 1", leaf, inter)
        _extract_data_for_lineage(node_index, node_time, lineage2, "direction 2", leaf, inter)

    return raw_data, encoded_data, weight, real_time, lineage_map


def optimize_a_b(model, X_list, Z_list, a, b, w_list, lambda_reg, input_size, device, learning_rate=0.1,
                 epochs=100, threshold=0.5, tol=1e-6, patience=5):

    optimizer = torch.optim.Adam([a, b], lr=learning_rate)
    prev_loss = None
    bad_epochs = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Initialize loss function
        loss = 0
        k = b.shape[0]
        cosine = []

        for j in range(k+1):

            X = X_list[j]
            Z = Z_list[j]

            # Truncate weights
            q = w_list[j][:-1].clone()
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
            all_embedding = encode_gmp(model, G, M, P)
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

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, Cosine (direction 1): {cosine[0]:6f}, Cosine (direction 2): {cosine[1]:6f}, regularization: {lambda_reg * torch.norm(b[0,:],1):6f}")

        # Convergence check: stop when correlation has been decreasing for `patience` consecutive epochs
        current_loss = loss.item()
        if prev_loss is not None and current_loss > prev_loss + tol:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Stopped at epoch {epoch + 1} (correlation decreased for {patience} consecutive epochs)")
                break
        else:
            bad_epochs = 0
        prev_loss = current_loss

    return a.detach(), b.detach()  # Return optimized values


def init_parameters(X_list, dist_w, threshold = 0.5):
    b_list = []
    filtered_data = []
    for j in range(len(X_list)):
        X1 = X_list[j][:-1,:]
        X2 = X_list[j][1:,:]
        w = dist_w[j][:-1]
        q = w.clone()
        diff = X2-X1
        if j == 0:
            a = torch.mean(diff[(q > 0) & (q <= threshold),:],dim=0)
        else:
            b_list.append(torch.mean(diff[(q > 0) & (q <= threshold),:],dim=0)-a)
        filtered_data.append(X2[(q > 0) & (q <= threshold)]-X1[(q > 0) & (q <= threshold)])

    a = a.detach().clone().requires_grad_()
    b = torch.stack(b_list).detach().clone().requires_grad_()
    return a, b




def _select_driver_threshold(current_b, compute_loss,
                             target_fraction=0.9,
                             knee_sensitivity=1.0,
                             marginal_floor=None,
                             n_grid=200):
    """Pick the |b| threshold separating driver species from the noise floor.

    Species are ranked by |b|. For each kept-set size k, we keep only the top-k
    species and measure the fraction of achievable alignment gain recovered:

        R(k) = (null_loss - loss(top-k)) / (null_loss - baseline_loss)

    where:
        baseline_loss = loss(full b)
        null_loss     = loss(0)

    The selected number of drivers is chosen primarily by the knee of the
    recovered-gain curve R(k), which corresponds to the point of diminishing
    returns. To make knee detection stable when R(k) is noisy or slightly
    non-monotone, the knee is detected on

        R_mono(k) = cumulative max of R(k).

    If no knee is detected, the fallback is the smallest k that recovers
    `target_fraction` of the achievable gain.

    Parameters
    ----------
    current_b : torch.Tensor, shape (K,)
        Species-specific coefficient vector.

    compute_loss : callable
        Function mapping a candidate b vector to a scalar loss. Lower is better.

    target_fraction : float, default=0.9
        Fallback recovery target, and also an optional upper cap on selected_k.
        For example, 0.9 means do not keep species beyond the first k that
        recovers 90% of the achievable gain unless no such k exists.

    knee_sensitivity : float, default=1.0
        Sensitivity parameter passed to KneeLocator.

    marginal_floor : float or None, default=None
        Optional diagnostic only. If not None, computes the first k where the
        marginal gain falls below `marginal_floor * peak_slope`. This value is
        returned in `info["floor_k"]`, but is not used for selection in this
        knee-primary version.

    n_grid : int, default=200
        Maximum number of k values to evaluate. The grid is dense near small k
        and sparse in the tail when K is large.

    Returns
    -------
    selected_tau : float
        Threshold on |b|. Keeping |b| >= selected_tau gives selected_k species.

    selected_index : torch.LongTensor
        Original indices of selected driver species.

    info : dict
        Diagnostic information, including R, R_mono, k_grid, selected_k,
        knee_k, target_k, floor_k, baseline_loss, and null_loss.
    """

    if current_b.ndim != 1 or current_b.numel() == 0 or not torch.isfinite(current_b).all():
        raise ValueError("Driver scores must be a nonempty finite vector.")
    if not 0 < target_fraction <= 1 or n_grid < 2:
        raise ValueError("Require target_fraction in (0,1] and n_grid >= 2.")
    abs_b = torch.abs(current_b).detach()
    order = torch.argsort(abs_b, descending=True)
    sorted_abs = abs_b[order]
    K = current_b.shape[0]

    # Full b and null b reference losses.
    baseline_loss = compute_loss(current_b).item()
    null_loss = compute_loss(torch.zeros_like(current_b)).item()
    if not np.isfinite([baseline_loss, null_loss]).all():
        raise ValueError("Threshold loss must be finite.")
    denom = null_loss - baseline_loss

    # Degenerate case: b gives no improvement over the null model.
    if denom <= 1e-12:
        info = {
            "R": np.array([0.0]),
            "R_mono": np.array([0.0]),
            "k_grid": np.array([0]),
            "selected_k": 0,
            "knee_k": None,
            "target_k": 0,
            "floor_k": None,
            "baseline_loss": baseline_loss,
            "null_loss": null_loss,
            "selection_rule": "degenerate_no_gain",
        }
        return sorted_abs[0].item() + 1.0, torch.empty(0, dtype=torch.long), info

    # Adaptive grid over number of kept species.
    # Dense near the top-ranked species, sparse in the long tail.
    if K <= n_grid:
        k_grid = np.arange(1, K + 1)
    else:
        n_dense = max(1, n_grid // 2)
        dense = np.arange(1, n_dense + 1)
        sparse = np.linspace(n_dense + 1, K, n_grid - n_dense).astype(int)
        k_grid = np.unique(np.concatenate([dense, sparse]))

    # Compute recovered-gain curve R(k).
    R = [0.0]
    for k in k_grid:
        k = int(k)
        trial_b = torch.zeros_like(current_b)
        keep = order[:k]
        trial_b[keep] = current_b[keep]

        loss_k = compute_loss(trial_b).item()
        R_k = (null_loss - loss_k) / denom
        R.append(R_k)

    ks = np.concatenate([[0], k_grid]).astype(float)
    R = np.asarray(R, dtype=float)

    # Monotone-smoothed curve for knee and target detection.
    # This avoids instability when adding one species slightly decreases R.
    R_mono = np.maximum.accumulate(R)

    # Knee detection on monotone-smoothed recovery curve.
    knee_k = None
    if KneeLocator is not None and len(ks) >= 3:
        try:
            kl = KneeLocator(
                ks,
                R_mono,
                curve="concave",
                direction="increasing",
                S=knee_sensitivity,
                interp_method="polynomial",
            )
            if kl.knee is not None:
                knee_k = int(round(float(kl.knee)))
        except Exception:
            knee_k = None

    # Fallback / cap: first k reaching target_fraction of recovered gain.
    reach = np.where(R_mono >= target_fraction)[0]
    target_k = int(ks[reach[0]]) if len(reach) else K

    # Optional diagnostic marginal floor, not used for selection.
    floor_k = None
    if marginal_floor is not None:
        slope = np.diff(R) / np.maximum(np.diff(ks), 1.0)
        peak = slope.max() if len(slope) else 0.0
        if peak > 0:
            below = np.where(slope < marginal_floor * peak)[0]
            if len(below):
                floor_k = int(ks[below[0]])

    # Selection rule:
    #   primary: knee
    #   fallback: target_fraction
    #   cap: do not exceed target_k unless knee is unavailable
    if knee_k is not None and knee_k > 0:
        selected_k = knee_k
        selection_rule = "knee"

        # If the knee is later than the target point, cap at target_k.
        # This avoids keeping a long weak tail after the target gain is reached.
        if target_k > 0:
            selected_k = min(selected_k, target_k)
            if selected_k == target_k and knee_k > target_k:
                selection_rule = "target_cap_after_knee"
    else:
        selected_k = target_k
        selection_rule = "target_fallback"

    selected_k = int(max(1, min(selected_k, K)))

    # Convert selected_k into threshold tau.
    # Threshold is midway between kth and k+1th strongest |b|.
    if selected_k >= K:
        selected_tau = sorted_abs[-1].item() * 0.5
    else:
        selected_tau = 0.5 * (
            sorted_abs[selected_k - 1].item()
            + sorted_abs[selected_k].item()
        )

    selected_index = order[:selected_k].sort().values

    info = {
        "R": R,
        "R_mono": R_mono,
        "k_grid": ks,
        "selected_k": selected_k,
        "knee_k": knee_k,
        "target_k": target_k,
        "floor_k": floor_k,
        "baseline_loss": baseline_loss,
        "null_loss": null_loss,
        "selection_rule": selection_rule,
    }

    return selected_tau, selected_index, info


def driver_species_detection(model,
                                  node_ind,
                                  raw_data,
                                  encoded_data,
                                  weight,
                                  input_size,
                                  device,
                                  lineage_map=None,
                                  sparse_penalty = 0.01,
                                  epoches = 300,
                                  learning_rate = 0.001,
                                  threshold = 0.5,
                                  tol = 1e-6,
                                  patience = 5,
                                  target_fraction = 0.9,
                                  knee_sensitivity = 1.0,
                                  marginal_floor = 0.0,
                                  n_grid = 200):

    X = [torch.tensor(raw_data[node_ind][i], dtype=torch.float) for i in raw_data[node_ind]]
    Z = [torch.tensor(encoded_data[node_ind][i], dtype=torch.float) for i in raw_data[node_ind]]
    if len(X) != 2 or any(x.ndim != 2 or len(x) < 2 for x in X):
        raise ValueError("Driver detection requires at least two observations in each direction.")
    if any(z.ndim != 2 or len(z) != len(x) for x, z in zip(X, Z)):
        raise ValueError("Raw and encoded observations must align in each direction.")
    if any(not torch.isfinite(t).all() for t in X + Z):
        raise ValueError("Driver observations must be finite.")
    w_list = [torch.tensor(weight[node_ind][i], dtype=torch.float) for i in weight[node_ind]]
    if any(w.ndim != 1 or len(w) != len(x) or not torch.isfinite(w).all()
           or not ((w[:-1] > 0) & (w[:-1] <= threshold)).any() for x, w in zip(X, w_list)):
        raise ValueError("Each direction needs aligned positive weights within the threshold.")
    if epoches < 1:
        raise ValueError("epoches must be positive.")
    model.to(device).eval()
    # Initialization
    a,b = init_parameters(X,w_list, threshold = threshold)
    # Optimize a and b
    optimized_a, optimized_b = optimize_a_b(model, X, Z, a, b, w_list, device= device, input_size = input_size, lambda_reg = sparse_penalty,
                                            epochs = epoches, learning_rate=learning_rate, threshold = threshold, tol=tol, patience=patience)

    # choose optimal tau
    def compute_loss(b_vec, threshold = threshold):
        X2, Z2 = X[1], Z[1]
        q = w_list[1][:-1].clone()
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
        all_embedding = encode_gmp(model, G, M, P)
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
    selected_tau, selected_index, info = _select_driver_threshold(
        current_b, compute_loss,
        target_fraction=target_fraction,
        knee_sensitivity=knee_sensitivity,
        marginal_floor=marginal_floor,
        n_grid=n_grid)
    print(f"threshold selection: knee_k={info['knee_k']}, floor_k={info['floor_k']}, "
          f"target_k={info['target_k']}, selected_k={info['selected_k']} "
          f"of {current_b.shape[0]} species; tau={selected_tau:.4g}")

    if lineage_map is not None and node_ind in lineage_map:
        dir1 = lineage_map[node_ind]["direction 1"]
        dir2 = lineage_map[node_ind]["direction 2"]
        dir2_str = ", ".join(dir2) if isinstance(dir2, list) else dir2
        print(f"\nNode {node_ind} — detected driver species (by direction):")
        for idx in selected_index:
            i = idx.item() if hasattr(idx, "item") else int(idx)
            faster = f"direction 2 ({dir2_str})" if current_b[i] > 0 else f"direction 1 ({dir1})"
            print(f"  Species index {i}: faster increase in {faster}")

    return current_b, selected_index




#################################################################################### Plot

def plot_species_scores(
    b,
    species,
    lineage_map,
    node_ind,
    diff_species,
    ax=None,
    top_n=20
):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    # Extract and sort values
    b_val = b.detach().cpu().numpy()
    species_sorted = [species[i] for i in diff_species]
    b_sorted = [b_val[i] for i in diff_species]

    # Create and sort DataFrame
    df = pd.DataFrame({
        "Species": species_sorted,
        "Magnitude": np.asarray(b_sorted) / max(float(np.max(np.abs(b_val))), 1e-8)
    })

    # Keep only the top_n species with the largest magnitudes
    if top_n is not None:
        df = df.reindex(df["Magnitude"].abs().sort_values(ascending=False).index).head(top_n)

    df = df.sort_values(by="Magnitude", ascending=True)

    # Set colors based on sign
    colors = ["#aa0e0ec1" if val < 0 else "#1f77b4" for val in df["Magnitude"]]

    # Plot on given axis
    bars = ax.bar(df["Species"], df["Magnitude"], color=colors)

    # Reference line at 0
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    dir1 = lineage_map[node_ind]["direction 1"]
    dir2 = lineage_map[node_ind]["direction 2"]
    dir2_str = ", ".join(dir2) if isinstance(dir2, list) else dir2

    # Custom legend (placed below axis, not overlapping x-ticks)
    legend_patches = [
        mpatches.Patch(color="#aa0e0ec1", label=f"Faster increase/slower decrease in {dir1}"),
        mpatches.Patch(color="#1f77b4", label=f"Faster increase/slower decrease in {dir2_str}")
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
    ax.set_xticks(range(len(df)), labels=df["Species"], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(ax.get_yticks())
    ax.tick_params(axis="y", labelsize=9)
    return ax


def plot_differential_species(
    data_raw,
    growth_stage_matrix,
    all_species,
    merged_bifurcation_pairs,
    lineage_map,
    diff_species,
    labels,
    save_dir=None,
    color_palette=None,
    node=1
):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    data_raw = np.asarray(data_raw)
    growth_stage_matrix = np.asarray(growth_stage_matrix)
    if data_raw.ndim != 3 or growth_stage_matrix.ndim != 3 or data_raw.shape[:2] != growth_stage_matrix.shape[:2]:
        raise ValueError("Raw data and stages must have matching (cells, time, features) axes.")

    dir1 = lineage_map[node]["direction 1"]

    label = np.array(labels)
    if label.size == data_raw.shape[0]:
        cell_label = label
    elif label.size == data_raw.shape[0] * data_raw.shape[1]:
        cell_label = label.reshape(data_raw.shape[:2])[:, 0]
    else:
        raise ValueError("Supply one label per cell or per observation.")
    lineages = [f"Lineage{i+1}" for i in range(growth_stage_matrix.shape[-1])]

    if color_palette is None:
        color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    colors = {item: color_palette[i % len(color_palette)] for i, item in enumerate(lineages)}

    for k in diff_species:
        k = int(k.item()) if hasattr(k, "item") else int(k)
        plt.figure(figsize=(4, 4))
        handles = {}

        for item in lineages:
            idx = np.where(cell_label == item)[0]
            if len(idx) == 0:
                continue
            mean_time = growth_stage_matrix[idx, :, :].mean(axis=0)
            mean_value = data_raw[idx, :, :].mean(axis=0)

            lineage_idx = np.where(np.array(lineages) == item)[0][0]
            x_vals = mean_time[:, lineage_idx]
            y_vals = mean_value[:, k]

            line, = plt.plot(x_vals, y_vals, color=colors[item], linewidth=2.5, label=item)
            handles[item] = line

        for i in range(data_raw.shape[0]):
            item = cell_label[i]
            lineage_idx = np.where(np.array(lineages) == item)[0][0]
            x_vals = growth_stage_matrix[i, :, lineage_idx]
            y_vals = data_raw[i, :, k]

            plt.plot(x_vals, y_vals, color=colors[item], alpha=0.05, linestyle="--")

        plt.axvline(
            x=merged_bifurcation_pairs[dir1],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Bifurcation ({dir1})"
        )

        plt.title(f"Species: {all_species[k]}", fontsize=14)
        plt.xlabel("Growth Stage", fontsize=12)
        plt.ylabel("Scaled Count", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.legend(
            handles.values(),
            handles.keys(),
            title="Cell Types",
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),  # X = 0.5 centers it, Y = -0.15 places it below
            ncol=len(handles),            # All items in one row
            fontsize=10,
            title_fontsize=11,
            frameon=False
        )
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        if save_dir is not None:
            import re
            filename = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(all_species[k]))
            plt.savefig(os.path.join(save_dir, f"{filename}_node{node}.jpeg"), bbox_inches='tight', dpi=300)
        else:
            plt.show()
