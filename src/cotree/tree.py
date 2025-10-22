from collections import defaultdict, deque, OrderedDict
from tqdm import tqdm

import numpy as np

from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import PathCollection
import seaborn as sns

from skmisc.loess import loess
from pcurvepy2 import PrincipalCurve

from cotree.DTMAE import seperate_output


# Tree

def kmeans_clustering(data, time, n_clusters_before = 20, n_clusters_after = 3):

    # find initial nodes
    kmeans_before = KMeans(n_clusters=n_clusters_before, init='k-means++', n_init=200, random_state=42)
    kmeans_before.fit(data[(time<105) & (time>0),:])
    labels_before = kmeans_before.labels_
    centers_before = kmeans_before.cluster_centers_

    kmeans_after = KMeans(n_clusters=n_clusters_after, init='k-means++', n_init=200, random_state=42)
    kmeans_after.fit(data[time==105,:])
    labels_after = kmeans_after.labels_+n_clusters_before
    centers_after = kmeans_after.cluster_centers_

    root =  np.unique(data[time == 0,:], axis=0)

    labels = np.zeros(len(time))
    labels[(time<105) & (time>0)] = labels_before+1
    labels[time==105] = labels_after+1
    
    centers = np.concatenate((root, centers_before,centers_after),axis = 0)
    return labels, centers





def mst_root_to_leaf_lineages(mst_sparse, root, centers):
    
    A = mst_sparse.tocsr()
    Au = A + A.T

    n = Au.shape[0]
    indptr, indices = Au.indptr, Au.indices
    adj = [list(indices[indptr[i]:indptr[i+1]]) for i in range(n)]

    # BFS from root to get parent pointers within the root's component
    visited = np.zeros(n, dtype=bool)
    parent = {root: None}
    q = deque([root])
    visited[root] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                q.append(v)

    # Leaves in the root's component (degree==1), excluding the root itself
    leaves = [i for i in range(n) if visited[i] and len(adj[i]) == 1 and i != root]

    # If the root is isolated or the component is a single node
    if not leaves and visited[root] and len(adj[root]) == 0:
        return {"Lineage1": [root]}

    # Reconstruct each root→leaf path using parent pointers
    lineages = {}
    k = 1
    for leaf in sorted(leaves):
        path = []
        cur = leaf
        while cur is not None:     # climb parents to the root
            path.append(cur)
            cur = parent.get(cur, None)
        path.reverse()
        lineages[f"Lineage{k}"] = centers[path,:]
        k += 1

    # Fallback: if no leaves (e.g., degenerate 2-node component where root is a leaf),
    # include the single path root→neighbor
    if not lineages and len(adj[root]) == 1:
        neighbor = adj[root][0]
        lineages["Lineage1"] = [root, neighbor]

    return lineages


def piecewise_linear_tree(flatten_embeddings, cell_num, time_length,
                          n_clusters_before=20, n_clusters_after=3):
    

    # --- nodes / clustering ---
    time = np.array(list(range(time_length)) * cell_num)
    clusters, centers = kmeans_clustering(
        flatten_embeddings,
        time=time,
        n_clusters_before=n_clusters_before,
        n_clusters_after=n_clusters_after
    )
    cell_label = np.repeat(range(1, cell_num + 1), time_length)

    # --- per-cell paths (cluster visits over time, deduping consecutive repeats) ---
    time_order_idx = [np.unique(clusters[cell_label == i], return_index=True)[1]
                      for i in range(1, cell_num + 1)]
    time_order_clusters = [np.unique(clusters[cell_label == i], return_index=True)[0]
                           for i in range(1, cell_num + 1)]
    time_order = [time_order_clusters[i][np.argsort(time_order_idx[i])].astype(int)
                  for i in range(cell_num)]
    

    # --- counts of adjacent cluster connections across all cell paths ---
    K = centers.shape[0]
    adjacency_counts = np.zeros((K, K), dtype=int)

    for seq in time_order:
        if len(seq) < 2:
            continue
        # Count each consecutive transition as an undirected edge
        for u, v in zip(seq[:-1], seq[1:]):
            if u == v:
                continue
            adjacency_counts[u, v] += 1
            adjacency_counts[v, u] += 1

    # --- distance matrix: 1 / count for existing edges;
    distance_matrix = np.zeros((K, K), dtype=float)
    mask = adjacency_counts > 0
    distance_matrix[mask] = 1.0 / adjacency_counts[mask]
    np.fill_diagonal(distance_matrix, 0.0)


    graph = csr_matrix(distance_matrix)
    mst_sparse = minimum_spanning_tree(graph) 
    root = time_order_clusters[0][0].astype(int)
    lineages = mst_root_to_leaf_lineages(mst_sparse, root, centers)
    
    return lineages



def project_embeddings_to_tree(flatten_embeddings, trajectory_tree, time_length, stretch = 2, project_diff = 0.25, sigma = 100):

    num_cells_all = len(flatten_embeddings)
    num_lineages = len(trajectory_tree)
    num_cells = num_cells_all // time_length

    # Initialize matrices
    growth_stage_matrix = np.zeros((num_cells_all, num_lineages))
    projection_distances = np.zeros((num_cells_all, num_lineages))
    segment_distances = np.zeros((num_cells, num_lineages))
    segment_w = np.zeros((num_cells_all, num_lineages))

    # Process each lineage in the trajectory tree
    for lineage_idx, leaf in enumerate(trajectory_tree):
        p_curve = PrincipalCurve(k=4)
        lineage_points = np.vstack(trajectory_tree[leaf])

        # Project each point to its corresponding lineage
        d_squared, distances = p_curve.project_to_curve(flatten_embeddings, points=lineage_points, stretch=stretch)
        growth_stage_matrix[:, lineage_idx] = p_curve.pseudotimes_interp
        projection_distances[:, lineage_idx] = d_squared


        #Adjust growth stages to ensure monotonic increase over time
        for i in range(1,num_cells+1):
            idx_segment = list(range((i-1)*time_length,(time_length*i)))
            for j in range(len(idx_segment)-1):
                if growth_stage_matrix[idx_segment[j+1],lineage_idx] < growth_stage_matrix[idx_segment[j],lineage_idx]+project_diff:
                    growth_stage_matrix[idx_segment[j+1],lineage_idx] = growth_stage_matrix[idx_segment[j],lineage_idx]+project_diff
            
        # distance between cell trajectory and each lineage
        segment_distances[:, lineage_idx] = np.array([
            projection_distances[((i+1)*time_length-1),lineage_idx]
            for i in range(num_cells)])   
        segment_w[:, lineage_idx] = segment_distances[:,lineage_idx].repeat(time_length)
    segment_w = np.exp(-segment_w/np.max(segment_w)*sigma)
    
    # Normalize distances for soft lineage assignment
    segment_assignments = segment_w / segment_w.sum(axis=1, keepdims=True)
    
    return growth_stage_matrix, projection_distances, segment_assignments, segment_distances


def detect_bifurcations(interpolations, new_curve_time, time_seg, epsilon=1e-3):

    final_l = []
    final_t = []
    include_lineage = list(interpolations.keys())
    merged_bifurcations = [(include_lineage, 0)]
    detect_matrix = interpolations[include_lineage[0]][0] # root
    time_len_each_lineage = [len(x) for x in new_curve_time.values()]

    # Find the time range shared with all the cells
    common_time = min(time_len_each_lineage)

    # Detect bifurcations
    for i in range(common_time):
        points = np.array([interpolations[j][i, :] for j in include_lineage])
        ave = np.mean(points, axis=0) 
        # Calculate the distance from each lineage to the avg of all the lineages
        distpoint = [
            np.dot(interpolations[j][i, :] - ave, interpolations[j][i, :] - ave)
            for j in include_lineage] 
        # If anly lineage shows significant deviation from the centers, current time is identified as 
        # bifucation point
        if np.max(distpoint) > epsilon:
            final_t.append(time_seg[i])
            final_l.append(include_lineage.copy())

            max_idx = np.argmax(distpoint)
            include_lineage.pop(max_idx)
        
        if len(include_lineage) == 1:
            break

    # Merge
    for i in range(len(final_t)):
        t_val = final_t[i]

        # Initialize the parts that needs update
        nrow_ = np.sum(time_seg <= t_val)
        ave_l = np.zeros((nrow_, interpolations[include_lineage[0]].shape[1]))
        
        # Sum of all the relevant lineages
        for l in range(len(final_l[i])):
            ave_l += interpolations[final_l[i][l]][(new_curve_time[final_l[i][l]]<=t_val), :]
        
        # Average
        ave_l /= len(final_l[i])
        
        # Overwrite each lineage's values up to time t_val
        for l in range(len(final_l[i])):
            interpolations[final_l[i][l]][(new_curve_time[final_l[i][l]]<=t_val), :] = ave_l
        
        # Add change_point
        last_row = ave_l[-1, :]  
        detect_matrix = np.vstack((detect_matrix, last_row))
        merged_bifurcations.append((final_l[i],final_t[i]))

    return detect_matrix, interpolations, merged_bifurcations



def detect_merge_bifurcations(trajectory_tree, lineage_stage_indices, epsilon=1e-3):

    max_stages = [np.max(stage) for stage in lineage_stage_indices] + [0]
    ordered_indices = np.argsort(max_stages)

    # Generate common time segmentation
    time_seg = np.array([])
    for i in range(len(max_stages) - 1):
        start_time, end_time = max_stages[ordered_indices[i]], max_stages[ordered_indices[i + 1]]
        num_points = int(np.floor((end_time - start_time) / 0.1))
        time_seg = np.concatenate((time_seg, np.linspace(start_time, end_time, num=num_points)))

    interpolations, new_curve_times = {}, {}
    for leaf_index, leaf in enumerate(trajectory_tree.keys()):
        leaf_curve, lambda_seg = [], time_seg[time_seg <= np.max(lineage_stage_indices[leaf_index])]
        new_curve_times[leaf] = lambda_seg

        for dim in range(trajectory_tree[leaf].shape[1]):
            interpolator = interp1d(lineage_stage_indices[leaf_index], trajectory_tree[leaf][:, dim])
            leaf_curve.append(interpolator(lambda_seg))

        interpolations[leaf] = np.array(leaf_curve).T

    # Detect bifurcations and update the shared branch
    detect_matrix, updated_interpolations, merged_bifurcation_raw = detect_bifurcations(interpolations, new_curve_times, time_seg, epsilon)

    # Generate lineage pairs for inference part
    merged_bifurcation_pairs = {}
    for bifurcation in merged_bifurcation_raw[::-1]:
        included_l = bifurcation[0]
        time = bifurcation[1]
        for lineage in included_l:
            if lineage not in merged_bifurcation_pairs:
                merged_bifurcation_pairs[lineage] = time
            if len(merged_bifurcation_pairs) == len(interpolations):
                break
    

    final_trajectory_tree = {}
    for i, leaf in enumerate(trajectory_tree.keys()):
        final_trajectory_tree[leaf] = updated_interpolations[leaf]

    return detect_matrix, final_trajectory_tree, new_curve_times, merged_bifurcation_pairs




def construct_principal_tree(flatten_embeddings, time_length, cell_num, alpha = 0.05,
                                                                weight_factor=5,
                                                                sample_num=106,
                                                                max_iter= 20,
                                                                epsilon=0.1,
                                                                span = 0.75,
                                                                n_clusters_before = 5,
                                                                n_clusters_after = 3,
                                                                project_diff = 0.25):
    

    """
    Construct a principal tree from flattened embeddings.

    Args:
        flatten_embeddings: Concatenated cell embeddings. Accepts NumPy array.

        time_length: int. Number of time steps per cell trajectory.

        cell_num: int. Number of cells replicates.

        alpha: float, default=0.05. Blend factor between point-wise projection weights
            and segment/lineage weights: `combined = alpha*point_w + (1-alpha)*segment_w`.

        weight_factor: float, default=5. Scaling factor inside the exponential when computing point-wise weights.

        sample_num: int, default=106. Number of evenly spaced growth-stage samples
            along each lineage used to fit/predict LOESS curves (per embedding dim).

        max_iter: int, default=20. Maximum number of refinement iterations.

        epsilon: float, default=0.1. Tolerance used in bifurcation detection/merging
            to decide whether nearby curve segments should be merged.

        span: float, default=0.75. LOESS smoothing span (fraction in (0, 1]) that
            controls local window size for curve fitting.

        n_clusters_before: int, default=5. Number of clusters used to initialize
            the piecewise-linear tree (intermediate states of trajectory).

        n_clusters_after: int, default=3. Number of clusters used to initialize
            the piecewise-linear tree (final states of trajectory).

        project_diff: float, default=0.25. Adjustment constant for growth stage.

    Returns:
        refined_tree: np.ndarray 
            Final tree.

        updated_detect_matrix: np.ndarray
            Diagnostics from the bifurcation detection/merge step (format depends on
            `detect_merge_bifurcations` implementation).

        sepration_time: dictionary
            Time when lineages seperate from each other.
    """

    embedding_dim = flatten_embeddings.shape[1]


    # Construct initial tree
    refined_tree = piecewise_linear_tree(flatten_embeddings, cell_num, time_length,n_clusters_before, n_clusters_after)

    # Initialize distance object
    prev_distance = np.inf 

    pbar = tqdm(range(max_iter), desc="Principal Tree Construction", leave=False)

    for iteration in pbar:
        # Project cells to the tree
        growth_stage_matrix, projection_d, segment_weight, segment_d  = project_embeddings_to_tree(flatten_embeddings, refined_tree, time_length, stretch=0, project_diff=project_diff)

        lineage_stage_indices = []

        # Calculate weights
        point_w = [np.exp(-projection_d[:, lineage_idx]/np.max(projection_d)*weight_factor) for lineage_idx in range(projection_d.shape[1])]
        cell_w = [segment_weight[:,lineage_idx] for lineage_idx in range(projection_d.shape[1])]
        combined_weights_all = [alpha*point_w[i]+(1-alpha)*cell_w[i] for i in range(len(cell_w))]# weights: single points, segment: lineages
        combined_weights_all = combined_weights_all/np.sum(combined_weights_all,axis=0)

        # Calculate loss function
        current_distance = (projection_d*combined_weights_all.T).mean()
        pbar.set_description(f"Iter {iteration}: Loss = {current_distance:.6f}")

        # Stop if distance sum stabilizes
        if np.abs(current_distance - prev_distance) < 1e-2 * prev_distance:
            break

        prev_distance = current_distance

        # Update lineages
        for lineage_idx, leaf in enumerate(refined_tree):

            # Adjust the weight of root to make sure all the lineages start with the same point
            combined_weights = combined_weights_all[lineage_idx]
            max_weight = np.max(combined_weights)
            start_point_indices = np.arange(len(flatten_embeddings)) % time_length == 0
            combined_weights[start_point_indices] = max_weight + 50

            # Adjust growth stages starting from 0
            growth_stage_matrix[start_point_indices, lineage_idx] = 0

            # Determine growth stage range for the lineage
            lineage_growth_stages = growth_stage_matrix[:, lineage_idx]
            start_time = 0
            end_time = np.max(lineage_growth_stages)
            sorted_growth_stages = np.sort(lineage_growth_stages)
            sorted_indices = np.argsort(lineage_growth_stages)

            # Fit LOESS regression for each embedding dimension
            new_lineage = []
            sample_indices = np.linspace(start_time, end_time, num=sample_num)
            for dim in range(embedding_dim):
                loess_regressor = loess(sorted_growth_stages, flatten_embeddings[sorted_indices, dim],
                                        weights=combined_weights[sorted_indices],
                                        span=span)
                loess_regressor.fit()
                new_lineage.append(loess_regressor.predict(sample_indices).values)

            refined_tree[leaf] = np.array(new_lineage).T
            lineage_stage_indices.append(sample_indices)

        # Update bifurcations
        bifurcation_point, update_tree, new_curve_time, sepration_time = detect_merge_bifurcations(refined_tree, lineage_stage_indices,epsilon=epsilon)
        refined_tree = update_tree
    growth_stage_matrix, projection_d, segment_weight, segment_d  = project_embeddings_to_tree(flatten_embeddings, refined_tree, time_length, stretch=0)
    return refined_tree, bifurcation_point, sepration_time


########################################################################################## Tree plot


def vis_tree(flatten_embeddings, trajectory_tree, label,
             bifurcation_point, save_path=None):
    """
    Visualize a principal tree from flattened embeddings.

    Args:
        flatten_embeddings: Concatenated cell embeddings. Accepts NumPy array.

        trajectory_tree: Output of construct_principal_tree.

        label: Clustering label of cells.

        bifurcation_point: Bifurcation point.

        save_path: Path to save the figure.

    """
   

    pca = PCA(n_components=5)
    X = np.asarray(flatten_embeddings)
    Z = pca.fit_transform(X) 
    N, D = X.shape

    def to_reduced_coords(arr):
        a = np.asarray(arr)
        if a.ndim == 1:
            if np.issubdtype(a.dtype, np.integer) or np.all((a.astype(int) == a) & (a >= 0) & (a < N)):
                return Z[a.astype(int)]
            if a.size == D:
                return pca.transform(a.reshape(1, -1))
            raise ValueError("Lineage/bifurcation 1D array must be indices or one D-dim point.")
        else:
            if a.shape[1] == D:
                return pca.transform(a)
            elif a.shape[1] <= Z.shape[1]:
                return a  # assume already reduced
            raise ValueError("Lineage/bifurcation has incompatible shape.")

    labels_arr = np.asarray(label)
    uniques = list(OrderedDict.fromkeys(labels_arr.tolist()))
    palette = ["#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"]
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(uniques)}

    def make_proxy_handles(handles, labels, markersize=6):
        proxies = []
        for h, lab in zip(handles, labels):
            if isinstance(h, Line2D):
                proxies.append(Line2D(
                    [0], [0],
                    color=h.get_color(),
                    linestyle=h.get_linestyle(),
                    linewidth=h.get_linewidth(),
                    marker=h.get_marker(),
                    markersize=markersize,
                    markerfacecolor=h.get_markerfacecolor() if hasattr(h, "get_markerfacecolor") else h.get_color(),
                    markeredgecolor=h.get_markeredgecolor() if hasattr(h, "get_markeredgecolor") else h.get_color(),
                    alpha=1.0,
                    label=lab
                ))
                continue
            if hasattr(h, "get_facecolors") or isinstance(h, PathCollection):
                face = None
                if hasattr(h, "get_facecolors") and h.get_facecolors().size:
                    face = h.get_facecolors()[0]
                elif hasattr(h, "get_edgecolors") and h.get_edgecolors().size:
                    face = h.get_edgecolors()[0]
                else:
                    face = "k"
                proxies.append(Line2D(
                    [0], [0],
                    marker="o", linestyle="None",
                    markerfacecolor=face, markeredgecolor=face,
                    markersize=markersize, alpha=1.0, label=lab
                ))
                continue
            if isinstance(h, Patch):
                proxies.append(Patch(
                    facecolor=h.get_facecolor(),
                    edgecolor=h.get_edgecolor(),
                    label=lab, alpha=1.0
                ))
                continue
            proxies.append(h)
        return proxies

    sns.set_theme(style="white")

    fig, ax = plt.subplots(figsize=(9.5, 5))
    j = 1

    for cat in uniques:
        m = labels_arr == cat
        ax.scatter(
            Z[m, 0], Z[m, j],
            c=color_map[cat], alpha=0.5, s=0.5, label=str(cat)
        )

    leaf_label = 1
    for _, lineage in trajectory_tree.items():
        try:
            L = to_reduced_coords(lineage)
            ax.plot(L[:, 0], L[:, j], label=f"Lineage {leaf_label}", linewidth=2)
            leaf_label += 1
        except Exception:
            continue

    if bifurcation_point is not None:
        try:
            BP = to_reduced_coords(np.atleast_2d(bifurcation_point))
            ax.scatter(BP[:, 0], BP[:, j], c="black", s=15, label="_nolegend_")
        except Exception:
            pass

    ax.set_title("COTree ", loc="left", fontsize=14)
    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel(f"PC {j + 1}", fontsize=12)

    handles_all, labels_all = ax.get_legend_handles_labels()
    by_label = OrderedDict()
    for h, l in zip(handles_all, labels_all):
        if l == "_nolegend_":
            continue
        if l not in by_label:
            by_label[l] = h

    desired_order = [f"Lineage {i+1}" for i in range(len(trajectory_tree.keys()))]
    ordered_labels = [l for l in desired_order if l in by_label]
    ordered_labels += [l for l in by_label.keys() if l not in ordered_labels]
    ordered_handles = [by_label[l] for l in ordered_labels]

    proxy_handles = make_proxy_handles(ordered_handles, ordered_labels, markersize=6)
    ax.legend(
        proxy_handles, ordered_labels,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0, fontsize=10, markerscale=1.2,
        frameon=True, framealpha=1.0
    )

    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # room for legend
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    return None

# Clustering
def lineage_clustering(projection_distances, time_length, trajectory_tree):
    """
    Cell clustering.

    Args:
    projection_distances: distance from cells to trajectory tree
    
    time_length: int. Number of time steps per cell trajectory.

    trajectory_tree: Output tree.
    
    """
    distances_to_lineage = projection_distances.reshape(-1, time_length, projection_distances.shape[-1])
    leaf = np.array(list(trajectory_tree.keys()))
    clustering_label = leaf[np.argmin(np.sum(distances_to_lineage,axis=1), axis=1)]
    return clustering_label

# Time inference
def vis_predictive_time(trajectory_tree,
                        projection_distances,
                        clustering_labels,
                        time_length,
                        save_path=None):
    """
    Predictive time.

    Args:
    trajectory_tree: Trajectory tree.
    
    projection_matrix: Distance from cells to trajectory tree.

    clustering_labels: Clustering labels of cells.

    time_length: int. Number of time steps per cell trajectory.

    save_path: Output path to save the figure.

    """
   
    leaf = list(trajectory_tree.keys())
    lineage_num = len(leaf)
    cell_projection_matrix = projection_distances.reshape(-1, time_length, lineage_num)

    predictive_time_all = defaultdict(list)

    for cell_index, cell in enumerate(cell_projection_matrix):
        lineage_label = clustering_labels[cell_index]
        for t in range(time_length, -1, -1):
            single_cell_projection = cell[:t,:]
            current_assign = single_cell_projection.sum(axis = 0)
            min_val = np.min(current_assign)
            is_unique = np.sum(current_assign == min_val) == 1
            if leaf[np.argmin(current_assign)] != lineage_label or not is_unique:
                predictive_time_all[lineage_label].append(t)
                break

    order = {f"Lineage {i+1}": f"Lineage{i+1}" for i in range(len(leaf))}
    plot_dic = {key: predictive_time_all[order[key]] for key in order.keys()}
    fig, ax = plt.subplots(figsize=(4, 3))
    color_list = ["#aec7e8", "#ffbb78", "#98df8a"]  
    
    sns.boxplot(
        data=plot_dic,
        palette=color_list,
        ax=ax,
        showfliers=False,
        width=0.6,
        linewidth=0.5
    )

    sns.stripplot(
        data=plot_dic,
        ax=ax,
        color='black',
        jitter=0.25,
        size=1,
        zorder=10,
       # alpha=0.7
    )

    ax.set_xticks(range(lineage_num))
    ax.set_ylim(-1, 108)
    ax.set_ylabel('Real Time (min)', fontsize=9)
    ax.set_title('Predictive Time', fontsize=11, pad=10)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return predictive_time_all

def vis_retrospective_time(trajectory_tree,
                   growth_stage_matrix,
                   seperation_time,
                   clustering_labels,
                   time_length,
                   save_path=None):
    """
    Retrospective time.
    
    Args:

    trajectory_tree: Trajectory tree.
    
    growth_stage_matrix: Growth stage matrix of cells.

    clustering_labels: Clustering labels of cells.

    time_length: int. Number of time steps per cell trajectory.

    save_path: Output path to save the figure.

    seperation_time: Lineage seperation time.

    clustering_labels: Clustering labels of cells.

    time_length: int. Number of time steps per cell trajectory.

    save_path: Output path to save the figure.

    """
   
    lineage_num = len(seperation_time)
    cell_growth_time_matrix = growth_stage_matrix.reshape(-1, time_length, lineage_num)

    retrospective_time_all = defaultdict(list)

    leaf_list = np.array(list(trajectory_tree.keys()))
    for cell_index, cell in enumerate(cell_growth_time_matrix):
        lineage_label = clustering_labels[cell_index]
        i = np.where(leaf_list == lineage_label)[0][0]
        
        cell_time = cell[:,i]
        enter_time = np.min(np.where(cell_time>seperation_time[lineage_label]))

        retrospective_time_all[lineage_label].append(enter_time)
       
    order = {f"Lineage {i+1}": f"Lineage{i+1}" for i in range(len(leaf_list))}
    plot_dic = {key: retrospective_time_all[order[key]] for key in order.keys()}

    fig, ax = plt.subplots(figsize=(4, 3))

    color_list = ["#aec7e8", "#ffbb78", "#98df8a"]  

    sns.boxplot(
        data=plot_dic,
        palette=color_list,
        ax=ax,
        showfliers=False,
        width=0.6,
        linewidth=0.5
    )

    sns.stripplot(
        data=plot_dic,
        ax=ax,
        color='black',
        jitter=0.25,
        size=1,
        zorder=10,
       # alpha=0.7
    )

    ax.set_xticks(range(lineage_num))
    ax.set_ylim(-1, 108)
    ax.set_ylabel('Real Time (min)', fontsize=9)
    ax.set_title('Retrospective Time', fontsize=11, pad=10)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return retrospective_time_all


