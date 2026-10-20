from collections import defaultdict, deque, OrderedDict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import PathCollection
import seaborn as sns

from pcurvepy2 import PrincipalCurve
from skmisc.loess import loess



################################# Tree algorithm ###############################


def kmeans_clustering(data, time, n_clusters_before = 20, n_clusters_after = 3):
    """
        Clustering on intermediate states and terminal states.
    """
    t_pos = time[time > 0]
    t_max = t_pos.max()

    before_mask = (time > 0) & (time < t_max)
    after_mask = time == t_max

    # find initial nodes
    kmeans_before = KMeans(n_clusters=n_clusters_before, init='k-means++', n_init=200, random_state=42)
    kmeans_before.fit(data[before_mask,:])
    labels_before = kmeans_before.labels_
    centers_before = kmeans_before.cluster_centers_

    kmeans_after = KMeans(n_clusters=n_clusters_after, init='k-means++', n_init=200, random_state=42)
    kmeans_after.fit(data[after_mask,:])
    labels_after = kmeans_after.labels_+n_clusters_before
    centers_after = kmeans_after.cluster_centers_

    root = np.mean(data[time == 0, :], axis=0, keepdims=True)

    labels = np.zeros(len(time))
    labels[before_mask] = labels_before+1
    labels[after_mask] = labels_after+1

    centers = np.concatenate((root, centers_before,centers_after),axis = 0)
    return labels, centers


def mst_root_to_leaf_lineages(mst_sparse, root, centers):
    """
        MST construction.
    """
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
        return {"Lineage1": centers[[root], :]}

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

    if not lineages and len(adj[root]) == 1:
        neighbor = adj[root][0]
        lineages["Lineage1"] = centers[[root, neighbor], :]

    return lineages


def _augment_mst_by_detour(mst_sparse, adjacency_counts, detour_ratio=2.0):
    """
        Non-tree graph construction.
    """
    n = adjacency_counts.shape[0]

    # Direct edge lengths (1/count), symmetric; 0 marks "no candidate edge".
    weight = np.zeros((n, n), dtype=float)
    mask = adjacency_counts > 0
    weight[mask] = 1.0 / adjacency_counts[mask]
    weight = np.maximum(weight, weight.T)

    # Dense symmetric adjacency of the MST itself (edge length or 0).
    mst = mst_sparse.toarray()
    adj = np.maximum(mst, mst.T)

    # Candidate non-MST edges (upper triangle), strongest connection first.
    iu, ju = np.triu_indices(n, k=1)
    cand = [(weight[i, j], i, j) for i, j in zip(iu, ju)
            if weight[i, j] > 0 and adj[i, j] == 0]
    cand.sort()

    for w, i, j in cand:
        # Distance u->v through the graph accepted so far.
        d = shortest_path(csr_matrix(adj), method="D", indices=i)[j]
        if (not np.isfinite(d)) or (d >= detour_ratio * w):
            adj[i, j] = w
            adj[j, i] = w
    return adj


def _graph_root_to_leaf_lineages(adj, root, centers, leaf_nodes=None, max_paths=64):
    """
    Non-tree graph lineage collection.
    """
    n = adj.shape[0]
    neighbors = [np.nonzero(adj[i] > 0)[0].tolist() for i in range(n)]

    # BFS to find the root's connected component.
    visited = np.zeros(n, dtype=bool)
    q = deque([root])
    visited[root] = True
    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)

    if leaf_nodes is not None:
        leaves = [int(i) for i in leaf_nodes if visited[i] and i != root]
    else:
        leaves = [i for i in range(n)
                  if visited[i] and len(neighbors[i]) == 1 and i != root]
    if not leaves:
        return {"Lineage1": centers[[root], :]}

    lineages = {}
    k = 1
    for leaf in sorted(leaves):
        # DFS enumerate simple paths root->leaf.
        stack = [(root, [root])]
        while stack and k <= max_paths:
            node, path = stack.pop()
            if node == leaf:
                lineages[f"Lineage{k}"] = centers[path, :]
                k += 1
                continue
            for v in neighbors[node]:
                if v not in path:
                    stack.append((v, path + [v]))
    return lineages if lineages else {"Lineage1": centers[[root], :]}


def _build_linear_tree_for_K(flatten_embeddings, time, cell_label, cell_num,
                              n_clusters_before, K, use_tree=True,
                              detour_ratio=2.0):
    """
    Build a piecewise-linear trajectory with K endpoint clusters.
    """
    clusters, centers = kmeans_clustering(
        flatten_embeddings, time=time,
        n_clusters_before=n_clusters_before, n_clusters_after=K
    )

    time_order_idx = [np.unique(clusters[cell_label == i], return_index=True)[1]
                      for i in range(1, cell_num + 1)]
    time_order_clusters = [np.unique(clusters[cell_label == i], return_index=True)[0]
                           for i in range(1, cell_num + 1)]
    time_order = [time_order_clusters[i][np.argsort(time_order_idx[i])].astype(int)
                  for i in range(cell_num)]

    n_centers = centers.shape[0]
    adjacency_counts = np.zeros((n_centers, n_centers), dtype=int)
    for seq in time_order:
        if len(seq) < 2:
            continue
        for u, v in zip(seq[:-1], seq[1:]):
            if u == v:
                continue
            adjacency_counts[u, v] += 1
            adjacency_counts[v, u] += 1

    root = time_order_clusters[0][0].astype(int)

    distance_matrix = np.zeros((n_centers, n_centers), dtype=float)
    mask = adjacency_counts > 0
    distance_matrix[mask] = 1.0 / adjacency_counts[mask]
    np.fill_diagonal(distance_matrix, 0.0)
    graph = csr_matrix(distance_matrix)
    mst_sparse = minimum_spanning_tree(graph)

    if use_tree:
        return mst_root_to_leaf_lineages(mst_sparse, root, centers)
    else:
        adj = _augment_mst_by_detour(mst_sparse, adjacency_counts, detour_ratio)
        endpoint_nodes = range(n_centers - K, n_centers)
        return _graph_root_to_leaf_lineages(adj, root, centers, leaf_nodes=endpoint_nodes)



def _recon_distance(flatten_embeddings, lineages, time_length):
    """
    Mean nearest-branch reconstruction distance of cells to the lineage tree.
    """
    _, proj_d, _, _ = project_embeddings_to_tree(
        flatten_embeddings, lineages, time_length)
    return float(proj_d.min(axis=1).mean())



def _max_pairwise_lineage_scale(tree):
    """
    Squared maximum distance between any two lineage curves (node-for-node).
    """
    arrs = [np.asarray(tree[L], float) for L in tree]
    if len(arrs) < 2:
        return 0.0
    m = min(a.shape[0] for a in arrs)
    arrs = [a[:m] for a in arrs]
    dmax = 0.0
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            d = float(np.linalg.norm(arrs[i] - arrs[j], axis=1).max())
            if d > dmax:
                dmax = d
    return float(dmax ** 2)


def _greedy_trim_trajectory(flatten_embeddings, lineages, time_length,
                            K_floor=1):
    """
    Over-split → trim: greedily strip the most redundant branch, scoring each level.
    """
    cur = dict(lineages)
    traj, trees = [], {}
    while True:
        d = _recon_distance(flatten_embeddings, cur, time_length)
        traj.append((len(cur), d))
        trees[len(cur)] = dict(cur)
        if len(cur) <= max(K_floor, 1):
            break
        best_name, best_dist = None, np.inf
        for name in list(cur):                     # most redundant = lowest resulting dist
            sub = {k: v for k, v in cur.items() if k != name}
            dd = _recon_distance(flatten_embeddings, sub, time_length)
            if dd < best_dist:
                best_name, best_dist = name, dd
        cur = {k: v for k, v in cur.items() if k != best_name}
    return traj, trees


def _trim_elbow_K(traj, jump_tol=0.10):
    """
    Selection of final number of lineages.
    """
    traj_sorted = sorted(traj)
    ns = np.array([t[0] for t in traj_sorted])
    ds = np.array([t[1] for t in traj_sorted], dtype=float) + 1e-9
    if len(ns) <= 1:
        return int(ns[0])
    ratio = ds[:-1] / ds[1:]
    catastrophic = np.where(ratio - 1.0 > jump_tol)[0]
    if catastrophic.size == 0:
        # Single lineage case.
        return int(ns[0])
    i = int(catastrophic.max())
    return int(ns[i] + 1)


def _refit_lineages(flatten_embeddings, lineages, time_length,
                    membership="endpoint"):
    """
    Reposition each surviving lineage's nodes onto the cells now assigned to it.
    """
    _, _, segment_w, _ = project_embeddings_to_tree(
        flatten_embeddings, lineages, time_length, stretch=0,
        membership=membership)
    X = np.asarray(flatten_embeddings, dtype=float)
    refit = {}
    for li, leaf in enumerate(lineages):
        P = np.asarray(lineages[leaf], dtype=float).copy()
        w = segment_w[:, li]                                   # (N,) soft membership
        d2 = (P ** 2).sum(axis=1)[None, :] - 2.0 * (X @ P.T)   # (N, m)
        nearest = d2.argmin(axis=1)
        newP = P.copy()
        for j in range(P.shape[0]):
            wj = w[nearest == j]
            sw = wj.sum()
            if sw > 1e-8:
                newP[j] = (wj[:, None] * X[nearest == j]).sum(axis=0) / sw
        newP[0] = P[0]                                          # keep root anchored
        refit[leaf] = newP
    return refit


def piecewise_linear_tree(flatten_embeddings, cell_num, time_length,
                          n_before=10, K_max=5, K_min=2,
                          use_tree=True, detour_ratio=2.0,
                          dist_drop_tol=0.10, refit_final=True):
    """
    Build a piecewise-linear trajectory graph/tree, selecting the number of
    endpoint clusters (K) by an over-split-then-trim strategy.
    """
    flatten_embeddings = _validate_embeddings(flatten_embeddings, time_length, cell_num)
    if not 1 <= K_min <= K_max <= cell_num:
        raise ValueError("Require 1 <= K_min <= K_max <= cell_num.")
    if not 1 <= n_before <= cell_num * (time_length - 2):
        raise ValueError("n_before exceeds the available intermediate observations.")

    time_full  = np.array(list(range(time_length)) * cell_num)
    label_full = np.repeat(range(1, cell_num + 1), time_length)

    over = _build_linear_tree_for_K(
        flatten_embeddings, time_full, label_full, cell_num, n_before, K_max,
        use_tree=use_tree, detour_ratio=detour_ratio)
    traj, trees = _greedy_trim_trajectory(
        flatten_embeddings, over, time_length, K_floor=1)
    target_K = min(len(over), max(_trim_elbow_K(traj, jump_tol=dist_drop_tol), K_min))
    traj_str = " ".join(f"{n}:{d:.3f}" for n, d in sorted(traj, reverse=True))
    print(f"\n{'overK':>5}  {'trim trajectory (n:dist)':<34}  {'recK':>5}")
    print("─" * 60)
    print(f"{len(over):>5}  {traj_str:<34}  {target_K:>5}")

    best_lineages = trees[target_K]
    if refit_final:
        membership = "endpoint" if use_tree else "integrated"
        best_lineages = _refit_lineages(flatten_embeddings, best_lineages, time_length,
                                        membership=membership)
    best_dist = _recon_distance(flatten_embeddings, best_lineages, time_length)

    print(f"\n→ Selected n_before={n_before}, K_actual={target_K}."
          f"(recon dist = {best_dist:.4f})\n")
    return best_lineages


def compute_segment_weights(segment_distances, time_length, eps=1e-8):
    """Hard one-hot cell-to-lineage assignment from nearest-lineage distance.
    """
    d = np.asarray(segment_distances, dtype=float)

    num_cells, num_lineages = d.shape
    if num_lineages == 1:
        segment_w_cell = np.ones((num_cells, 1), dtype=float)
        segment_w = np.repeat(segment_w_cell, time_length, axis=0)
        return segment_w, segment_w_cell, eps

    best = np.argmin(d, axis=1)
    segment_w_cell = np.zeros((num_cells, num_lineages), dtype=float)
    segment_w_cell[np.arange(num_cells), best] = 1.0
    segment_w = np.repeat(segment_w_cell, time_length, axis=0)
    return segment_w, segment_w_cell, eps



def project_embeddings_to_tree(flatten_embeddings, trajectory_tree, time_length, stretch=2.0,
                               membership="endpoint"):
    """
    Project cells onto each lineage curve and hard-assign cells to lineages.
    """
    flatten_embeddings = _validate_embeddings(flatten_embeddings, time_length)
    if membership not in ("endpoint", "integrated"):
        raise ValueError("membership must be 'endpoint' or 'integrated'.")
    if not trajectory_tree:
        raise ValueError("trajectory_tree must contain at least one lineage.")
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
        lineage_points = np.asarray(trajectory_tree[leaf], dtype=float)
        if (lineage_points.ndim != 2 or not len(lineage_points)
                or lineage_points.shape[1] != flatten_embeddings.shape[1]
                or not np.isfinite(lineage_points).all()):
            raise ValueError("Each lineage must contain finite points matching the embedding dimension.")
        if np.allclose(lineage_points, lineage_points[0]):
            squared = ((flatten_embeddings - lineage_points[0]) ** 2).sum(axis=1)
            projection_distances[:, lineage_idx] = squared
            cell_dist = squared.reshape(num_cells, time_length)
            segment_distances[:, lineage_idx] = cell_dist.sum(axis=1) if membership == 'integrated' else cell_dist[:, -1]
            continue

        # Project each point to its corresponding lineage
        d_squared, distances = p_curve.project_to_curve(flatten_embeddings, points=lineage_points, stretch=stretch)
        growth_stage_matrix[:, lineage_idx] = p_curve.pseudotimes_interp
        projection_distances[:, lineage_idx] = d_squared

        if membership == "integrated":
            pd = projection_distances[:, lineage_idx].reshape(num_cells, time_length)
            segment_distances[:, lineage_idx] = pd.sum(axis=1)
        else:
            segment_distances[:, lineage_idx] = np.array([
                projection_distances[((i+1)*time_length-1),lineage_idx]
                for i in range(num_cells)])

    segment_w, segment_w_cell, h = compute_segment_weights(segment_distances, time_length)

    return growth_stage_matrix, projection_distances, segment_w, segment_distances


def detect_bifurcations(interpolations, new_curve_time, time_seg, abs_epsilon):
    """
    Bifurcation detection.
    """
    final_l = []
    final_t = []
    include_lineage = list(interpolations.keys())
    merged_bifurcations = [(include_lineage, 0)]
    detect_matrix = np.atleast_2d(interpolations[include_lineage[0]][0]).copy() # root
    time_len_each_lineage = [len(x) for x in new_curve_time.values()]

    # Find the time range shared with all the cells
    common_time = min(time_len_each_lineage)

    # Detect bifurcations
    for i in range(common_time):
        while len(include_lineage) > 1:
            points = np.array([interpolations[j][i, :] for j in include_lineage])
            ave = np.mean(points, axis=0)
            distpoint = [
                np.dot(interpolations[j][i, :] - ave, interpolations[j][i, :] - ave)
                for j in include_lineage]
            if np.max(distpoint) > abs_epsilon:
                final_t.append(time_seg[i])
                final_l.append(include_lineage.copy())
                max_idx = np.argmax(distpoint)
                include_lineage.pop(max_idx)
            else:
                break

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

    detect_matrix[0] = interpolations[next(iter(interpolations))][0]
    return detect_matrix, interpolations, merged_bifurcations



def detect_convergences(interpolations, new_curve_time, time_seg, abs_epsilon):
    """
    Detect convergence point for non-tree topology.
    """
    lineages = list(interpolations.keys())
    n_stage = {j: interpolations[j].shape[0] for j in lineages}
    lambda_max = max(n_stage.values()) - 1

    def _present(dropped, i):
        """Lineages that reach stage i and have not been split off yet."""
        return [j for j in lineages if j not in dropped and n_stage[j] > i]

    dropped = set()
    groups = []

    for i in range(lambda_max, -1, -1):
        while True:
            include = _present(dropped, i)
            if len(include) < 2:
                break
            points = np.array([interpolations[j][i, :] for j in include])
            ave = np.mean(points, axis=0)
            distpoint = np.einsum('ij,ij->i', points - ave, points - ave)
            if np.max(distpoint) <= abs_epsilon:
                break
            groups.append((include.copy(), i))
            dropped.add(include[int(np.argmax(distpoint))])
        if len(lineages) - len(dropped) < 2:
            break

    orig = {j: interpolations[j].copy() for j in lineages}
    merge_points = []
    for group, i_lo in groups:
        hi = min(n_stage[j] for j in group)
        if i_lo >= hi:
            continue
        block = np.mean([orig[j][i_lo:hi, :] for j in group], axis=0)
        for j in group:
            interpolations[j][i_lo:hi, :] = block
        merge_points.append(block[0, :])   # upstream end of the merged suffix

    return interpolations, merge_points



def detect_merge_bifurcations(trajectory_tree, lineage_stage_indices, abs_epsilon=None,
                              allow_convergence=False, merge_epsilon_scale=1.0):
    """
        Construct shared branches.
    """

    max_stages = [np.max(stage) for stage in lineage_stage_indices] + [0]
    ordered_indices = np.argsort(max_stages)

    # Generate common time segmentation
    time_seg = np.array([0.0])
    for i in range(len(max_stages) - 1):
        start_time, end_time = max_stages[ordered_indices[i]], max_stages[ordered_indices[i + 1]]
        if end_time > start_time:
            num_points = max(2, int(np.floor((end_time - start_time) / 0.1)))
            time_seg = np.concatenate((time_seg, np.linspace(start_time, end_time, num=num_points)))
    time_seg = np.unique(time_seg)

    interpolations, new_curve_times = {}, {}
    for leaf_index, leaf in enumerate(trajectory_tree.keys()):
        leaf_curve, lambda_seg = [], time_seg[time_seg <= np.max(lineage_stage_indices[leaf_index])]
        new_curve_times[leaf] = lambda_seg

        for dim in range(trajectory_tree[leaf].shape[1]):
            xp, idx = np.unique(lineage_stage_indices[leaf_index], return_index=True)
            leaf_curve.append(np.interp(lambda_seg, xp, trajectory_tree[leaf][idx, dim]))

        interpolations[leaf] = np.array(leaf_curve).T

    # Detect bifurcations and update the shared branch
    detect_matrix, updated_interpolations, merged_bifurcation_raw = detect_bifurcations(interpolations, new_curve_times, time_seg, abs_epsilon=abs_epsilon)

    if allow_convergence:
        updated_interpolations, merge_points = detect_convergences(
            updated_interpolations, new_curve_times, time_seg,
            abs_epsilon=abs_epsilon * merge_epsilon_scale)
        for mp in merge_points:
            detect_matrix = np.vstack((detect_matrix, mp))

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



def _update_pseudotime(growth_stage_matrix, cell_num, time_length):
    """
    Update pseudotime using isotonic regression.
    """

    tau_proj = np.asarray(growth_stage_matrix, dtype=float)
    n_lin = tau_proj.shape[1]

    tau3 = tau_proj.reshape(cell_num, time_length, n_lin)
    tau_hat3 = np.zeros_like(tau3, dtype=float)

    # Time index
    t = np.arange(time_length, dtype=float)

    # Isotonic regression model
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")

    for c in range(cell_num):
        for l in range(n_lin):
            y = tau3[c, :, l]
            # Standard monotone projection
            tau_hat3[c, :, l] = iso.fit_transform(t, y)

    # Return to original shape: (C*T, L)
    tau_hat = tau_hat3.reshape(cell_num * time_length, n_lin)
    return tau_hat


def _assignment_entropy(proj_d, n_cells, time_length):
    """
        Assignment entropy calculation.
    """

    n_lineages = proj_d.shape[1]
    traj_d = proj_d.reshape(n_cells, time_length, n_lineages).sum(axis=1)

    flat  = traj_d[traj_d > 0]

    sigma = float(np.median(flat)) if flat.size > 0 else 1.0

    w = np.exp(-traj_d / (sigma + 1e-8))
    w /= w.sum(axis=1, keepdims=True)

    eps = 1e-8
    H = -(w * np.log(w + eps)).sum(axis=1)  # (n_cells,)
    return float(H.mean())




def construct_principal_tree(flatten_embeddings, time_length, cell_num,
                             sample_num=106,
                             max_iter=100,
                             epsilon=None,
                             n_before=15,
                             K_max=6,
                             K_min=1,
                             use_tree=True,
                             conv_tol=0.1,
                             span = 0.7,
                             detour_ratio=2.0,
                             dist_drop_tol=0.10,
                             refit_final = True,
                             alpha_range=[0, 0.1, 5],
                             allow_nontree=False,
                             merge_epsilon_scale=1.0):
    """
        Main algorithm.
    """

    flatten_embeddings = _validate_embeddings(flatten_embeddings, time_length, cell_num)
    if max_iter < 1 or sample_num < 2 or not 0 < span <= 1 or conv_tol < 0:
        raise ValueError("Require max_iter >= 1, sample_num >= 2, span in (0,1], and conv_tol >= 0.")
    if len(alpha_range) != 3 or int(alpha_range[2]) < 1 or not 0 <= alpha_range[0] <= alpha_range[1] <= 1:
        raise ValueError("alpha_range must be [lower, upper, count] within [0, 1].")
    if epsilon is not None and (not np.isfinite(epsilon) or epsilon < 0):
        raise ValueError("epsilon must be finite and nonnegative.")
    linear_tree = piecewise_linear_tree(flatten_embeddings, cell_num, time_length,
                                         n_before, K_max, K_min,
                                         use_tree=use_tree,
                                         detour_ratio=detour_ratio,
                                         dist_drop_tol=dist_drop_tol, refit_final=refit_final)



    membership = "integrated" if (not use_tree or allow_nontree) else "endpoint"

    def _refine(flat, ncell, start_tree, abs_e, alpha, allow_conv=allow_nontree):
        cur, prev = start_tree, np.inf
        detect, bif = None, {}
        emb_dim = flat.shape[1]
        for _ in range(max_iter):
            gsm, proj_d, seg_w, _ = project_embeddings_to_tree(
                flat, cur, time_length, stretch=0,
                membership=membership)

            flat_d   = proj_d[proj_d > 0]
            bw_point = np.median(flat_d) if flat_d.size > 0 else 1.0
            point_w  = np.array([np.exp(-proj_d[:, li] / bw_point)
                                 for li in range(proj_d.shape[1])])
            cell_w   = np.array([seg_w[:, li]
                                 for li in range(proj_d.shape[1])])

            point_w_norm = point_w / (point_w.sum(axis=0, keepdims=True) + 1e-8)
            w = alpha * point_w_norm + (1 - alpha) * cell_w
            w = w / (np.sum(w, axis=0) + 1e-8)
            cur_dist = (proj_d * w.T).mean()
            if np.abs(cur_dist - prev) < conv_tol * prev:
                break
            prev = cur_dist
            gsm = _update_pseudotime(gsm, ncell, time_length)
            cand_tree, cand_idx = _loess_update(
            cur, gsm, flat, w, sample_num, span, emb_dim, time_length)
            detect, upd_tree, _, bif = detect_merge_bifurcations(
                cand_tree, cand_idx,
                abs_epsilon=abs_e,
                allow_convergence=allow_conv,
                merge_epsilon_scale=merge_epsilon_scale)
            cur = upd_tree
        return cur, detect, bif

    if epsilon is not None:
        abs_eps = float(epsilon)
        print(f"→ fixed epsilon = {abs_eps:.6f}  (caller-supplied)\n")
    else:
        finest_tree = _refine(flatten_embeddings, cell_num, linear_tree, 0.0, alpha=0.0,
                              allow_conv=False)[0]
        R2  = _max_pairwise_lineage_scale(finest_tree)
        f   = 0.05
        abs_eps = (f ** 2) * R2
        print(f"fixed epsilon = {abs_eps:.6f}\n")

    alpha_values = np.linspace(alpha_range[0], alpha_range[1], int(alpha_range[2]))

    best_alpha             = alpha_values[0]
    best_entropy           = np.inf
    best_final_tree        = linear_tree
    best_detect_matrix     = None
    best_bifurcation_pairs = None

    header = f"\n{'alpha':>8}  {'Val Entropy':>12}"
    print(header)
    print("─" * len(header.expandtabs()))

    for alpha_cand in np.linspace(alpha_range[0], alpha_range[1], int(alpha_range[2])):
        cur, detect, bif = _refine(
            flatten_embeddings, cell_num, linear_tree, abs_eps, alpha_cand)

        _, proj_d_val, _, _ = project_embeddings_to_tree(
            flatten_embeddings, cur, time_length, stretch=0)
        entropy = _assignment_entropy(proj_d_val, cell_num, time_length)

        print(f"{alpha_cand:>8.2f}  {entropy:>12.4f}")

        if entropy <= best_entropy:
            best_entropy           = entropy
            best_alpha             = alpha_cand
            best_final_tree        = cur
            best_detect_matrix     = detect
            best_bifurcation_pairs = bif

    print(f"\n→ Selected alpha={best_alpha:.2f}  (val entropy = {best_entropy:.4f})\n")

    # Final projection on all cells using the fitted tree.
    growth_stage_matrix, projection_d, _, _ = \
        project_embeddings_to_tree(flatten_embeddings, best_final_tree,
                                   time_length, stretch=0)

    # Construction error
    construction_error = float(projection_d.min(axis=1).mean())
    print(f"→ Construction error (all-cell reconstruction) = "
          f"{construction_error:.6f}  (K={len(best_final_tree)}, "
          f"epsilon={abs_eps if abs_eps is None else round(abs_eps, 6)})\n")

    # Reorder lineages by bifurcation time ascending so Lineage 1 diverges earliest
    old_keys = list(best_final_tree.keys())
    sorted_keys = sorted(old_keys, key=lambda k: best_bifurcation_pairs.get(k, 0))
    col_order = [old_keys.index(k) for k in sorted_keys]
    best_final_tree = {f"Lineage{i+1}": best_final_tree[k] for i, k in enumerate(sorted_keys)}
    best_bifurcation_pairs = {f"Lineage{i+1}": best_bifurcation_pairs.get(k, 0.0) for i, k in enumerate(sorted_keys)}
    growth_stage_matrix = _update_pseudotime(growth_stage_matrix[:, col_order], cell_num, time_length)
    projection_d = projection_d[:, col_order]

    return (best_final_tree, growth_stage_matrix, best_detect_matrix,
            best_bifurcation_pairs, projection_d)


def _robust_loess_fit(x, y, w, span, sample_x):
    """
    LOESS algorithm.
    """
    for sp in (span, min(span * 1.5, 0.95), min(span * 2.0, 0.99)):
        try:
            reg = loess(x, y, weights=w, span=sp)
            reg.fit()
            return reg.predict(sample_x).values
        except Exception:
            continue
    ux, inv = np.unique(x, return_inverse=True)
    uy  = np.zeros_like(ux, dtype=float)
    cnt = np.zeros_like(ux, dtype=float)
    np.add.at(uy, inv, y)
    np.add.at(cnt, inv, 1.0)
    uy /= np.maximum(cnt, 1.0)
    return np.interp(sample_x, ux, uy)


def _loess_update(tree, growth_stage_matrix, embeddings, weights_all,
                  sample_num, span, embedding_dim, time_length):
    """LOESS re-fit of each lineage.
    """
    new_tree = dict(tree)
    gs = growth_stage_matrix.copy()
    start_pt = np.arange(len(embeddings)) % time_length == 0
    lineage_stage_indices = []

    for lineage_idx, leaf in enumerate(new_tree):
        w = weights_all[lineage_idx].copy()
        w[start_pt] = np.max(w) + 50
        gs[start_pt, lineage_idx] = 0

        lineage_gs = gs[:, lineage_idx]
        sorted_gs  = np.sort(lineage_gs)
        sorted_idx = np.argsort(lineage_gs)
        sample_indices = np.linspace(0, np.max(lineage_gs), num=sample_num)


        w_ = w[sorted_idx]
        new_lineage = []
        for dim in range(embedding_dim):
            y  = embeddings[sorted_idx, dim]
            new_lineage.append(
                _robust_loess_fit(sorted_gs, y, w_, span, sample_indices))

        new_tree[leaf] = np.array(new_lineage).T
        lineage_stage_indices.append(sample_indices)

    return new_tree, lineage_stage_indices



########################## Helper function for plots ###############################

def vis_tree_temporal(
    flatten_embeddings,
    trajectory_tree,
    time,
    corr_coef=None,
    bifurcation_point=None,
    save_path=None,
    visualization="2d",
    include_index=(1,),
    node_coords=None
):

    assert visualization == "2d", "Only 2D visualization supported here."
    j = int(include_index[0])  # PC{j+1} on the y-axis

    # PCA
    n_comp = max(2, j + 1)
    pca = PCA(n_components=n_comp)
    reduced_embeddings = pca.fit_transform(np.asarray(flatten_embeddings))

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter of all points colored by time
    sc = ax.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, j],
        c=np.asarray(time), s=4, alpha=0.6, linewidths=0, cmap="viridis"
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Time")

    # Draw lineage paths
    for _, lineage in trajectory_tree.items():
        lineage = np.asarray(lineage)
        if lineage.ndim == 1 and lineage.size > 0 and np.issubdtype(lineage.dtype, np.integer):
            if node_coords is None:
                pts = reduced_embeddings[lineage, :][:, [0, j]]
            else:
                pts = pca.transform(node_coords[lineage])[:, [0, j]]
        else:
            pts = pca.transform(lineage)[:, [0, j]]

        ax.plot(pts[:, 0], pts[:, 1], linewidth=2)

    if bifurcation_point is not None:
        bp = np.asarray(bifurcation_point)
        if bp.ndim == 1:
            bp = bp.reshape(1, -1)
        bp2d = pca.transform(bp)[:, [0, j]]
        ax.scatter(bp2d[:, 0], bp2d[:, 1], c='black', s=20, zorder=3)

    fig.suptitle('COTree', x=0.1, y=0.93, ha="left",
                     fontsize=12)
    if corr_coef is not None:
        ax.set_title(f"Pearson Correlation: {corr_coef}",
                     loc="left", y=1.05, fontsize=10, pad=2)
    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel(f"PC {j + 1}", fontsize=12)
    ax.tick_params(labelsize=11)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()

    return fig




def vis_tree(flatten_embeddings, trajectory_tree, label,
             bifurcation_point, type, save_path=None,
             visualization="2d", include_index=[1]):

    pca = PCA(n_components=min(5, *np.asarray(flatten_embeddings).shape))
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
    lineage_colors = {leaf: color_map.get(leaf, palette[i % len(palette)])
                      for i, leaf in enumerate(trajectory_tree)}

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

    if visualization == "2d":
        fig, ax = plt.subplots(figsize=(9.5, 5))
        j = include_index[0] if include_index else 1
        j = max(1, int(j))  # ensure y uses at least PC2

        for cat in uniques:
            m = labels_arr == cat
            ax.scatter(
                Z[m, 0], Z[m, j],
                c=color_map[cat], alpha=0.5, s=0.5, label=str(cat)
            )

        leaf_label = 1
        for leaf, lineage in trajectory_tree.items():
            try:
                L = to_reduced_coords(lineage)
                ax.plot(L[:, 0], L[:, j], color=lineage_colors[leaf],
                        label=f"Lineage {leaf_label}", linewidth=2)
                leaf_label += 1
            except Exception:
                continue

        if bifurcation_point is not None:
            try:
                BP = to_reduced_coords(np.atleast_2d(bifurcation_point))
                ax.scatter(BP[:, 0], BP[:, j], c="black", s=15, label="_nolegend_")
            except Exception:
                pass

        ax.set_title("COTree " + str(type), loc="left", fontsize=14)
        ax.set_xlabel("PC 1", fontsize=12)
        ax.set_ylabel(f"PC {j + 1}", fontsize=12)

        handles_all, labels_all = ax.get_legend_handles_labels()
        by_label = OrderedDict()
        for h, l in zip(handles_all, labels_all):
            if l == "_nolegend_":
                continue
            if l not in by_label:
                by_label[l] = h

        desired_order = [
            "Multi-replicated Cells", "Single-replicated Cells", "Unhealthy Cells",
            "Lineage 1", "Lineage 2", "Lineage 3"
        ]
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

    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        j1 = include_index[0] if include_index else 1
        j2 = include_index[1] if len(include_index) > 1 else 2

        leaf_label = 1
        for leaf, lineage in trajectory_tree.items():
            try:
                L = to_reduced_coords(lineage)
                ax.plot(L[:, 0], L[:, j1], L[:, j2], color=lineage_colors[leaf],
                        label=f"Lineage {leaf_label}", linewidth=2)
                leaf_label += 1
            except Exception:
                continue

        if bifurcation_point is not None:
            try:
                BP = to_reduced_coords(np.atleast_2d(bifurcation_point))
                ax.scatter(BP[:, 0], BP[:, j1], BP[:, j2], c="black", s=15, label="_nolegend_")
            except Exception:
                pass

        ax.set_xlabel("PC 1", fontsize=12)
        ax.set_ylabel(f"PC {j1 + 1}", fontsize=12)
        ax.set_zlabel(f"PC {j2 + 1}", fontsize=12)

        handles_all, labels_all = ax.get_legend_handles_labels()
        by_label = OrderedDict((l, h) for h, l in zip(handles_all, labels_all) if l != "_nolegend_")
        ordered_labels = list(by_label.keys())
        ordered_handles = [by_label[l] for l in ordered_labels]
        proxy_handles = make_proxy_handles(ordered_handles, ordered_labels, markersize=6)
        ax.legend(
            proxy_handles, ordered_labels,
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0, fontsize=10, markerscale=1.2,
            frameon=True, framealpha=1.0
        )

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

    return None

def vis_tree_per_lineage(flatten_embeddings, trajectory_tree, label,
                         bifurcation_point, lineage_index, save_path=None,
                         include_index=(1,), *, time_length=106):
    """Plot individual cells in a lineage; specify their actual time_length."""
    X = _validate_embeddings(flatten_embeddings, time_length)
    j = int(include_index[0])
    if j < 1 or j >= min(X.shape):
        raise ValueError("Requested principal component is unavailable.")
    pca = PCA(n_components=max(2, j + 1))
    Z = pca.fit_transform(X).reshape(-1, time_length, max(2, j + 1))
    labels = np.asarray(label)
    if labels.size == len(X):
        labels = labels.reshape(-1, time_length)
        if not np.all(labels == labels[:, :1]):
            raise ValueError("Each cell must have a single lineage label.")
        labels = labels[:, 0]
    elif labels.size != len(Z):
        raise ValueError("Supply one label per cell or per observation.")
    name = f"Lineage{lineage_index}"
    if name not in trajectory_tree:
        raise ValueError(f"Unknown lineage {name}.")
    color = sns.color_palette('tab10')[(lineage_index - 1) % 10]
    fig, ax = plt.subplots(figsize=(6, 4))
    for cell in Z[labels == name]:
        ax.plot(cell[:, 0], cell[:, j], color=color, alpha=0.1)
    curve = pca.transform(trajectory_tree[name])
    ax.plot(curve[:, 0], curve[:, j], color=color, linewidth=2,
            label=f"Lineage {lineage_index}")
    if bifurcation_point is not None:
        nodes = pca.transform(np.atleast_2d(bifurcation_point))
        ax.scatter(nodes[:, 0], nodes[:, j], c='black', s=15)
    ax.set(title=f"Trajectory Tree — Lineage {lineage_index}",
           xlabel='PC 1', ylabel=f'PC {j + 1}')
    ax.legend()
    sns.despine()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    else:
        plt.show()
    return ax


# Clustering
def lineage_clustering(projection_distances, time_length, trajectory_tree):
    distances_to_lineage = projection_distances.reshape(-1, time_length, projection_distances.shape[-1])
    leaf = np.array(list(trajectory_tree.keys()))
    clustering_label = leaf[np.argmin(np.sum(distances_to_lineage,axis=1), axis=1)]
    return clustering_label

# Time inference
def vis_predictive_time(trajectory_tree,
                      projection_matrix,
                      bifurcation_nodes,
                      clustering_labels,
                      time_length,
                      save_path=None):

    leaf = np.array(list(trajectory_tree.keys()))
    lineage_num = len(trajectory_tree)
    cell_projection_matrix = projection_matrix.reshape(-1, time_length, lineage_num)

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
        else:
            predictive_time_all[lineage_label].append(0)

    order = {f"Lineage {i+1}": f"Lineage{i+1}" for i in range(len(leaf))}
    plot_dic = {key: predictive_time_all[order[key]] for key in order.keys()}
    fig, ax = plt.subplots(figsize=(4, 3))
    color_list = sns.color_palette("pastel", n_colors=lineage_num)

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
    ax.set_ylim(-1, time_length + 1)
    ax.set_ylabel('Time step', fontsize=9)
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
                   bifurcation_nodes,
                   clustering_labels,
                   time_length,
                   save_path=None):

    lineage_num = len(trajectory_tree)
    cell_growth_time_matrix = growth_stage_matrix.reshape(-1, time_length, lineage_num)

    retrospective_time_all = defaultdict(list)

    leaf_list = np.array(list(trajectory_tree.keys()))
    for cell_index, cell in enumerate(cell_growth_time_matrix):
        lineage_label = clustering_labels[cell_index]
        i = np.where(leaf_list == lineage_label)[0][0]

        cell_time = cell[:,i]
        crossings = np.flatnonzero(cell_time > bifurcation_nodes.get(lineage_label, 0.0))
        enter_time = int(crossings[0]) if crossings.size else np.nan

        retrospective_time_all[lineage_label].append(enter_time)

    order = {f"Lineage {i+1}": f"Lineage{i+1}" for i in range(len(leaf_list))}
    plot_dic = {key: retrospective_time_all[order[key]] for key in order.keys()}

    fig, ax = plt.subplots(figsize=(4, 3))

    color_list = sns.color_palette("pastel", n_colors=lineage_num)

    if any(np.isfinite(v).any() for v in plot_dic.values()):
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

    else:
        ax.text(0.5, 0.5, 'No cells crossed the bifurcation time',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks(range(lineage_num), labels=list(plot_dic))

    ax.set_xticks(range(lineage_num))
    ax.set_ylim(-1, time_length + 1)
    ax.set_ylabel('Time step', fontsize=9)
    ax.set_title('Retrospective Time', fontsize=11, pad=10)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return retrospective_time_all


def _validate_embeddings(embeddings, time_length, cell_num=None):
    X = np.asarray(embeddings, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("embeddings must be a nonempty finite 2D array.")
    if int(time_length) != time_length or time_length < 3 or len(X) % time_length:
        raise ValueError("time_length must be >= 3 and divide the number of embedding rows.")
    if cell_num is not None and (cell_num < 1 or cell_num * time_length != len(X)):
        raise ValueError("cell_num * time_length must equal the number of embedding rows.")
    return X
