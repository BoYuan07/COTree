import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .DTMAE import seperate_output
from ._validation import integer_index
from ._model_api import encode_gmp, predict_gmp
from kneed import  KneeLocator
import numpy as np




def function_module_estimation(model, train_dataset, index, lambda_reg, input_size, device,
                               learning_rate=0.01, epochs=100):


    index = integer_index(index)
    if len(train_dataset) == 0 or epochs < 1:
        raise ValueError('Nonempty training data and positive epochs are required.')
    model.to(device).eval()
    f = torch.randn(input_size['G'] + input_size['M'] + input_size['P'], device=device)
    f = f/torch.norm(f)
    f.requires_grad_()

    dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    optimizer = torch.optim.Adam([f], lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        cosine_losses = []

        for batch in dataloader:
            G, M, P = (batch[0][m] for m in 'GMP')
            G = G.to(device)
            M = M.to(device)
            P = P.to(device)

            N, _ = G.shape

            v = f.unsqueeze(0).expand(N, -1)

            Z = encode_gmp(model, G, M, P)

            X = torch.cat((G, M, P), dim=-1)  # shape: [N, d]
            transformed_X = X + v

            assert transformed_X.shape == X.shape, "Shape mismatch after transformation"

            G_new, M_new, P_new, _ = seperate_output(transformed_X, input_size)
            G_new = G_new.to(device)
            M_new = M_new.to(device)
            P_new = P_new.to(device)

            all_embedding = encode_gmp(model, G_new, M_new, P_new)
            f_transformed_X = all_embedding.to(Z.device)

            h = f_transformed_X - Z  # shape: [N, d_embed]

            if index >= h.shape[1]:
                raise IndexError(f"index {index} out of bounds for embedding dim {h.shape[1]}")

            numerator = h[:, index]
            denominator = torch.norm(h, dim=1) + 1e-8  # shape: [N]
            loss_terms = numerator / denominator  # shape: [N]
            cosine_loss = torch.mean(loss_terms)

            batch_loss = -cosine_loss + lambda_reg * torch.norm(f, p=1)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            cosine_losses.append(cosine_loss.item())

    print(f"Index {index},"
            f"Mean Cosine: {sum(cosine_losses)/len(cosine_losses):.6f}, "
            f"Reg: {(lambda_reg * torch.norm(f, p=1)).item():.6f}")

    return f.detach().cpu().numpy()



def functional_module_species(G, M, P, v, model, index, input_size, device):

    index = integer_index(index)
    model.to(device).eval()

    def compute_loss_fn(G,M,P,v,index):
        G = G.to(device)
        M = M.to(device)
        P = P.to(device)
        v = v.to(device)

        N, _ = G.shape

        Z = encode_gmp(model, G, M, P)

        X = torch.cat((G, M, P), dim=-1)
        transformed_X = X + v

        G_new, M_new, P_new, _ = seperate_output(transformed_X, input_size)
        G_new = G_new.to(device)
        M_new = M_new.to(device)
        P_new = P_new.to(device)

        all_embedding = encode_gmp(model, G_new, M_new, P_new)
        f_transformed_X = all_embedding.to(Z.device)

        h = f_transformed_X - Z
        numerator = h[:, index]
        denominator = torch.norm(h, dim=1) + 1e-8
        loss_terms = numerator / denominator
        cosine_loss = torch.mean(loss_terms)

        return -cosine_loss

    species_score = np.abs(v)
    taus = np.linspace(np.min(species_score),np.max(species_score),100)
    cosine_diff = []
    selected_tau = 0
    v = torch.tensor(v, dtype=torch.float32)
    baseline_loss = compute_loss_fn(G,M,P,v,index)

    # Guard against a degenerate score distribution (all species_score equal ->
    # every tau identical -> flat curve that no knee detector can handle).
    if np.ptp(species_score) == 0:
        selected_tau = float(species_score.min())
        selected_index = np.where(species_score >= selected_tau)[0]
        print(f"Functional Module: {index+1} Finished (uniform scores, no threshold).")
        return selected_tau, selected_index, [], taus

    break_tau = None  # tau at which the 10% loss-recovery criterion was met
    for tau in taus:
        new_v = v.clone()
        idx = np.where(species_score < tau)[0]
        new_v[idx] = 0.0
        trial_loss = compute_loss_fn(G,M,P,new_v,index)
        cosine_diff.append(trial_loss.item()-baseline_loss.item())
        denom = baseline_loss.item()
        if denom != 0 and -(trial_loss.item()-baseline_loss.item())/denom >= 0.1:
            break_tau = tau
            # Keep collecting a few points past the break so KneeLocator has a
            # real curve to bend on; the early break alone often leaves only 1-2
            # points, for which knee.elbow is None (or crashes at a single point).
            if len(cosine_diff) >= 8:
                break

    selected_tau = None
    # KneeLocator needs enough points and a non-flat curve; skip it otherwise.
    if len(cosine_diff) >= 4 and np.ptp(cosine_diff) > 0:
        knee = KneeLocator(taus[:len(cosine_diff)], cosine_diff, curve="convex",
                           interp_method="polynomial", polynomial_degree=2)
        selected_tau = knee.elbow

    if selected_tau is None:
        # No detectable knee. Fall back to the 10% loss-recovery tau (a meaningful,
        # deterministic cutoff); if that never fired, keep all species (tau=min).
        selected_tau = break_tau if break_tau is not None else float(species_score.min())

    selected_index = np.where(species_score >= selected_tau)[0]
    print(f"Functional Module: {index+1} Finished.")

    return selected_tau, selected_index, cosine_diff, taus








def function_score_estimation(model, index, dataset, training_data, device="cpu"):
    """Prediction changes after removing one latent coordinate.

    Pass the fitted flexible ``training_data`` to reuse its feature masks and
    scales. ``index`` follows ``model.encoded_modalities`` order. Scores follow
    the processed G, M, P feature order.
    """
    index = integer_index(index)
    inputs = training_data.transform_samples(dataset.samples)
    G, M, P, R = (inputs[m] for m in 'GMPR')
    model.to(device).eval()
    with torch.no_grad():
        G, M, P, R = (t.to(device) for t in (G, M, P, R))
        Z = encode_gmp(model, G, M, P)
        X = predict_gmp(model, Z, R)
        Z = Z.clone()
        Z[:, index] = 0
        X_turb = predict_gmp(model, Z, R)
        delta = (X_turb - X).norm(p=2, dim=0) / (X.norm(p=2, dim=0) + 1e-8)
    return delta.detach().cpu().numpy()


def plot_metabolite_reaction(fn_ind, functional_module, file, function_module_val, seed=40, figsize=(10, 7), display_reaction_list = None, *, sbml_path):
    """Plot module reactions from an explicit SBML file; ``file`` is the output image.

    Reaction plotting dependencies are included in the standard installation.
    The three panels use the manuscript model's reaction ordering: the first
    24 reactions, the next 46 reactions, and the remaining reactions.
    """
    import cobra
    import networkx as nx
    model = cobra.io.read_sbml_model(str(sbml_path))
    display_reaction_list = set(display_reaction_list or [])

    metabolites = [m[2:] if m.startswith("M_") else m for m in functional_module[f"Functional Module {fn_ind}"]]

    ranges = [(0, 24), (24, 70), (70, len(model.reactions))]
    titles = ["Central Reactions", "Nucleotide Reactions", "Lipid Reactions"]

    # --- Layout: 2 plots on top, 1 centered below (same width) ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

    # Top row: two plots
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    # Bottom row: one centered plot (middle column)
    ax3 = fig.add_subplot(gs[1, 1:3])
    axes = [ax1, ax2, ax3]

    for idx, (start, end) in enumerate(ranges):
        G = nx.DiGraph()
        edge_labels = {}

        # Build graph connections
        for rxn in model.reactions[start:end]:
            rxn_mets = [met.id for met in rxn.metabolites]
            shared = [m for m in metabolites if m in rxn_mets]
            for i in range(len(shared)):
                for j in range(i + 1, len(shared)):
                    a, b = shared[i], shared[j]
                    G.add_edge(a, b)
                    if rxn.id in display_reaction_list:
                        edge_labels[(a, b)] = rxn.id

        # Keep connected nodes only
        connected_nodes = list(set([n for edge in G.edges() for n in edge]))
        G = G.subgraph(connected_nodes)

        # Color & size nodes based on module values
        node_colors, node_sizes = [], []
        for node in G.nodes():
            try:
                idx_f = metabolites.index(node)
                val = function_module_val[f"Functional Module {fn_ind}"][idx_f]
                color = 'lightblue' if val > 0 else "#F08080"
                size = 300 + 700 * abs(val)
            except ValueError:
                color = 'gray'
                size = 200
            node_colors.append(color)
            node_sizes.append(size)

        pos = nx.circular_layout(G)
        ax = axes[idx]
        labels = {node: node.removesuffix("_c") for node in G.nodes()}

        # Draw network
        nx.draw(
            G, pos, ax=ax, with_labels=True,
            labels=labels,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=7, arrowsize=1, alpha=0.8
        )

        # Add edge labels
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=edge_labels,
            font_size=6, rotate=False, label_pos=0.5
        )

        ax.set_title(titles[idx], fontsize=10)
        ax.axis("off")

    # Overall figure title and spacing
    fig.suptitle(f"Metabolite Module {fn_ind}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(file, dpi=300)
    plt.show()
