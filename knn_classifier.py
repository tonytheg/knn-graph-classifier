import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import os
from collections import Counter, deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'


def load_graph(filepath):
    G = nx.Graph()
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G


def extract_features(G):
    nodes = sorted(G.nodes())
    deg = dict(G.degree())
    cc = nx.clustering(G)
    avg_nd = nx.average_neighbor_degree(G)

    X = []
    for n in nodes:
        X.append([cc[n], avg_nd[n]])

    # label: high degree (above median) = 1, low degree = 0
    degrees = [deg[n] for n in nodes]
    median_deg = np.median(degrees)
    y = [1 if deg[n] > median_deg else 0 for n in nodes]

    return np.array(X), np.array(y)


def extract_features_fast(G):
    """Extract a low-cost feature without reusing degree as an input.

    Degree defines the synthetic high/low-degree target, so including degree in
    the feature matrix would leak the answer into the scalability benchmark.
    Average neighbor degree remains inexpensive to compute and independent of
    the target definition for the node being classified.
    """
    nodes = sorted(G.nodes())
    deg = dict(G.degree())
    avg_nd = nx.average_neighbor_degree(G)

    X = []
    for n in nodes:
        X.append([avg_nd[n]])

    degrees = [deg[n] for n in nodes]
    median_deg = np.median(degrees)
    y = [1 if deg[n] > median_deg else 0 for n in nodes]

    return np.array(X), np.array(y)


def bfs_sample(G, start_node, size):
    """Return up to ``size`` nodes discovered by breadth-first search."""
    visited = {start_node}
    queue = deque([start_node])

    while len(visited) < size and queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) >= size:
                    break

    return visited


def normalize(X_train, X_test):
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1
    return (X_train - mins) / rng, (X_test - mins) / rng


def knn_predict(X_train, y_train, X_test, k):
    preds = []
    for i in range(len(X_test)):
        dists = np.sqrt(np.sum((X_train - X_test[i]) ** 2, axis=1))
        idx = np.argsort(dists)[:k]
        neighbors = y_train[idx]
        counts = Counter(neighbors.tolist())
        preds.append(counts.most_common(1)[0][0])
    return np.array(preds)


def compute_f1(y_true, y_pred):
    classes = sorted(set(y_true) | set(y_pred))

    f1_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f1 = sum(f1_scores) / len(f1_scores)

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0

    return macro_f1, micro_f1


def split_data(X, y, test_ratio=0.3, seed=42):
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    test_size = int(n * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Small Dataset: CA-GrQc
    print("=" * 50)
    print("Loading CA-GrQc dataset...")
    G_small = load_graph(DATA_DIR / 'CA-GrQc.txt')
    print("Nodes:", G_small.number_of_nodes(), " Edges:", G_small.number_of_edges())

    print("Extracting features...")
    X, y = extract_features(G_small)
    print("Feature matrix shape:", X.shape)
    print("Class distribution:", Counter(y.tolist()))

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_n, X_test_n = normalize(X_train, X_test)

    # effectiveness test
    print("\n--- Effectiveness Test (CA-GrQc) ---")
    k_values = [1, 3, 5, 7, 9]
    macro_scores = []
    micro_scores = []
    run_times = []

    for k in k_values:
        t0 = time.time()
        y_pred = knn_predict(X_train_n, y_train, X_test_n, k)
        t1 = time.time()

        macro_f1, micro_f1 = compute_f1(y_test.tolist(), y_pred.tolist())
        macro_scores.append(macro_f1)
        micro_scores.append(micro_f1)
        run_times.append(t1 - t0)

        print("k=%d: Macro-F1=%.4f, Micro-F1=%.4f, Time=%.2fs" % (k, macro_f1, micro_f1, t1 - t0))

    # plot effectiveness
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, macro_scores, 'o-', label='Macro-F1')
    ax.plot(k_values, micro_scores, 's-', label='Micro-F1')
    ax.set_xlabel('k')
    ax.set_ylabel('F1 Score')
    ax.set_title('k-NN Effectiveness on CA-GrQc')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'effectiveness.png', dpi=150)
    plt.close()
    print("Saved results/effectiveness.png")

    # plot efficiency
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(k) for k in k_values], run_times, color='steelblue')
    ax.set_xlabel('k')
    ax.set_ylabel('Running Time (seconds)')
    ax.set_title('k-NN Efficiency on CA-GrQc')
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'efficiency.png', dpi=150)
    plt.close()
    print("Saved results/efficiency.png")

    # Large Dataset: com-dblp (Scalability) 
    print("\n" + "=" * 50)
    print("Loading com-dblp dataset for scalability test...")
    large_dataset = DATA_DIR / 'com-dblp.ungraph.txt'
    if not large_dataset.exists():
        print("Optional dataset not found; skipping scalability benchmark.")
        print("To enable it, place com-dblp.ungraph.txt in the data/ directory.")
        print("\nDone. Check the results/ folder for plots.")
        raise SystemExit(0)

    G_large = load_graph(large_dataset)
    print("Nodes:", G_large.number_of_nodes(), " Edges:", G_large.number_of_edges())

    sample_sizes = [2000, 5000, 10000]
    scale_times = []
    scale_macro = []
    scale_micro = []
    actual_sizes = []

    # pick a high-degree node as starting point for BFS sampling
    start_node = max(G_large.nodes(), key=lambda n: G_large.degree(n))

    for size in sample_sizes:
        print("\nBFS sampling %d nodes..." % size)
        visited = bfs_sample(G_large, start_node, size)

        G_sub = G_large.subgraph(list(visited)).copy()
        # remove isolated nodes just in case
        isolated = [n for n in G_sub.nodes() if G_sub.degree(n) == 0]
        G_sub.remove_nodes_from(isolated)
        print("  Nodes: %d, Edges: %d" % (G_sub.number_of_nodes(), G_sub.number_of_edges()))

        if G_sub.number_of_nodes() < 50:
            print("  Too few nodes, skipping...")
            continue

        t_start = time.time()
        X_s, y_s = extract_features_fast(G_sub)
        X_tr, X_te, y_tr, y_te = split_data(X_s, y_s)
        X_tr_n, X_te_n = normalize(X_tr, X_te)

        y_pred_s = knn_predict(X_tr_n, y_tr, X_te_n, 5)
        t_total = time.time() - t_start

        macro_f1, micro_f1 = compute_f1(y_te.tolist(), y_pred_s.tolist())
        scale_times.append(t_total)
        scale_macro.append(macro_f1)
        scale_micro.append(micro_f1)
        actual_sizes.append(G_sub.number_of_nodes())

        print("  k=5: Macro-F1=%.4f, Micro-F1=%.4f, Time=%.2fs" % (macro_f1, micro_f1, t_total))

    # plot scalability
    if scale_times:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(actual_sizes, scale_times, 'o-', color='steelblue')
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Running Time (seconds)')
        ax1.set_title('Scalability: Running Time')
        ax1.grid(True)

        ax2.plot(actual_sizes, scale_macro, 'o-', label='Macro-F1')
        ax2.plot(actual_sizes, scale_micro, 's-', label='Micro-F1')
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Scalability: F1 Scores')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'scalability.png', dpi=150)
        plt.close()
        print("Saved results/scalability.png")

    print("\nAll done. Check the results/ folder for plots.")
