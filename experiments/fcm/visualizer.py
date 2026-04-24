"""
FCM network visualization
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_fcm_radial_graph(W, names, output_path, threshold=0.2):

    target = names[-1]
    features = names[:-1]

    G = nx.DiGraph()

    for node in names:
        G.add_node(node)

    for i in range(len(names)):
        for j in range(len(names)):

            if i == j:
                continue

            weight = W[i, j]

            if abs(weight) >= threshold:
                G.add_edge(names[i], names[j], weight=weight)

    pos = {}

    pos[target] = (0, 0)

    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)

    radius = 4

    for i, feature in enumerate(features):

        pos[feature] = (
            radius * np.cos(angles[i]),
            radius * np.sin(angles[i])
        )

    plt.figure(figsize=(10, 10))

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[target],
        node_color="red",
        node_size=2500
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=features,
        node_color="skyblue",
        node_size=1200
    )

    nx.draw_networkx_edges(G, pos, arrows=True)

    nx.draw_networkx_labels(G, pos, font_size=8)

    edge_labels = {
        (u, v): f"{d['weight']:.2f}"
        for u, v, d in G.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=6
    )

    plt.axis("off")

    plt.title("FCM Network Graph")

    plt.savefig(output_path, dpi=300)

    plt.close()