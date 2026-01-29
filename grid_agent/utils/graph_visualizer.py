import networkx as nx
import matplotlib.pyplot as plt

def plot_network(net, violations):
    G = nx.Graph()

    for bus in net.bus.index:
        G.add_node(int(bus))

    for _, line in net.line.iterrows():
        if line.in_service:
            G.add_edge(int(line.from_bus), int(line.to_bus))

    node_colors = []
    for bus in G.nodes():
        if bus in violations.get("voltage", []):
            node_colors.append("red")
        else:
            node_colors.append("green")

    fig, ax = plt.subplots(figsize=(10, 6))

    _ = nx.draw(
        G,
        ax=ax,
        node_color=node_colors,
        with_labels=True,
        node_size=600,
        font_size=8
    )

    ax.set_title("Distribution Network Topology")

    return fig   # ⬅️ MUST BE LAST LINE
