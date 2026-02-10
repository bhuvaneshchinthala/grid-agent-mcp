import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_network(net, violations, style="modern"):
    """
    Plot network topology with multiple style options.
    
    Args:
        net: pandapower network
        violations: dict of violations
        style: 'modern', 'classic', 'dark', or 'minimal'
    """
    G = nx.Graph()

    for bus in net.bus.index:
        G.add_node(int(bus))

    for _, line in net.line.iterrows():
        if line.in_service:
            G.add_edge(int(line.from_bus), int(line.to_bus))

    # Color nodes based on violations
    node_colors = []
    node_sizes = []
    for bus in G.nodes():
        if bus in violations.get("voltage", []):
            node_colors.append("#FF6B6B")  # Red for voltage violations
            node_sizes.append(800)
        elif bus in violations.get("thermal", []):
            node_colors.append("#FFA500")  # Orange for thermal violations
            node_sizes.append(750)
        else:
            node_colors.append("#4ECDC4")  # Teal for healthy
            node_sizes.append(600)

    # Choose layout based on style
    if style == "modern":
        return _plot_modern(G, node_colors, node_sizes, net, violations)
    elif style == "dark":
        return _plot_dark(G, node_colors, node_sizes, net, violations)
    elif style == "minimal":
        return _plot_minimal(G, node_colors, node_sizes, net, violations)
    else:  # classic
        return _plot_classic(G, node_colors, node_sizes, net, violations)


def _plot_modern(G, node_colors, node_sizes, net, violations):
    """Modern style with spring layout and gradient background"""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#f8f9fa')
    ax.set_facecolor('#ffffff')
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#cccccc', width=1.5, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=node_sizes, edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold', font_color='white')
    
    ax.set_title("Network Topology - Modern Style", fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#4ECDC4', edgecolor='white', label='Healthy'),
        mpatches.Patch(facecolor='#FFA500', edgecolor='white', label='Thermal Violation'),
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='white', label='Voltage Violation')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    return fig


def _plot_dark(G, node_colors, node_sizes, net, violations):
    """Dark mode with circular layout"""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    
    pos = nx.circular_layout(G)
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#444444', width=1.5, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=node_sizes, edgecolors='#e0e0e0', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold', font_color='#ffffff')
    
    ax.set_title("Network Topology - Dark Mode", fontsize=16, fontweight='bold', 
                color='#ffffff', pad=20)
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#4ECDC4', edgecolor='#e0e0e0', label='Healthy'),
        mpatches.Patch(facecolor='#FFA500', edgecolor='#e0e0e0', label='Thermal Violation'),
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='#e0e0e0', label='Voltage Violation')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
             framealpha=0.9, facecolor='#1a1a2e', edgecolor='#444444', labelcolor='#ffffff')
    
    return fig


def _plot_minimal(G, node_colors, node_sizes, net, violations):
    """Minimal style with hierarchical layout"""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Use hierarchical layout
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#999999', width=1, alpha=0.4, style='dashed')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=node_sizes, edgecolors='#333333', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold', font_color='white')
    
    ax.set_title("Network Topology - Minimal Style", fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#4ECDC4', edgecolor='#333333', label='Healthy'),
        mpatches.Patch(facecolor='#FFA500', edgecolor='#333333', label='Thermal Violation'),
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='#333333', label='Voltage Violation')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    return fig


def _plot_classic(G, node_colors, node_sizes, net, violations):
    """Classic matplotlib style"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=1.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=node_sizes, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    
    ax.set_title("Distribution Network Topology", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    return fig
