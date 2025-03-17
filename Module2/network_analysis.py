import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import pandas as pd
from collections import OrderedDict
# Load the data
with open("cleaned_artists_data.json", "r") as file:
    data = json.load(file)

# Create a graph
G = nx.Graph()
collaboration_counts = {}

# Add nodes and edges
for artist in data:
    artist_name = artist["artist_name"]

    # Calculate total collaborations for node sizing
    total_collabs = sum(artist["artist_collaborations"].values())
    collaboration_counts[artist_name] = total_collabs

    # Ensure the artist node exists
    G.add_node(artist_name)

    for collaborator, weight in artist["artist_collaborations"].items():
        # Ensure collaborator node exists
        if collaborator not in collaboration_counts:
            collaboration_counts[collaborator] = 0  # Avoid KeyError

        # Add edge
        G.add_edge(artist_name, collaborator)

# Assign node sizes based on total collaborations
# node_sizes = [collaboration_counts[n] * 30 for n in G.nodes()]

# Use a colormap for distinct colors
cmap = cm.get_cmap("tab20")  # 20 distinct colors
colors = {artist: mcolors.rgb2hex(cmap(i % 20))
          for i, artist in enumerate(G.nodes())}

# Generate layout with more spacing
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=0.5, seed=42)  # Increase `k` to spread nodes apart

# Draw nodes (colored based on artist names)
node_colors = [colors[n] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, #node_size=node_sizes,
                       node_color=node_colors, edgecolors="black", alpha=0.85)

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)

# Draw labels with the same color as their nodes
for artist, (x, y) in pos.items():
    text = plt.text(
        x, y, artist, fontsize=10, fontweight="bold", color=colors[artist], ha="center", va="center"
    )
    text.set_path_effects([path_effects.Stroke(
        linewidth=2, foreground="white"), path_effects.Normal()])

# Add title
plt.title(
    "Collaborations Between Artists in Spotify's Top Artists of 2024 Global Playlist", fontsize=14)

plt.savefig("./Outputs/spotify_artists_colab_graph.png")

# calculate degree centrality (dictionary with artists' names as keys and values as 
# the fraction of nodes each artist/node is connected to)
degree_centrality = nx.degree_centrality(G)
combined_dict = {}
for key in collaboration_counts:
    combined_dict[key] = {
        "Degree_Centrality": degree_centrality.get(key, 0),
        # "Number_of_Track_Collaborations": collaboration_counts.get(key, 0)
    }

# Convert to DataFrame
df = pd.DataFrame.from_dict(combined_dict, orient='index')

# Ensure correct column naming
df.index.name = "Artist_Name"  # Set index name instead of renaming columns

# Sort by "Number of Collaborations" in descending order
df_sorted = df.sort_values(by="Degree_Centrality", ascending=False)

# Write results to CSV with the correct format
df_sorted.to_csv('./Outputs/artists_sorted_by_degree_centrality.csv',
                 index=True, index_label="Artist_Name")