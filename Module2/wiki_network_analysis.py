import requests
import time
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

# Wikipedia API base URL
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

cryptography_articles = ["Cryptography",
                         "Encryption", "Public-key_cryptography"]
privacy_law_articles = ["Privacy_law", "Data_privacy",
                        "General_Data_Protection_Regulation"]

def get_links(article_title):
    """
    Fetch outgoing links from a Wikipedia article and clean the data.
    """
    params = {
        "action": "query",
        "titles": article_title,
        "prop": "links",
        "pllimit": "max",
        "format": "json"
    }

    response = requests.get(WIKI_API_URL, params=params)
    data = response.json()

    # Extract article links
    links = []
    pages = data.get('query', {}).get('pages', {})
    for page_id, page_info in pages.items():
        if 'links' in page_info:
            for link in page_info['links']:
                link_title = link['title']

                # Clean the data: exclude disambiguation, admin, and non-article links
                if is_valid_article(link_title):
                    links.append(link_title)

    return links


def is_valid_article(link_title):
    """
    Check if a link is valid (not a disambiguation, list, or non-article link).
    """
    # Exclude disambiguation, list, or special administrative pages
    invalid_keywords = [
        'disambiguation', 'List of', 'User:', 'Talk:', 'Special:', 'File:',
        'Template:', 'Wikipedia:', 'Help:', 'Portal:', 'Category:'
    ]

    # Ensure the link doesn't contain any invalid keywords
    return not any(keyword in link_title for keyword in invalid_keywords)

def build_network(seed_articles):
    """
    Build a network graph from seed articles and their outgoing links.
    """
    G = nx.Graph()

    for article in seed_articles:
        links = get_links(article)
        for linked_article in links:
            G.add_edge(article, linked_article)
        time.sleep(0.5)  # To avoid hitting the API rate limit

    return G

# Combine cryptography and privacy law articles
all_seed_articles = cryptography_articles + privacy_law_articles

# Build the article network
article_network = build_network(all_seed_articles)

# Betweenness centrality to find bridging articles
centrality = nx.betweenness_centrality(article_network)

# Sort articles by centrality
sorted_centrality = sorted(
    centrality.items(), key=lambda x: x[1], reverse=True)

# Select the top 50 most central nodes for clearer visualization
top_central_nodes = [node for node, _ in sorted_centrality[:50]]
filtered_network = article_network.subgraph(top_central_nodes)

### Enhanced Visualization ###
plt.figure(figsize=(12, 12))

# Generate a spring layout for positioning nodes with more spacing
# Increased k for more spacing
pos = nx.spring_layout(filtered_network, k=0.25, iterations=100)

# Get node sizes based on betweenness centrality (scaled up for better visualization)
node_sizes = [centrality[node] *
              3000 if node in centrality else 100 for node in filtered_network.nodes()]

# Apply a color map to nodes based on centrality
cmap = cm.viridis
node_colors = [centrality[node]
               if node in centrality else 0 for node in filtered_network.nodes()]

# Draw the nodes with size and color gradients based on centrality
nx.draw_networkx_nodes(filtered_network, pos, node_color=node_colors,
                       node_size=node_sizes, cmap=cmap, alpha=0.9)

# Draw edges with transparency to reduce clutter
nx.draw_networkx_edges(filtered_network, pos, alpha=0.2)

# Adjust label positions slightly to avoid overlap
def adjust_text_position(pos, offset=(0.02, 0.02)):
    adjusted_pos = {}
    for node, (x, y) in pos.items():
        adjusted_pos[node] = (x + offset[0], y + offset[1])
    return adjusted_pos


# Adjust label positions to avoid squished text
adjusted_pos = adjust_text_position(pos)

# Only display labels for the most central nodes with a background for better visibility
important_nodes = {node: node for node, _ in sorted_centrality[:20]}
nx.draw_networkx_labels(filtered_network, adjusted_pos, labels=important_nodes,
                        font_size=10, font_color='yellow', font_weight='bold',
                        # Add background color
                        bbox=dict(facecolor='black', edgecolor='none', alpha=0.7))

# Print the betweenness centrality values for the most central nodes (top 20)
print("\nBetweenness Centrality for Top 20 Most Central Nodes:")
for node, centrality_score in sorted_centrality[:20]:
    print(f"Node: {node}, Betweenness Centrality: {centrality_score}")

# Add a color bar to show node centrality values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
    vmin=min(node_colors), vmax=max(node_colors)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label='Betweenness Centrality')

# Add a title to the graph
plt.title('Key Articles in Cryptography and Privacy Law: A Wikipedia Network Analysis', size=15)
# Display the graph
plt.axis('off')  # Turn off the axis for better visualization
plt.savefig("graph.png")
