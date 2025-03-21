import csv
import requests
import time
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

# Wikipedia API base URL
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

def search_articles(query, max_results):
    """
    Search Wikipedia for articles related to a specific query and return the top article titles.
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max_results,
        "format": "json"
    }

    response = requests.get(WIKI_API_URL, params=params)
    data = response.json()

    articles = [result['title'] for result in data['query']['search']]
    return articles


# Wikipedia API base URL
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"


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

    links = []
    pages = data.get('query', {}).get('pages', {})
    for page_info in pages.values():
        if 'links' in page_info:
            for link in page_info['links']:
                link_title = link['title']
                if is_valid_article(link_title):
                    links.append(link_title)

    return links


def is_valid_article(link_title):
    """
    Check if a link is valid (not a disambiguation, list, or non-article link).
    """
    
    invalid_keywords = [
        'disambiguation', 'List of', 'User:', 'Talk:', 'Special:', 'File:',
        'Template:', 'Wikipedia:', 'Help:', 'Portal:', 'Category:', 'ISBN', 'Doi', 'film',
        'term', 'Call of Duty', 'Kids','LCCN', 'ISSN'
    ]
    return not any(keyword in link_title for keyword in invalid_keywords)



def build_network(seed_articles):
    """
    Build a network graph from seed articles and their outgoing links
    """
    exclusion_keywords = ['World War II', 'Cold War', 'Tanks', 'Battle', 'Causes', 'Second']
    G = nx.Graph()

    for article in seed_articles:
        if not is_valid_article(article) or any(article.lower().startswith(keyword.lower()) for keyword in exclusion_keywords):
            continue  # Skip invalid or excluded articles

        links = get_links(article)
        for linked_article in links:
            if not is_valid_article(linked_article) or any(linked_article.lower().startswith(keyword.lower()) for keyword in exclusion_keywords):
                continue  # Skip invalid or excluded links

            G.add_edge(article, linked_article)

        time.sleep(0.5)  # To avoid hitting the API rate limit

    return G

# Search for the top 10 articles related to WW2 and the Cold War
ww2_articles = search_articles("World War II", max_results=10)
cold_war_articles = search_articles("Cold War", max_results=10)

# Combine articles
all_seed_articles = ww2_articles + cold_war_articles
# Build the article network
article_network = build_network(all_seed_articles)

# Betweenness centrality to find bridging articles
betweenness_centrality = nx.betweenness_centrality(article_network)

# Sort articles by betweeness centrality
sorted_betweenness_centrality = sorted(
    betweenness_centrality.items(), key=lambda x: x[1], reverse=True)

# Select the top 20 most central nodes for clearer visualization
top_central_nodes = [node for node, _ in sorted_betweenness_centrality[:20]]
filtered_network = article_network.subgraph(top_central_nodes)

### Enhanced Visualization ###
plt.figure(figsize=(12, 12))

# Generate a spring layout for positioning nodes with more spacing
# Increased k for more spacing
pos = nx.spring_layout(filtered_network, k=0.25, iterations=100)

# Get node sizes based on betweenness centrality (scaled up for better visualization)
node_sizes = [betweenness_centrality[node] *
              3000 if node in betweenness_centrality else 100 for node in filtered_network.nodes()]

# Apply a color map to nodes based on centrality
cmap = cm.viridis
node_colors = [betweenness_centrality[node]
               if node in betweenness_centrality else 0 for node in filtered_network.nodes()]

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
important_nodes = {node: node for node, _ in sorted_betweenness_centrality[:20]}
nx.draw_networkx_labels(filtered_network, adjusted_pos, labels=important_nodes,
                        font_size=10, font_color='yellow', font_weight='bold',
                        # Add background color
                        bbox=dict(facecolor='black', edgecolor='none', alpha=0.7))

with open("betweenness_centrality.csv", "w", newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["Node", "Betweenness_Centrality_Score"])

    # Write the top 20 nodes and their degree and betweenness centrality scores
    for node, b_centrality_score in sorted_betweenness_centrality[:20]:
        writer.writerow([node, b_centrality_score])

# Add a color bar to show node centrality values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
    vmin=min(node_colors), vmax=max(node_colors)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label='Betweenness Centrality')

# Add a title to the graph
plt.title('Key Wikipedia Articles that Connect Topics Related to WW2 and the Cold War', size=15)
# Display the graph
plt.axis('off')  # Turn off the axis for better visualization
plt.savefig("graph.png")
