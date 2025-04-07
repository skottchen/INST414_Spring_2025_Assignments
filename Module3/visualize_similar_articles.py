import csv
import matplotlib.pyplot as plt
from collections import Counter


def read_similar_articles_from_csv(csv_filename):
    """
    Read the CSV file and extract similar articles.
    """
    similar_articles = []

    with open(csv_filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            similar_articles.append(row[1])  # Collect similar article names

    return similar_articles


def plot_similar_articles_frequency(csv_filename):
    """
    Create a bar chart that shows how often each similar article appears in the CSV.
    The bars are sorted from greatest to least.
    """
    # Read the similar articles from the CSV file
    similar_articles = read_similar_articles_from_csv(csv_filename)

    # Count the frequency of each similar article
    article_counts = Counter(similar_articles)

    # Sort the articles by frequency in descending order
    sorted_articles = sorted(article_counts.items(),
                             key=lambda x: x[1], reverse=True)
    articles, counts = zip(*sorted_articles)

    # Plot the bar chart
    plt.figure(figsize=(14, 8))  # Increase figure size for better readability
    plt.barh(articles, counts, color='lightgreen')

    # Invert y-axis to have the highest count at the top
    plt.gca().invert_yaxis()

    # Add labels and title
    plt.xlabel('Frequency Count', fontsize=12) # number of times that each article shows up across all query articles
    plt.ylabel('Similar Articles', fontsize=12)
    plt.title('Frequency of Similar Articles', fontsize=14)

    # Adjust layout to ensure the y-axis labels are not cut off
    # Shift the graph to the right by adjusting the left margin
    plt.subplots_adjust(left=0.33)

    plt.savefig("article_freq.png")


# Specify your CSV filename
csv_filename = "similar_articles.csv"

# Generate the bar chart
plot_similar_articles_frequency(csv_filename)
