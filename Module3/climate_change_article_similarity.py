import csv
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Wikipedia API base URL
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"


def is_valid_article(link_title):
    """
    Check if a link is valid (not a disambiguation, list, or non-article link).
    """
    invalid_keywords = [
        'disambiguation', 'List of', 'User:', 'Talk:', 'Special:', 'File:',
        'Template:', 'Wikipedia:', 'Help:', 'Portal:', 'Category:', 'ISBN',
        'Doi', 'LCCN', 'ISSN', 'Glossary',
    ]
    return not any(keyword in link_title for keyword in invalid_keywords)


def search_articles(query, max_results=20):
    """
    Search Wikipedia for articles related to a specific query and return the top article titles.
    Filters out invalid articles using the is_valid_article function.
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

    # Filter articles using the is_valid_article function
    articles = [
        result['title'] for result in data.get('query', {}).get('search', [])
        if is_valid_article(result['title'])
    ]

    return articles


def get_article_content(article_title):
    """
    Fetch the introductory extract of a Wikipedia article.
    """
    params = {
        "action": "query",
        "titles": article_title,
        "prop": "extracts",
        "exintro": True,
        "format": "json"
    }

    response = requests.get(WIKI_API_URL, params=params)
    data = response.json()

    pages = data.get('query', {}).get('pages', {})
    page_id = list(pages.keys())[0]
    article_content = pages[page_id].get('extract', '')
    return article_content


def calculate_similarity(articles_content):
    """
    Calculate cosine similarity between articles.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(articles_content)
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix


def find_similar_pages(query_articles, all_articles, num_similar_pages=10):
    """
    Find the most similar pages for each query article.
    """
    all_articles_content = [get_article_content(
        article) for article in all_articles]

    # Compute similarity matrix
    similarity_matrix = calculate_similarity(all_articles_content)
    
    # Find the most similar pages for each query article
    similar_articles = {}
    for i, query_article in enumerate(query_articles):
        if query_article in all_articles:
            idx = all_articles.index(query_article)
            similarity_scores = similarity_matrix[idx]

            # Get the top N most similar articles (excluding itself)
            similar_indices = np.argsort(similarity_scores)[
                ::-1][1:num_similar_pages + 1]
            similar_articles[query_article] = [
                (all_articles[idx], similarity_scores[idx]) for idx in similar_indices
            ]


    return similar_articles


def write_results_to_csv(similar_articles, filename="similar_articles.csv"):
    """
    Write the results of similar articles and their cosine similarity scores to a CSV file.
    """
    with open(filename, "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Query Article", "Similar Article", "Cosine Similarity"])

        for query_article, similar_list in similar_articles.items():
            for similar_article, similarity_score in similar_list:
                writer.writerow(
                    [query_article, similar_article, similarity_score])


# Search for the top 20 articles related to climate change
all_articles = search_articles("Climate change", max_results=20)
# Use the first 3 as the main query articles
query_articles = all_articles[:3]

# Find 10 most similar pages for each query article
similar_articles = find_similar_pages(
    query_articles, all_articles, num_similar_pages=10)

# Write results to a CSV file
write_results_to_csv(similar_articles)
