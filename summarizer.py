# pip install requests on your bash
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

def get_news_article(url):
    # Retrieve the news article content from the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')

    # Extract text from paragraphs
    article_text = ' '.join([p.get_text() for p in paragraphs])
    return article_text

def summarize_news_article(article_text):
    # Load pre-trained BERT model for summarization
    summarizer = pipeline('summarization')

    # Generate summary
    summary = summarizer(article_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)[0]['summary_text']
    
    return summary

if __name__ == "__main__":
    # Example URL of a news article
    news_url = "https://www.example.com/sample-news-article"

    # Get the news article content
    article_text = get_news_article(news_url)

    # Summarize the news article
    summarized_text = summarize_news_article(article_text)

    # Display the original and summarized text
    print("Original Article:")
    print(article_text)
    
    print("\nSummarized Article:")
    print(summarized_text)
