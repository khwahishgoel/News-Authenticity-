import requests
import streamlit as st

API_KEY = st.secrets["NEWS_API_KEY"]


BASE_URL = "https://newsapi.org/v2/top-headlines"

def fetch_real_news(query, country="us", page_size=5):
    params = {
        "q": query,
        "apiKey": API_KEY,
        "language": "en",
        "pageSize": page_size
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        return []

    articles = response.json().get("articles", [])

    texts = []
    for article in articles:
        content = f"{article['title']} {article.get('description', '')}"
        texts.append(content)

    return texts
