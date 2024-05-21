# -*- coding: utf-8 -*-
"""
Script Name: stock_price_prediction_news_sentiment.py
Description: This script predicts S&P 500 prices using sentiment analysis of news headlines and Support Vector Regression (SVR).
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from monkeylearn import MonkeyLearn
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

ml = MonkeyLearn('YOUR_API_KEY')

def get_article_titles(search_term):
    url = f"https://news.google.com/search?q={search_term}&hl=en-US&gl=US&ceid=US%3Aen"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    articles = soup.find_all("a", {"class": "DY5T1d"})
    titles = [article.text for article in articles]
    return titles[:25]

search_terms = ["inflation", "interest rates", "economic indicators", "geopolitical events", "regulation", "stock market", "commodities", "technology"]
df = pd.DataFrame(columns=search_terms)

for search_term in search_terms:
    titles = get_article_titles(search_term)
    df[search_term] = titles

chunks = []
delimiter = ". "
for search_term in search_terms:
    titles = df[search_term].tolist()
    text = delimiter.join(titles)
    for i in range(0, len(text), 4096):
        chunk = text[i:i+4096]
        chunks.append(chunk)

summaries = []
for chunk in chunks:
    response = ml.classifiers.classify('cl_pi3C7JiL', [chunk])
    summaries.append(response.body[0]['classifications'][0]['tag_name'])

final_summary = "\n".join(summaries)

# Get SP500 data
spx = yf.download('^GSPC', interval='1m', period='5d')['Adj Close']
spx.name = 'spx'
spx = pd.DataFrame(data = spx)
df = df.join(spx)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
df.dropna(inplace=True)

# Encode Sentiment
df['Sentiment'] = LabelEncoder().fit_transform(df['Sentiment'])

# Correlation heatmap
corr_df = df.corr(method='pearson')
sns.heatmap(corr_df, cmap='RdYlGn', vmax=1.0, vmin=-1.0, linewidths=2.1)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

# Prepare features and labels
x = df.drop({'spx', 'Text'}, axis=1)
y = df['spx'].values.reshape(-1, 1)

# Scale features and labels
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Train SVR model
regressor = SVR(kernel='rbf')
regressor.fit(x, y.ravel())

# Plot predictions
plt.plot(sc_y.inverse_transform(y), color='red')
plt.plot(sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)), color='blue')
plt.title('SP500 Predicted (Blue) vs Actual (Red)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
