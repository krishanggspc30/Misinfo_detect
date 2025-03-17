import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re
import requests
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Ensure stopwords are downloaded
nltk.download("stopwords")

# Load the dataset
file_path = "data.jsonl"
df = pd.read_json(file_path, lines=True)

# Expand the nested "data" column
df_expanded = pd.json_normalize(df["data"])

# Keep only relevant columns
df_expanded = df_expanded[["title", "selftext", "subreddit", "created_utc"]]

# Convert timestamps to datetime
df_expanded["created_utc"] = pd.to_datetime(df_expanded["created_utc"], unit="s")

# Define text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Apply text preprocessing
df_expanded["clean_text"] = df_expanded["title"].fillna("") + " " + df_expanded["selftext"].fillna("")
df_expanded["clean_text"] = df_expanded["clean_text"].apply(clean_text)

# Convert text into numerical form
vectorizer = CountVectorizer(max_features=1000, stop_words="english")
text_matrix = vectorizer.fit_transform(df_expanded["clean_text"])

# Apply LDA for topic modeling
num_topics = 5
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(text_matrix)

# Extract top words per topic
words = vectorizer.get_feature_names_out()
topics = {}
for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [words[i] for i in topic.argsort()[-10:]]
    topics[f"Topic {topic_idx+1}"] = top_words

# Assign topics to posts
topic_distributions = lda_model.transform(text_matrix)
df_expanded["topic"] = np.argmax(topic_distributions, axis=1)

# Group by date for topic trends
df_expanded["date"] = df_expanded["created_utc"].dt.date
topic_trends = df_expanded.groupby(["date", "topic"]).size().unstack().fillna(0)

# Streamlit Dashboard
st.title("🔍 Social Media Analysis Dashboard")

# 📈 Time Series: Posts Matching a Search Query
st.subheader("📈 Time Series of Posts (Matching a Search Query)")
search_query = st.text_input("Enter a keyword to filter posts:", "")
if search_query:
    filtered_df = df_expanded[df_expanded["clean_text"].str.contains(search_query, case=False, na=False)]
    posts_over_time = filtered_df.groupby("date").size()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(posts_over_time.index, posts_over_time.values, marker="o", linestyle="-", color="b")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Posts")
    ax.set_title(f"Number of Posts Containing '{search_query}' Over Time")
    ax.grid(True)
    st.pyplot(fig)

# 📊 Time Series of Key Topics (Trending Themes)
st.subheader("📊 Trending Topics Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
ax.stackplot(topic_trends.index, topic_trends.T, labels=[f"Topic {i+1}" for i in range(num_topics)], alpha=0.6)
ax.set_xlabel("Date")
ax.set_ylabel("Proportion of Discussions")
ax.set_title("Trending Discussion Topics Over Time")
ax.legend(loc="upper left")
ax.grid(True)
st.pyplot(fig)

# 🥧 Pie Chart of Communities (Top Subreddits)
st.subheader("🥧 Top Contributing Subreddits")
subreddit_counts = df_expanded["subreddit"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(subreddit_counts, labels=subreddit_counts.index, autopct="%1.1f%%", startangle=140)
ax.set_title("Top 10 Subreddits by Post Count")
st.pyplot(fig)

# 🔗 Network Visualization of Shared Keywords
st.subheader("🔗 Network of Shared Keywords")
top_words = vectorizer.get_feature_names_out()[:30]  # Top 30 words
word_pairs = [(top_words[i], top_words[j]) for i in range(len(top_words)) for j in range(i+1, len(top_words)) if i != j]

# Build network graph
G = nx.Graph()
G.add_edges_from(word_pairs[:50])  # Use only top 50 connections to avoid clutter

fig, ax = plt.subplots(figsize=(10, 7))
nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", font_size=10, ax=ax)
st.pyplot(fig)

# ✅ Display Topics
st.subheader("🔍 Identified Topics")
for topic, words in topics.items():
    st.write(f"**{topic}:** {', '.join(words)}")

# 🔥 Misinformation Detection Using Your LLM Model
st.subheader("🚨 Misinformation Detection")

user_input = st.text_area("Enter a Reddit post to check for misinformation:")
if st.button("Analyze"):
    if user_input.strip():
        # Send request to the LLM API
        try:
            response = requests.post("http://127.0.0.1:8000/predict/", json={"text": user_input})
            result = response.json()
            st.write(f"**Prediction:** {result['prediction']}")
        except:
            st.error("Error: Unable to connect to the LLM API. Make sure it's running.")
    else:
        st.warning("Please enter some text.")
