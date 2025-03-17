import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np

# Ensure NLTK stopwords are downloaded
nltk.download("stopwords")

# Load your full dataset (replace with actual file path if needed)
file_path = "data.jsonl"
df = pd.read_json(file_path, lines=True)

# Expand the nested "data" column
df_expanded = pd.json_normalize(df["data"])

# Keep only relevant columns (ensure these exist in your dataset)
df_expanded = df_expanded[["title", "selftext", "created_utc"]]

# Convert timestamps to datetime format
df_expanded["created_utc"] = pd.to_datetime(df_expanded["created_utc"], unit="s")

# Define text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    stop_words = set(stopwords.words("english"))  # Load stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply text preprocessing
df_expanded["clean_text"] = df_expanded["title"].fillna("") + " " + df_expanded["selftext"].fillna("")
df_expanded["clean_text"] = df_expanded["clean_text"].apply(clean_text)

# Convert text into numerical form using CountVectorizer
vectorizer = CountVectorizer(max_features=1000, stop_words="english")
text_matrix = vectorizer.fit_transform(df_expanded["clean_text"])

# Apply LDA Topic Modeling
num_topics = 5  # Change this based on how many topics you want
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(text_matrix)

# Get top words for each topic
words = vectorizer.get_feature_names_out()
topics = {}
for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [words[i] for i in topic.argsort()[-10:]]  # Top 10 words per topic
    topics[f"Topic {topic_idx+1}"] = top_words

# Print topics
import pprint
pprint.pprint(topics)

# Assign each post a topic distribution
topic_distributions = lda_model.transform(text_matrix)
df_expanded["topic"] = np.argmax(topic_distributions, axis=1)

# Group by date to see topic trends over time
df_expanded["date"] = df_expanded["created_utc"].dt.date
topic_trends = df_expanded.groupby(["date", "topic"]).size().unstack().fillna(0)

# Plot stacked area chart for topic trends
plt.figure(figsize=(12, 6))
plt.stackplot(topic_trends.index, topic_trends.T, labels=[f"Topic {i+1}" for i in range(num_topics)], alpha=0.6)
plt.xlabel("Date")
plt.ylabel("Proportion of Discussions")
plt.title("Trending Discussion Topics Over Time")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
