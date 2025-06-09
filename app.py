import streamlit as st
import snscrape.modules.twitter as sntwitter
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from collections import Counter
import re

st.set_page_config(page_title="Twitter Similarity Checker", layout="centered")

st.title("📌 Twitter Profile Similarity")

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_user_data(username, max_tweets=100):
    tweets = []
    mentions = []
    bio = ""

    scraper = sntwitter.TwitterUserScraper(username)
    user = scraper.entity
    bio = user.description if user else ""

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f"from:{username}").get_items()):
        if i >= max_tweets:
            break
        tweets.append(tweet.content)
        mentions += re.findall(r'@\w+', tweet.content)

    return bio, tweets, mentions

def embed_and_compare(text1, text2):
    embeddings = model.encode([text1, text2])
    cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return cosine_similarity

with st.form("input_form"):
    user_a = st.text_input("Username pertama (tanpa @)", "elonmusk")
    user_b = st.text_input("Username kedua (tanpa @)", "BillGates")
    submitted = st.form_submit_button("Cek Kesamaan")

if submitted:
    st.info(f"Mengambil data @{user_a} dan @{user_b}...")

    bio_a, tweets_a, mentions_a = get_user_data(user_a)
    bio_b, tweets_b, mentions_b = get_user_data(user_b)

    if not tweets_a or not tweets_b:
        st.warning("Salah satu atau kedua profil tidak ditemukan atau tidak memiliki tweet yang cukup.")
    else:
        # Similarity Bio
        bio_sim = embed_and_compare(bio_a, bio_b)

        # Similarity Tweets (gabungan tweet jadi satu string besar)
        tweet_sim = embed_and_compare(" ".join(tweets_a), " ".join(tweets_b))

        # Interaksi (mentions overlap)
        counter_a = Counter(mentions_a)
        counter_b = Counter(mentions_b)

        top_mentions_a = set([x[0] for x in counter_a.most_common(10)])
        top_mentions_b = set([x[0] for x in counter_b.most_common(10)])
        mention_overlap = len(top_mentions_a.intersection(top_mentions_b)) / max(len(top_mentions_a.union(top_mentions_b)), 1)

        # Menampilkan hasil
        st.subheader("🔗 Hasil Kesamaan")
        st.write(f"**Kesamaan Bio:** `{bio_sim:.2f}`")
        st.write(f"**Kesamaan Tweet:** `{tweet_sim:.2f}`")
        st.write(f"**Overlap Interaksi (mentions):** `{mention_overlap:.2f}`")

        st.subheader("🔎 Detail Profile")
        df_profiles = pd.DataFrame({
            "Profile": [user_a, user_b],
            "Bio": [bio_a, bio_b],
            "Jumlah Tweet": [len(tweets_a), len(tweets_b)],
            "Top Mentions": [", ".join(top_mentions_a), ", ".join(top_mentions_b)]
        })

        st.dataframe(df_profiles)

        st.success("Analisis selesai!")

