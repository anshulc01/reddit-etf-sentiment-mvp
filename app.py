import time
from datetime import datetime, timezone
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================================================
# CONFIG
# =========================================================

SUBREDDITS = [
    "PersonalFinanceCanada",
    "investing",
    "ETFs",
    "stocks",
    "canadianinvestor",
]

QUERY = "ETF OR ETFs OR stock OR stocks OR fund OR index"
POST_LIMIT = 50

HEADERS = {
    "User-Agent": "Mozilla/5.0 (CapstoneETFSentiment/1.0)"
}

st.set_page_config(page_title="Reddit ETF Sentiment – MVP", layout="wide")
st.title("🧠 Reddit ETF & Industry Sentiment – MVP")
st.caption("Powered by Reddit Mobile API (no OAuth, no secrets, no 403).")


# =========================================================
# SENTIMENT MODELS
# =========================================================

@st.cache_resource
def get_vader():
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


@st.cache_resource
def get_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def apply_vader(df):
    sia = get_vader()
    texts = (df["title"].fillna("") + " " + df["body"].fillna(""))
    scores = texts.apply(sia.polarity_scores).tolist()
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(scores)], axis=1)


def finbert_batch(texts):
    tokenizer, model = get_finbert()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(texts), 16):
            chunk = texts[i:i+16]
            enc = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=1)
            conf, lab = torch.max(probs, dim=1)

            for l, c in zip(lab, conf):
                l = int(l)
                c = float(c)
                all_labels.append(label_map[l])
                all_scores.append(c if l == 2 else -c if l == 0 else 0)

    return all_labels, all_scores


def apply_finbert(df, max_rows=300):
    df = df.copy()
    texts = (df["title"].fillna("") + " " + df["body"].fillna(""))

    if len(texts) > max_rows:
        texts = texts.sample(max_rows, random_state=42)

    labels, scores = finbert_batch(texts.tolist())

    df["finbert_label"] = labels + [None] * (len(df) - len(labels))
    df["finbert_score"] = scores + [None] * (len(df) - len(scores))
    return df


# =========================================================
# REDDIT SCRAPER (MOBILE API)
# =========================================================

def fetch_mobile_api(subreddit):
    url = f"https://api.reddit.com/r/{subreddit}/search"
    params = {"q": QUERY, "sort": "new", "restrict_sr": "1", "limit": POST_LIMIT}

    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        posts = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            posts.append({
                "id": d.get("id"),
                "subreddit": subreddit,
                "title": d.get("title", ""),
                "body": d.get("selftext", ""),
                "score": d.get("score", 0),
                "num_comments": d.get("num_comments", 0),
                "created_utc": d.get("created_utc"),
                "url": "https://reddit.com" + d.get("permalink", "")
            })

        return posts

    except Exception as e:
        return []


@st.cache_data(ttl=3600)
def fetch_all():
    all_posts = []
    for sub in SUBREDDITS:
        posts = fetch_mobile_api(sub)
        all_posts.extend(posts)
        time.sleep(1)
    return pd.DataFrame(all_posts)


# =========================================================
# TICKERS & INDUSTRIES
# =========================================================

EXCLUDE = {"ETF", "ETFS", "NAV", "EPS"}

INDUSTRIES = {
    "Tech": ["tech", "ai", "chip", "semiconductor"],
    "Banks": ["bank", "financial", "credit"],
    "Energy": ["oil", "gas", "energy"],
    "Mining / Metals": ["gold", "silver", "mining"],
    "Crypto": ["crypto", "bitcoin", "eth", "blockchain"],
}

def extract_tickers(text):
    if not isinstance(text, str):
        return []
    raw = re.findall(r"\$?[A-Z]{2,5}", text)
    return [t.replace("$", "") for t in raw if t.replace("$", "") not in EXCLUDE]


def tag_industries(text):
    text = text.lower()
    hits = []
    for name, keys in INDUSTRIES.items():
        if any(k in text for k in keys):
            hits.append(name)
    return hits


# =========================================================
# UI
# =========================================================

if st.button("🔁 Force refresh"):
    fetch_all.clear()

df = fetch_all()

if df.empty:
    st.error("Reddit mobile API returned no posts. Try again later.")
    st.stop()

# Date conversion
df["created_dt"] = df["created_utc"].apply(
    lambda x: datetime.fromtimestamp(x, tz=timezone.utc) if x else None
)

# Sentiment
with st.spinner("Running VADER…"):
    df = apply_vader(df)

with st.spinner("Running FinBERT…"):
    df = apply_finbert(df)

# Tickers
df["combined"] = df["title"] + " " + df["body"]
df["tickers"] = df["combined"].apply(extract_tickers)
df_t = df.explode("tickers").dropna(subset=["tickers"])

# Industries
df["industries"] = df["combined"].apply(tag_industries)
df_i = df.explode("industries").dropna(subset=["industries"])

# =========================================================
# DASHBOARD
# =========================================================

st.subheader("📊 Snapshot")
c1, c2, c3 = st.columns(3)
c1.metric("Total Posts", len(df))
c2.metric("Tickers Found", df_t["tickers"].nunique())
c3.metric("Industries Found", df_i["industries"].nunique())

# Ticker sentiment
st.subheader("🏷️ Ticker Sentiment")

if df_t.empty:
    st.info("No tickers found.")
else:
    agg = df_t.groupby("tickers").agg(
        mentions=("id", "count"),
        vader=("compound", "mean"),
        finbert=("finbert_score", "mean")
    ).reset_index()

    agg["sentiment"] = (agg["vader"] + agg["finbert"]) / 2

    st.dataframe(agg.sort_values("mentions", ascending=False))

    fig = px.bar(
        agg.sort_values("sentiment"),
        x="sentiment",
        y="tickers",
        orientation="h"
    )
    st.plotly_chart(fig, use_container_width=True)

# Industry sentiment
st.subheader("🏭 Industry Sentiment")

if df_i.empty:
    st.info("No industries detected.")
else:
    a2 = df_i.groupby("industries").agg(
        mentions=("id", "count"),
        vader=("compound", "mean"),
        finbert=("finbert_score", "mean")
    ).reset_index()

    a2["sentiment"] = (a2["vader"] + a2["finbert"]) / 2

    st.dataframe(a2)

    fig2 = px.bar(a2, x="industries", y="sentiment")
    st.plotly_chart(fig2, use_container_width=True)

# Raw data
st.subheader("🧾 Raw Posts")
st.dataframe(df)