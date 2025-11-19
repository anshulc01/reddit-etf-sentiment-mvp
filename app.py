import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import praw
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px

# -----------------------------------------------
# Streamlit Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Reddit ETF Sentiment – MVP",
    layout="wide",
)

st.title("🧠 Reddit ETF & Industry Sentiment – MVP")
st.caption("Agentic AI sentiment monitor across key investing subreddits (last 24 hours)")


# -----------------------------------------------
# Config
# -----------------------------------------------
SUBREDDITS = [
    "PersonalFinanceCanada",
    "investing",
    "ETFs",
    "stocks",
    "canadianinvestor",
]

QUERY = "ETF OR ETFs OR stock OR stocks OR fund OR index"
TIME_FILTER = "day"
POST_LIMIT = 150


# -----------------------------------------------
# Reddit Authentication
# -----------------------------------------------
@st.cache_resource
reddit = praw.Reddit(
    client_id="o6Zs31tEtOrXylN63wLC2Q",         # Your 14-character client ID
    client_secret="WDTefleTeB78pbJv-JXo49YE04tr4w", # Your 27-character client secret
    user_agent="QueensCapstoneETF-sentiment/1.0 by u/CapstoneProject01"        # Your custom user agent string
)
    return reddit


# -----------------------------------------------
# Sentiment Engines
# -----------------------------------------------
@st.cache_resource
def get_vader():
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


@st.cache_resource
def get_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model


# -----------------------------------------------
# Fetch Reddit Data
# -----------------------------------------------
@st.cache_data(ttl=86400)  # 24 hours
def fetch_reddit():
    reddit = get_reddit_client()
    rows = []

    for sub in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(sub)
            for post in subreddit.search(
                QUERY,
                sort="new",
                time_filter=TIME_FILTER,
                limit=POST_LIMIT
            ):
                created_dt = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)

                rows.append({
                    "id": post.id,
                    "type": "post",
                    "subreddit": sub,
                    "created_dt": created_dt,
                    "title": post.title,
                    "body": post.selftext or "",
                    "score": post.score,
                    "url": f"https://www.reddit.com{post.permalink}"
                })

                post.comments.replace_more(limit=0)
                for c in post.comments[:25]:
                    rows.append({
                        "id": f"{post.id}_{c.id}",
                        "type": "comment",
                        "subreddit": sub,
                        "created_dt": datetime.fromtimestamp(c.created_utc, tz=timezone.utc),
                        "title": post.title,
                        "body": c.body,
                        "score": c.score,
                        "url": f"https://www.reddit.com{c.permalink}"
                    })

        except Exception as e:
            st.error(f"❌ Error fetching from r/{sub}: {e}")

    return pd.DataFrame(rows)


# -----------------------------------------------
# Apply VADER
# -----------------------------------------------
def apply_vader(df):
    sia = get_vader()
    texts = (df["title"] + " " + df["body"]).astype(str)
    scores = texts.apply(sia.polarity_scores)
    scores_df = pd.DataFrame(scores.tolist())
    return pd.concat([df.reset_index(drop=True), scores_df.add_prefix("vader_")], axis=1)


# -----------------------------------------------
# Apply FinBERT (sampled to stay fast)
# -----------------------------------------------
def apply_finbert(df, max_rows=350):
    tokenizer, model = get_finbert()
    model.eval()

    df = df.copy()
    texts = (df["title"] + " " + df["body"]).astype(str)

    if len(texts) > max_rows:
        idx = np.random.choice(len(texts), max_rows, replace=False)
    else:
        idx = np.arange(len(texts))

    labels_out = [None] * len(texts)
    scores_out = [None] * len(texts)

    with torch.no_grad():
        for i in idx:
            encoding = tokenizer(
                texts[i],
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )
            logits = model(**encoding).logits
            probs = torch.softmax(logits, dim=1)
            label_idx = torch.argmax(probs).item()
            score = probs[0][label_idx].item()

            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            label = label_map[label_idx]

            labels_out[i] = label
            scores_out[i] = (-score if label == "negative" else score if label == "positive" else 0.0)

    df["finbert_label"] = labels_out
    df["finbert_score"] = scores_out
    return df


# -----------------------------------------------
# Extract Tickers
# -----------------------------------------------
TICKER_BLACKLIST = {"ETF", "ETFs", "TSX", "CAD", "USD", "PE", "RRSP", "TFSA"}

def extract_tickers(text):
    import re
    if not isinstance(text, str):
        return []
    candidates = re.findall(r"(?:\$)?\\b[A-Z]{2,5}\\b", text.upper())
    return [c.replace("$", "") for c in candidates if c.replace("$", "") not in TICKER_BLACKLIST]


def explode_tickers(df):
    df = df.copy()
    df["tickers"] = (df["title"] + " " + df["body"]).apply(extract_tickers)
    return df.explode("tickers").dropna(subset=["tickers"])


# -----------------------------------------------
# Industry Tagging
# -----------------------------------------------
INDUSTRIES = {
    "Tech / AI": ["tech", "technology", "ai", "software", "chip", "semiconductor"],
    "Financials": ["bank", "financial", "insurance", "mortgage"],
    "Energy": ["oil", "gas", "pipeline", "energy"],
    "REITs": ["reit", "real estate", "property"],
    "Materials/Metals": ["gold", "silver", "mining", "metal"],
    "Crypto": ["bitcoin", "crypto", "ethereum", "btc", "eth"],
}

def explode_industries(df):
    df = df.copy()
    df["industries"] = df["body"].str.lower().apply(
        lambda t: [ind for ind, keywords in INDUSTRIES.items() if any(k in t for k in keywords)]
    )
    return df.explode("industries").dropna(subset=["industries"])


# -----------------------------------------------
# Load Data
# -----------------------------------------------
refresh = st.button("🔁 Force refresh data")
if refresh:
    fetch_reddit.clear()

with st.spinner("Fetching Reddit data..."):
    df_raw = fetch_reddit()

if df_raw.empty:
    st.warning("❌ No data returned from Reddit. Check authentication.")
    st.stop()

st.success(f"Loaded {len(df_raw)} posts & comments.")


# -----------------------------------------------
# Apply Sentiment
# -----------------------------------------------
with st.spinner("Applying VADER..."):
    df_vader = apply_vader(df_raw)

with st.spinner("Applying FinBERT..."):
    df_final = apply_finbert(df_vader)


# -----------------------------------------------
# Aggregations
# -----------------------------------------------
df_tickers = explode_tickers(df_final)
df_industry = explode_industries(df_final)

# Snapshot
st.subheader("📊 Snapshot Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("Posts + Comments", len(df_final))
c2.metric("Unique Tickers", df_tickers["tickers"].nunique())
c3.metric("Industries Tagged", df_industry["industries"].nunique())


# -----------------------------------------------
# Ticker Sentiment
# -----------------------------------------------
st.subheader("🏷️ Ticker Sentiment")

if df_tickers.empty:
    st.info("No tickers mentioned.")
else:
    agg = df_tickers.groupby("tickers").agg(
        mentions=("id", "count"),
        vader=("vader_compound", "mean"),
        finbert=("finbert_score", "mean"),
    ).reset_index()

    agg["overall"] = agg[["vader", "finbert"]].mean(axis=1)

    st.dataframe(agg.sort_values("mentions", ascending=False))

    fig = px.bar(
        agg.sort_values("overall"),
        x="overall",
        y="tickers",
        title="Overall Sentiment by Ticker",
        orientation="h"
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------
# Industry Sentiment
# -----------------------------------------------
st.subheader("🏭 Industry Sentiment")

if df_industry.empty:
    st.info("No industry keywords detected.")
else:
    agg2 = df_industry.groupby("industries").agg(
        mentions=("id", "count"),
        vader=("vader_compound", "mean"),
        finbert=("finbert_score", "mean"),
    ).reset_index()

    agg2["overall"] = agg2[["vader", "finbert"]].mean(axis=1)

    st.dataframe(agg2.sort_values("mentions", ascending=False))

    fig2 = px.bar(
        agg2.sort_values("overall"),
        x="overall",
        y="industries",
        title="Overall Sentiment by Industry",
        orientation="h"
    )
    st.plotly_chart(fig2, use_container_width=True)


# -----------------------------------------------
# Raw Data
# -----------------------------------------------
st.subheader("📝 Raw Posts & Comments")
st.dataframe(df_final, use_container_width=True)

