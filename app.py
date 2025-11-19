
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import praw
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.express as px

# ---------------------------
# CONFIG
# ---------------------------

SUBREDDITS = [
    "PersonalFinanceCanada",
    "investing",
    "ETFs",
    "stocks",
    "canadianinvestor",
]

QUERY = "ETF OR ETFs OR stock OR stocks OR fund OR index"
TIME_FILTER = "day"  # last 24h
POST_LIMIT_PER_SUB = 150  # reasonable for MVP


# ---------------------------
# CACHED RESOURCES
# ---------------------------

@st.cache_resource
def get_reddit_client():
    """Create a PRAW Reddit client from Streamlit secrets / env vars."""
    client_id = st.secrets.get("REDDIT_CLIENT_ID", os.getenv("REDDIT_CLIENT_ID"))
    client_secret = st.secrets.get("REDDIT_CLIENT_SECRET", os.getenv("REDDIT_CLIENT_SECRET"))
    user_agent = st.secrets.get(
        "REDDIT_USER_AGENT",
        os.getenv("REDDIT_USER_AGENT", "etf-sentiment-mvp/0.1 by your_username"),
    )

    if not client_id or not client_secret:
        st.error(
            "Reddit API credentials not found.\n\n"
            "Please set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT "
            "in Streamlit secrets or environment variables."
        )
        st.stop()

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )
    return reddit


@st.cache_resource
def get_vader():
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


@st.cache_resource
def get_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


# ---------------------------
# FETCH DATA FROM REDDIT
# ---------------------------

@st.cache_data(ttl=60 * 60 * 24)  # cache for 24 hours
def fetch_reddit_data():
    reddit = get_reddit_client()
    rows = []

    for sub in SUBREDDITS:
        subreddit = reddit.subreddit(sub)
        try:
            for submission in subreddit.search(
                QUERY,
                sort="new",
                time_filter=TIME_FILTER,
                limit=POST_LIMIT_PER_SUB,
            ):
                created_dt = datetime.fromtimestamp(
                    submission.created_utc, tz=timezone.utc
                )

                # Main post
                rows.append(
                    {
                        "id": submission.id,
                        "type": "post",
                        "subreddit": sub,
                        "created_utc": submission.created_utc,
                        "created_dt": created_dt,
                        "title": submission.title or "",
                        "body": submission.selftext or "",
                        "score": submission.score,
                        "num_comments": submission.num_comments,
                        "url": f"https://www.reddit.com{submission.permalink}",
                    }
                )

                # A few top-level comments
                submission.comments.replace_more(limit=0)
                for c in submission.comments[:30]:
                    if isinstance(c.body, str):
                        rows.append(
                            {
                                "id": f"{submission.id}_{c.id}",
                                "type": "comment",
                                "subreddit": sub,
                                "created_utc": c.created_utc,
                                "created_dt": datetime.fromtimestamp(
                                    c.created_utc, tz=timezone.utc
                                ),
                                "title": submission.title or "",
                                "body": c.body or "",
                                "score": c.score,
                                "num_comments": None,
                                "url": f"https://www.reddit.com{c.permalink}",
                            }
                        )
        except Exception as e:
            st.warning(f"Error fetching from r/{sub}: {e}")

    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "type",
                "subreddit",
                "created_utc",
                "created_dt",
                "title",
                "body",
                "score",
                "num_comments",
                "url",
            ]
        )

    df = pd.DataFrame(rows)
    return df


# ---------------------------
# SENTIMENT FUNCTIONS
# ---------------------------

def apply_vader(df: pd.DataFrame) -> pd.DataFrame:
    sia = get_vader()
    texts = (df["title"].fillna("") + " " + df["body"].fillna("")).astype(str)

    vader_scores = texts.apply(sia.polarity_scores).tolist()
    vader_df = pd.DataFrame(vader_scores)
    out = pd.concat([df.reset_index(drop=True), vader_df.add_prefix("vader_")], axis=1)
    return out


def finbert_sentiment(texts, batch_size=16):
    tokenizer, model = get_finbert()
    model.eval()

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            scores, labels = torch.max(probs, dim=-1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_scores.extend(scores.cpu().numpy().tolist())

    # Map labels to finance sentiment: 0 negative, 1 neutral, 2 positive
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    finbert_label = [label_map[int(l)] for l in all_labels]

    # numeric score: -1, 0, +1 scaled by confidence
    numeric_scores = []
    for lab, sc in zip(finbert_label, all_scores):
        if lab == "negative":
            numeric_scores.append(-float(sc))
        elif lab == "positive":
            numeric_scores.append(float(sc))
        else:
            numeric_scores.append(0.0)

    return finbert_label, numeric_scores


def apply_finbert(df: pd.DataFrame, max_rows: int = 400) -> pd.DataFrame:
    # To keep it fast, only run FinBERT on a subset of rows
    df = df.copy()
    texts = (df["title"].fillna("") + " " + df["body"].fillna("")).astype(str)

    if len(texts) > max_rows:
        idx = np.arange(len(texts))
        np.random.seed(42)
        sampled_idx = np.random.choice(idx, size=max_rows, replace=False)
        mask = np.zeros(len(texts), dtype=bool)
        mask[sampled_idx] = True
    else:
        mask = np.ones(len(texts), dtype=bool)

    finbert_labels = [None] * len(texts)
    finbert_scores = [None] * len(texts)

    if mask.any():
        subset_texts = texts[mask].tolist()
        labels, scores = finbert_sentiment(subset_texts)
        j = 0
        for i, use in enumerate(mask):
            if use:
                finbert_labels[i] = labels[j]
                finbert_scores[i] = scores[j]
                j += 1

    df["finbert_label"] = finbert_labels
    df["finbert_score"] = finbert_scores
    return df


# ---------------------------
# TICKER & INDUSTRY EXTRACTION
# ---------------------------

TICKER_BLACKLIST = {
    "ETF",
    "ETFs",
    "TSX",
    "CAD",
    "USD",
    "RRSP",
    "TFSA",
    "USA",
    "UK",
    "IMO",
    "DIY",
    "LOL",
    "NAV",
    "EPS",
    "PE",
    "TSLAQ",
}

INDUSTRY_KEYWORDS = {
    "Tech / AI": ["tech", "technology", "ai", "software", "semiconductor", "chip"],
    "Financials / Banks": ["bank", "banks", "financial", "lender", "insurance"],
    "Energy": ["oil", "gas", "energy", "pipeline"],
    "Real Estate / REITs": ["reit", "real estate", "property", "mortgage"],
    "Utilities": ["utility", "utilities", "hydro"],
    "Gold / Metals": ["gold", "silver", "metal", "mining", "precious"],
    "Crypto / Blockchain": ["crypto", "bitcoin", "btc", "eth", "ethereum", "blockchain"],
}


def extract_tickers(text: str):
    import re

    if not isinstance(text, str):
        return []
    # Very simple heuristic: 2–5 uppercase letters, optionally prefixed with $
    pattern = r"(?:\$)?\b[A-Z]{2,5}\b"
    candidates = re.findall(pattern, text.upper())
    cleaned = []
    for c in candidates:
        c = c.replace("$", "")
        if c in TICKER_BLACKLIST:
            continue
        cleaned.append(c)
    return list(set(cleaned))


def explode_tickers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combined_text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).astype(
        str
    )
    df["tickers"] = df["combined_text"].apply(extract_tickers)
    exploded = df.explode("tickers").dropna(subset=["tickers"])
    return exploded


def tag_industries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combined_text"] = (
        df["title"].fillna("") + " " + df["body"].fillna("")
    ).astype(str).str.lower()

    def find_industries(text):
        hits = []
        for label, keys in INDUSTRY_KEYWORDS.items():
            for k in keys:
                if k in text:
                    hits.append(label)
                    break
        return hits

    df["industries"] = df["combined_text"].apply(find_industries)
    exploded = df.explode("industries").dropna(subset=["industries"])
    return exploded


# ---------------------------
# STREAMLIT APP
# ---------------------------

st.set_page_config(
    page_title="Reddit ETF Sentiment – MVP",
    layout="wide",
)

st.title("🧠 Reddit ETF & Industry Sentiment – MVP")
st.caption(
    "Agentic AI-style sentiment monitor across key Reddit investing communities (last 24 hours)."
)

with st.expander("ℹ️ About this MVP", expanded=False):
    st.markdown(
        """
        This prototype:
        - Scrapes a curated set of investing subreddits via the **Reddit API**
        - Extracts **ETF tickers**, **stock tickers**, and **industry themes**
        - Applies **two sentiment engines**:
          - **VADER** (rule-based, fast)
          - **FinBERT** (transformer model trained on financial text, sampled subset)
        - Aggregates sentiment at the **ticker** and **industry** level.

        **Subreddits:**
        - r/PersonalFinanceCanada
        - r/investing
        - r/ETFs
        - r/stocks
        - r/canadianinvestor
        """
    )

# Controls
col_left, col_right = st.columns([1, 1])
with col_left:
    refresh = st.button("🔁 Force refresh data (ignore 24h cache)")

if refresh:
    fetch_reddit_data.clear()

with st.spinner("Fetching latest Reddit data (last 24h)..."):
    df_raw = fetch_reddit_data()

if df_raw.empty:
    st.warning(
        "No data returned from Reddit. Try again later or adjust the query in the code."
    )
    st.stop()

st.success(f"Loaded {len(df_raw)} posts & comments from Reddit.")

# Sentiment
with st.spinner("Running VADER sentiment..."):
    df_vader = apply_vader(df_raw)

with st.spinner("Running FinBERT sentiment on a sampled subset..."):
    df_sent = apply_finbert(df_vader)

# ---------------------------
# AGGREGATIONS
# ---------------------------

df_sent["date"] = df_sent["created_dt"].dt.tz_convert("UTC").dt.date

# Tickers
df_tickers = explode_tickers(df_sent)
if not df_tickers.empty:
    df_tickers["vader_compound"] = df_tickers["vader_compound"].astype(float)
    df_tickers["finbert_score"] = pd.to_numeric(
        df_tickers["finbert_score"], errors="coerce"
    )

# Industries
df_industries = tag_industries(df_sent)
if not df_industries.empty:
    df_industries["vader_compound"] = df_industries["vader_compound"].astype(float)
    df_industries["finbert_score"] = pd.to_numeric(
        df_industries["finbert_score"], errors="coerce"
    )

# Summary metrics
total_items = len(df_sent)
unique_tickers = df_tickers["tickers"].nunique() if not df_tickers.empty else 0
unique_subs = df_sent["subreddit"].nunique()
time_span = (
    f"{df_sent['created_dt'].min().date()} → {df_sent['created_dt'].max().date()}"
)

st.subheader("📊 Snapshot")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total posts & comments", f"{total_items}")
m2.metric("Unique tickers mentioned", f"{unique_tickers}")
m3.metric("Subreddits covered", f"{unique_subs}")
m4.metric("Time span (UTC)", time_span)

# ---------------------------
# TICKER SENTIMENT
# ---------------------------

st.subheader("🏷️ Ticker Sentiment (VADER & FinBERT)")

if df_tickers.empty:
    st.info(
        "No tickers detected in the last 24 hours. Try again later or broaden the query."
    )
else:
    agg_ticker = (
        df_tickers.groupby("tickers")
        .agg(
            mentions=("id", "count"),
            avg_vader=("vader_compound", "mean"),
            avg_finbert=("finbert_score", "mean"),
        )
        .reset_index()
    )

    agg_ticker["overall_sentiment"] = (
        agg_ticker["avg_vader"].fillna(0) * 0.5
        + agg_ticker["avg_finbert"].fillna(0) * 0.5
    )

    top_n = st.slider(
        "Show top N tickers by mentions", min_value=5, max_value=50, value=15, step=5
    )
    top_tickers = agg_ticker.sort_values("mentions", ascending=False).head(top_n)

    st.dataframe(
        top_tickers.style.format(
            {
                "mentions": "{:,.0f}",
                "avg_vader": "{:.3f}",
                "avg_finbert": "{:.3f}",
                "overall_sentiment": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    fig_ticker = px.bar(
        top_tickers.sort_values("overall_sentiment", ascending=True),
        x="overall_sentiment",
        y="tickers",
        orientation="h",
        hover_data=["mentions", "avg_vader", "avg_finbert"],
        title="Overall sentiment score by ticker (higher = more positive)",
    )
    st.plotly_chart(fig_ticker, use_container_width=True)

# ---------------------------
# INDUSTRY SENTIMENT
# ---------------------------

st.subheader("🏭 Industry / Theme Sentiment")

if df_industries.empty:
    st.info(
        "No industry/theme keywords detected yet. This will populate as more posts reference sectors (tech, banks, REITs, etc.)."
    )
else:
    agg_industry = (
        df_industries.groupby("industries")
        .agg(
            mentions=("id", "count"),
            avg_vader=("vader_compound", "mean"),
            avg_finbert=("finbert_score", "mean"),
        )
        .reset_index()
    )
    agg_industry["overall_sentiment"] = (
        agg_industry["avg_vader"].fillna(0) * 0.5
        + agg_industry["avg_finbert"].fillna(0) * 0.5
    )

    st.dataframe(
        agg_industry.style.format(
            {
                "mentions": "{:,.0f}",
                "avg_vader": "{:.3f}",
                "avg_finbert": "{:.3f}",
                "overall_sentiment": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    fig_ind = px.bar(
        agg_industry.sort_values("overall_sentiment", ascending=True),
        x="overall_sentiment",
        y="industries",
        orientation="h",
        hover_data=["mentions", "avg_vader", "avg_finbert"],
        title="Overall sentiment score by industry/theme",
    )
    st.plotly_chart(fig_ind, use_container_width=True)

# ---------------------------
# RAW DATA VIEW
# ---------------------------

st.subheader("🧾 Underlying Reddit Posts & Comments")
with st.expander("Show raw data (for debugging / validation)", expanded=False):
    st.dataframe(
        df_sent[
            [
                "type",
                "subreddit",
                "created_dt",
                "title",
                "body",
                "score",
                "vader_compound",
                "finbert_label",
                "finbert_score",
                "url",
            ]
        ].sort_values("created_dt", ascending=False),
        use_container_width=True,
    )

st.caption(
    "Prototype for research / demonstration purposes only – not investment advice."
)
