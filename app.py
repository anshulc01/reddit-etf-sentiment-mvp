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

# Simple investing-related query – used with Reddit's public search.json
QUERY = "ETF OR ETFs OR stock OR stocks OR fund OR index"
POST_LIMIT_PER_SUB = 80          # keep modest for speed
TIME_FILTER = "day"              # last 24h window (approx via sort=new)

# Public JSON endpoints do NOT need auth, but they require a User-Agent
HEADERS = {
    "User-Agent": "CapstoneETFSentiment/1.0 (contact: u/CapstoneProject01)"
}

# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Reddit ETF & Industry Sentiment – MVP (No OAuth)",
    layout="wide",
)

st.title("🧠 Reddit ETF & Industry Sentiment – MVP")
st.caption(
    "Agentic-style sentiment monitor across key investing subreddits "
    "(using Reddit's public JSON endpoints; no OAuth required)."
)

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown(
        """
        This MVP:
        - Queries Reddit's public JSON API for a set of investing subreddits
        - Filters posts with a simple query: `ETF OR ETFs OR stock OR stocks OR fund OR index`
        - Extracts tickers and rough industry keywords from post titles + bodies
        - Applies **two sentiment engines**:
            - **VADER** – rule-based, lexicon sentiment
            - **FinBERT** – transformer model fine-tuned on financial text
        - Aggregates sentiment by **ticker** and **industry/theme**
        
        Because we use Reddit's public endpoints, we **do not need API keys or OAuth**, which avoids
        rate-limit / 401 issues on Streamlit Cloud.
        """
    )

# =========================================================
# UTIL: Sentiment resources
# =========================================================

@st.cache_resource
def get_vader():
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


@st.cache_resource
def get_finbert():
    """
    FinBERT model + tokenizer.
    NOTE: this is ~400MB; first run will be slower while it downloads.
    """
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


# =========================================================
# FETCH REDDIT DATA – PUBLIC JSON (NO AUTH)
# =========================================================

def _fetch_subreddit_json(subreddit: str, query: str, limit: int) -> list[dict]:
    """
    Fetch posts from a subreddit using Reddit's public JSON API.
    Uses search.json; no OAuth required.
    """
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": query,
        "restrict_sr": 1,
        "sort": "new",
        "t": TIME_FILTER,
        "limit": limit,
    }

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if resp.status_code != 200:
            st.error(
                f"❌ Error fetching from r/{subreddit}: HTTP {resp.status_code} – "
                f"{resp.text[:200]}"
            )
            return []

        data = resp.json()
        children = data.get("data", {}).get("children", [])
        posts = []

        for child in children:
            d = child.get("data", {})
            created_utc = d.get("created_utc")
            created_dt = (
                datetime.fromtimestamp(created_utc, tz=timezone.utc)
                if created_utc
                else None
            )

            posts.append(
                {
                    "id": d.get("id"),
                    "subreddit": subreddit,
                    "created_utc": created_utc,
                    "created_dt": created_dt,
                    "title": d.get("title") or "",
                    "body": d.get("selftext") or "",
                    "score": d.get("score"),
                    "num_comments": d.get("num_comments"),
                    "url": f"https://www.reddit.com{d.get('permalink', '')}",
                }
            )

        return posts

    except Exception as e:
        st.error(f"❌ Exception fetching r/{subreddit}: {e}")
        return []


@st.cache_data(ttl=60 * 60 * 24)  # cache for 24h
def fetch_reddit_data() -> pd.DataFrame:
    """
    Fetch posts for all configured subreddits via public JSON.
    """
    all_rows: list[dict] = []

    for sub in SUBREDDITS:
        posts = _fetch_subreddit_json(sub, QUERY, POST_LIMIT_PER_SUB)
        all_rows.extend(posts)
        # be polite to Reddit
        time.sleep(1.0)

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "id",
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

    return pd.DataFrame(all_rows)


# =========================================================
# SENTIMENT
# =========================================================

def apply_vader(df: pd.DataFrame) -> pd.DataFrame:
    sia = get_vader()
    texts = (df["title"].fillna("") + " " + df["body"].fillna("")).astype(str)
    scores = texts.apply(sia.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores).add_prefix("vader_")
    return pd.concat([df.reset_index(drop=True), scores_df], axis=1)


def finbert_sentiment(texts: list[str], batch_size: int = 16):
    tokenizer, model = get_finbert()

    all_labels = []
    all_scores = []
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)
            scores, labels = torch.max(probs, dim=-1)
            labels = labels.cpu().numpy().tolist()
            scores = scores.cpu().numpy().tolist()

            for lab, sc in zip(labels, scores):
                all_labels.append(label_map[int(lab)])
                all_scores.append(float(sc))

    numeric = []
    for label, conf in zip(all_labels, all_scores):
        if label == "negative":
            numeric.append(-conf)
        elif label == "positive":
            numeric.append(conf)
        else:
            numeric.append(0.0)

    return all_labels, numeric


def apply_finbert(df: pd.DataFrame, max_rows: int = 400) -> pd.DataFrame:
    df = df.copy()
    texts = (df["title"].fillna("") + " " + df["body"].fillna("")).astype(str)

    if len(texts) == 0:
        df["finbert_label"] = []
        df["finbert_score"] = []
        return df

    # sample for speed if too large
    if len(texts) > max_rows:
        idx = np.arange(len(texts))
        np.random.seed(42)
        chosen = np.random.choice(idx, size=max_rows, replace=False)
        mask = np.zeros(len(texts), dtype=bool)
        mask[chosen] = True
    else:
        mask = np.ones(len(texts), dtype=bool)

    labels_all = [None] * len(texts)
    scores_all = [None] * len(texts)

    subset = texts[mask].tolist()
    labels, scores = finbert_sentiment(subset)
    j = 0
    for i, use in enumerate(mask):
        if use:
            labels_all[i] = labels[j]
            scores_all[i] = scores[j]
            j += 1

    df["finbert_label"] = labels_all
    df["finbert_score"] = scores_all
    return df


# =========================================================
# TICKERS & INDUSTRIES
# =========================================================

TICKER_BLACKLIST = {
    "ETF",
    "ETFS",
    "TSX",
    "CAD",
    "USD",
    "RRSP",
    "TFSA",
    "EPS",
    "NAV",
    "PE",
    "IMO",
}

INDUSTRY_KEYWORDS = {
    "Tech / AI": ["tech", "technology", "ai", "software", "chip", "semiconductor"],
    "Financials / Banks": ["bank", "financial", "insurance", "mortgage", "lender"],
    "Energy": ["oil", "gas", "pipeline", "energy"],
    "Real Estate / REITs": ["reit", "real estate", "property"],
    "Gold / Metals": ["gold", "silver", "mining", "metal", "precious"],
    "Crypto / Blockchain": ["crypto", "bitcoin", "btc", "eth", "ethereum", "blockchain"],
}


def extract_tickers(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    pattern = r"(?:\$)?\b[A-Z]{2,5}\b"
    candidates = re.findall(pattern, text.upper())
    cleaned = []
    for c in candidates:
        token = c.replace("$", "")
        if token in TICKER_BLACKLIST:
            continue
        cleaned.append(token)
    return list(set(cleaned))


def explode_tickers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combined"] = (df["title"].fillna("") + " " + df["body"].fillna("")).astype(str)
    df["tickers"] = df["combined"].apply(extract_tickers)
    exploded = df.explode("tickers").dropna(subset=["tickers"])
    return exploded


def tag_industries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combined"] = (df["title"].fillna("") + " " + df["body"].fillna("")).astype(str)
    df["combined_lower"] = df["combined"].str.lower()

    def find_inds(text: str) -> list[str]:
        hits = []
        for label, keywords in INDUSTRY_KEYWORDS.items():
            for k in keywords:
                if k in text:
                    hits.append(label)
                    break
        return hits

    df["industries"] = df["combined_lower"].apply(find_inds)
    exploded = df.explode("industries").dropna(subset=["industries"])
    return exploded


# =========================================================
# UI – FETCH DATA
# =========================================================

col_btn, _ = st.columns([1, 3])
with col_btn:
    if st.button("🔁 Force refresh data (ignore 24h cache)"):
        fetch_reddit_data.clear()

with st.spinner("Fetching latest Reddit data from public JSON API..."):
    df_raw = fetch_reddit_data()

if df_raw.empty:
    st.error(
        "No data returned from Reddit. This can happen if Reddit temporarily rate-limits "
        "public search. Try again in a few minutes."
    )
    st.stop()

st.success(f"Loaded {len(df_raw)} posts from {len(SUBREDDITS)} subreddits.")

# =========================================================
# APPLY SENTIMENT
# =========================================================

with st.spinner("Running VADER sentiment..."):
    df_vader = apply_vader(df_raw)

with st.spinner("Running FinBERT sentiment (sampled subset)..."):
    df_sent = apply_finbert(df_vader)

df_sent["date"] = df_sent["created_dt"].dt.date

# Tickers & industries
df_tickers = explode_tickers(df_sent)
df_inds = tag_industries(df_sent)

# =========================================================
# SNAPSHOT METRICS
# =========================================================

st.subheader("📊 Snapshot")

total_posts = len(df_sent)
unique_tickers = df_tickers["tickers"].nunique() if not df_tickers.empty else 0
unique_inds = df_inds["industries"].nunique() if not df_inds.empty else 0
time_span = (
    f"{df_sent['created_dt'].min().strftime('%Y-%m-%d %H:%M UTC')} → "
    f"{df_sent['created_dt'].max().strftime('%Y-%m-%d %H:%M UTC')}"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total posts", f"{total_posts}")
c2.metric("Unique tickers", f"{unique_tickers}")
c3.metric("Industries tagged", f"{unique_inds}")
c4.metric("Time span", time_span)

# =========================================================
# TICKER SENTIMENT
# =========================================================

st.subheader("🏷️ Ticker Sentiment (VADER + FinBERT)")

if df_tickers.empty:
    st.info("No tickers detected in the current data window.")
else:
    agg_t = (
        df_tickers.groupby("tickers")
        .agg(
            mentions=("id", "count"),
            avg_vader=("vader_compound", "mean"),
            avg_finbert=("finbert_score", "mean"),
        )
        .reset_index()
    )
    agg_t["overall_sentiment"] = (
        agg_t["avg_vader"].fillna(0) * 0.5 + agg_t["avg_finbert"].fillna(0) * 0.5
    )

    top_n = st.slider(
        "Show top N tickers by mentions", 5, 40, 15, step=5, key="ticker_n"
    )
    top = agg_t.sort_values("mentions", ascending=False).head(top_n)

    st.dataframe(
        top.style.format(
            {
                "mentions": "{:,.0f}",
                "avg_vader": "{:.3f}",
                "avg_finbert": "{:.3f}",
                "overall_sentiment": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    fig_t = px.bar(
        top.sort_values("overall_sentiment", ascending=True),
        x="overall_sentiment",
        y="tickers",
        orientation="h",
        title="Overall sentiment by ticker (higher = more positive)",
        hover_data=["mentions", "avg_vader", "avg_finbert"],
    )
    st.plotly_chart(fig_t, use_container_width=True)

# =========================================================
# INDUSTRY SENTIMENT
# =========================================================

st.subheader("🏭 Industry / Theme Sentiment")

if df_inds.empty:
    st.info("No industry/theme keywords detected yet.")
else:
    agg_i = (
        df_inds.groupby("industries")
        .agg(
            mentions=("id", "count"),
            avg_vader=("vader_compound", "mean"),
            avg_finbert=("finbert_score", "mean"),
        )
        .reset_index()
    )
    agg_i["overall_sentiment"] = (
        agg_i["avg_vader"].fillna(0) * 0.5 + agg_i["avg_finbert"].fillna(0) * 0.5
    )

    st.dataframe(
        agg_i.style.format(
            {
                "mentions": "{:,.0f}",
                "avg_vader": "{:.3f}",
                "avg_finbert": "{:.3f}",
                "overall_sentiment": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    fig_i = px.bar(
        agg_i.sort_values("overall_sentiment", ascending=True),
        x="overall_sentiment",
        y="industries",
        orientation="h",
        title="Overall sentiment by industry/theme",
        hover_data=["mentions", "avg_vader", "avg_finbert"],
    )
    st.plotly_chart(fig_i, use_container_width=True)

# =========================================================
# RAW DATA
# =========================================================

st.subheader("🧾 Underlying Reddit Posts")

with st.expander("Show raw data", expanded=False):
    st.dataframe(
        df_sent[
            [
                "subreddit",
                "created_dt",
                "title",
                "body",
                "score",
                "num_comments",
                "vader_compound",
                "finbert_label",
                "finbert_score",
                "url",
            ]
        ].sort_values("created_dt", ascending=False),
        use_container_width=True,
    )

st.caption(
    "Prototype for research / educational purposes only. Not investment advice. "
    "Data sourced from Reddit via public JSON endpoints."
)