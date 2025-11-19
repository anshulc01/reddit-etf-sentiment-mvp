
# Reddit ETF & Industry Sentiment – MVP

This is a **Streamlit** MVP that:
- Scrapes investing-related subreddits via the **Reddit API**
- Extracts ETF & stock tickers and industry themes
- Applies **VADER** and **FinBERT** sentiment
- Aggregates and visualizes sentiment in a simple dashboard

## Subreddits

- r/PersonalFinanceCanada
- r/investing
- r/ETFs
- r/stocks
- r/canadianinvestor

## How it works (high level)

1. Uses PRAW (Python Reddit API Wrapper) to query recent posts & comments (last 24 hours).
2. Cleans text and runs:
   - VADER sentiment on all items
   - FinBERT sentiment on a sampled subset for performance
3. Extracts ticker symbols via a simple regex-based heuristic.
4. Tags posts with industry themes based on keyword lists (Tech, Banks, REITs, etc.).
5. Aggregates sentiment per ticker and per industry and displays in a Streamlit dashboard.

## Running locally

1. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Reddit API credentials as environment variables:

```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
export REDDIT_USER_AGENT="etf-sentiment-mvp/0.1 by your_reddit_username"
```

4. Run the app:

```bash
streamlit run app.py
```

## Deploying on Streamlit Cloud

1. Push this folder to a new **GitHub** repository.
2. Go to Streamlit Cloud and create a new app from the repo.
3. In **App settings → Secrets**, add:

```toml
REDDIT_CLIENT_ID = "your_client_id"
REDDIT_CLIENT_SECRET = "your_client_secret"
REDDIT_USER_AGENT = "etf-sentiment-mvp/0.1 by your_reddit_username"
```

4. Set the main file to `app.py`.
5. Deploy. You’ll get a shareable URL you can send to RBC, professors, etc.

## Notes / Next Steps

- FinBERT is run only on a sample of rows to keep performance acceptable.
- Ticker extraction is intentionally simple for the MVP; can be improved using symbol lists.
- Industry classification uses keyword-based tagging, which can be upgraded to ML models later.

This is a **prototype** for research / educational purposes – not investment advice.
