
# Reddit ETF & Industry Sentiment – MVP (Full, Patched)

This is a **Streamlit** MVP that:
- Scrapes investing-related subreddits via the **Reddit API**
- Uses full script authentication (client id/secret + username/password)
- Extracts ETF & stock tickers and industry themes
- Applies **VADER** and **FinBERT** sentiment
- Aggregates and visualizes sentiment in a simple dashboard

## Subreddits

- r/PersonalFinanceCanada
- r/investing
- r/ETFs
- r/stocks
- r/canadianinvestor

## Streamlit Secrets (example)

```toml
REDDIT_CLIENT_ID = "o6zS31tEt0rXylN63wLC2Q"
REDDIT_CLIENT_SECRET = "WDTefleTeB7pbJv-JXo49YE04tr4w"
REDDIT_USER_AGENT = "QueensCapstoneETF/0.1 by CapstoneProject01"
REDDIT_USERNAME = "CapstoneProject01"
REDDIT_PASSWORD = "your_password_here"
```

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

3. Set your Reddit API credentials as environment variables or `.streamlit/secrets.toml`.

4. Run the app:

```bash
streamlit run app.py
```

## Deploying on Streamlit Cloud

1. Push this folder to a new **GitHub** repository.
2. Go to Streamlit Cloud and create a new app from the repo.
3. In **App settings → Secrets**, add the TOML block above.
4. Set the main file to `app.py`.
5. Deploy. You’ll get a shareable URL.

This is a **prototype** for research / educational purposes – not investment advice.
