
import streamlit as st
import praw

def get_reddit_client():
    client_id = st.secrets["REDDIT_CLIENT_ID"]
    client_secret = st.secrets["REDDIT_CLIENT_SECRET"]
    user_agent = st.secrets["REDDIT_USER_AGENT"]
    username = st.secrets["REDDIT_USERNAME"]
    password = st.secrets["REDDIT_PASSWORD"]

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password,
        check_for_async=False,
    )
    return reddit

st.write("Reddit auth test loaded. This is the patched version.")
