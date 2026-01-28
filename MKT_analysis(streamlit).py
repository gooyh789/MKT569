import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import re

st.set_page_config(page_title="Starbucks Review Dashboard", layout="wide")

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("reviews_data.csv")

st.title("⭐ Starbucks Review Dashboard")

# -----------------------------
# Preprocessing
# -----------------------------
df['Date'] = pd.to_datetime(df['Date'], format='%b. %Y')
df['Month'] = df['Date'].dt.to_period('M').astype(str)
df = df.sort_values('Date')

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['clean_review'] = df['Review'].apply(clean_text)
low_review = df[df['Rate'] <= 2].copy()
low_review['clean_review'] = low_review['Review'].apply(clean_text)

# -----------------------------
# Sidebar
# -----------------------------
branch = st.sidebar.selectbox("Select Branch", sorted(df['Branch'].unique()))

df_b = df[df['Branch'] == branch]
low_b = low_review[low_review['Branch'] == branch]

# -----------------------------
# Metrics (Top Row)
# -----------------------------
avg_rate = df_b['Rate'].mean()
low_ratio = (df_b['Rate'] <= 2).mean()
complaint_density = low_b['clean_review'].str.split().str.len().mean()

col1, col2, col3 = st.columns(3)
col1.metric("Average Rating", f"{avg_rate:.2f}")
col2.metric("Low Rating Ratio", f"{low_ratio:.2%}")
col3.metric("Complaint Density", f"{complaint_density:.2f}")

# -----------------------------
# Boxplot + Line Chart (Middle Row)
# -----------------------------
col4, col5 = st.columns(2)

# Boxplot
with col4:
    st.subheader("📦 Rating Distribution")
    fig, ax = plt.subplots(figsize=(5,3))

    df.boxplot(
        column='Rate',
        by='Branch',
        ax=ax,
        boxprops=dict(color='black', linewidth=1.5),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='gray', linewidth=1.5),
        capprops=dict(color='gray', linewidth=1.5),
        flierprops=dict(marker='o', markersize=4, markerfacecolor='blue')
    )

    ax.set_title("")
    plt.suptitle("")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

# Line Chart
with col5:
    st.subheader("📈 Monthly Trend")
    monthly_avg = df.groupby(['Branch','Month'])['Rate'].mean().reset_index()
    trend_b = monthly_avg[monthly_avg['Branch'] == branch]
    trend_b2 = trend_b.copy()
    trend_b2['Month'] = pd.to_datetime(trend_b2['Month'])
    st.line_chart(trend_b2.set_index('Month')['Rate'])

# -----------------------------
# Wordcloud + Keywords (Bottom Row)
# -----------------------------
col6, col7 = st.columns([2,1])

# Wordcloud
with col6:
    st.subheader("☁ Wordcloud (Low Ratings)")
    text = " ".join(low_b['clean_review'])
    if len(text.strip()) > 0:
        wc = WordCloud(width=600, height=300, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.write("No low-rating reviews.")

# Keywords
with col7:
    st.subheader("🔍 Top Keywords")
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(low_b['clean_review'])
    words = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1
    df_wc = pd.DataFrame({'word': words, 'count': counts}).sort_values('count', ascending=False).head(20)
    st.table(df_wc)
