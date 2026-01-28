import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import re

st.set_page_config(page_title="Starbucks Review Dashboard", layout="wide")

# -----------------------------
# 1) Load Data
# -----------------------------
df = pd.read_csv("reviews_data.csv")

st.title("⭐ Starbucks Review Dashboard (Danbury & Nearby Branches)")

# -----------------------------
# 2) Preprocessing
# -----------------------------
df['Date'] = pd.to_datetime(df['Date'], format='%b. %Y')
df['Month'] = df['Date'].dt.to_period('M').astype(str)
df = df.sort_values('Date')

# Clean text function
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
# 3) Sidebar
# -----------------------------
branch = st.sidebar.selectbox("Select Branch", sorted(df['Branch'].unique()))

df_b = df[df['Branch'] == branch]
low_b = low_review[low_review['Branch'] == branch]

# -----------------------------
# 4) Metrics
# -----------------------------
avg_rate = df_b['Rate'].mean()
low_ratio = (df_b['Rate'] <= 2).mean()
complaint_density = low_b['clean_review'].str.split().str.len().mean()

st.subheader(f"📊 Metrics for {branch}")
col1, col2, col3 = st.columns(3)
col1.metric("Average Rating", f"{avg_rate:.2f}")
col2.metric("Low Rating Ratio", f"{low_ratio:.2%}")
col3.metric("Complaint Density", f"{complaint_density:.2f}")

# -----------------------------
# 5) Boxplot (matplotlib)
# -----------------------------
st.subheader("📦 Rating Distribution by Branch")

fig, ax = plt.subplots(figsize=(6,4))
df.boxplot(column='Rate', by='Branch', ax=ax)
plt.title("Rating Distribution by Branch")
plt.suptitle("")
st.pyplot(fig)

# -----------------------------
# 6) Line Chart (Streamlit)
# -----------------------------
st.subheader("📈 Monthly Average Rating Trend")

monthly_avg = df.groupby(['Branch','Month'])['Rate'].mean().reset_index()
trend_b = monthly_avg[monthly_avg['Branch'] == branch]

trend_b2 = trend_b.copy()
trend_b2['Month'] = pd.to_datetime(trend_b2['Month'])

st.line_chart(trend_b2.set_index('Month')['Rate'])

# -----------------------------
# 7) Wordcloud (matplotlib)
# -----------------------------
st.subheader(f"☁ Wordcloud for Low Ratings ({branch})")

text = " ".join(low_b['clean_review'])

if len(text.strip()) > 0:
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("No low-rating reviews available for this branch.")

# -----------------------------
# 8) Top Keywords (CountVectorizer)
# -----------------------------
st.subheader("🔍 Top Complaint Keywords")

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(low_b['clean_review'])

words = vectorizer.get_feature_names_out()
counts = X.sum(axis=0).A1

df_wc = pd.DataFrame({
    'word': words,
    'count': counts
}).sort_values('count', ascending=False).head(20)

st.table(df_wc)