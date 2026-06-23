import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

st.set_page_config(page_title="Sales Sentiment Dashboard",
                   layout="wide")

st.title("📊 Sales & Sentiment Analysis Dashboard")

# Upload Dataset
file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    # Date Conversion
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Month'] = df['Order_Date'].dt.month_name()

    # Sentiment Analysis
    def get_sentiment(text):
        score = TextBlob(str(text)).sentiment.polarity

        if score > 0:
            return "Positive"
        elif score < 0:
            return "Negative"
        else:
            return "Neutral"

    df['Sentiment'] = df['Review'].apply(get_sentiment)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    # KPI Metrics
    total_sales = df['Sales'].sum()
    total_qty = df['Quantity'].sum()

    col1, col2 = st.columns(2)

    col1.metric("Total Sales", f"₹{total_sales:,.0f}")
    col2.metric("Products Sold", total_qty)

    # Region Wise Sales
    st.subheader("Region Wise Sales")

    region_sales = df.groupby('Region')['Sales'].sum()

    fig, ax = plt.subplots()
    region_sales.plot(kind='bar', ax=ax)
    plt.ylabel("Sales")
    st.pyplot(fig)

    best_region = region_sales.idxmax()

    st.success(f"Highest Selling Region: {best_region}")

    # Monthly Sales
    st.subheader("Monthly Sales Trend")

    month_sales = df.groupby('Month')['Sales'].sum()

    fig, ax = plt.subplots(figsize=(10,5))
    month_sales.plot(kind='line', marker='o', ax=ax)
    plt.ylabel("Sales")
    st.pyplot(fig)

    high_month = month_sales.idxmax()
    low_month = month_sales.idxmin()

    st.success(f"Highest Sales Month: {high_month}")
    st.error(f"Lowest Sales Month: {low_month}")

    # Product Wise Sales
    st.subheader("Product Wise Sales")

    product_sales = df.groupby('Product')['Sales'].sum()

    fig, ax = plt.subplots()
    product_sales.plot(kind='pie',
                       autopct='%1.1f%%',
                       ax=ax)
    plt.ylabel("")
    st.pyplot(fig)

    # Sentiment Distribution
    st.subheader("Customer Sentiment")

    sentiment_count = df['Sentiment'].value_counts()

    fig, ax = plt.subplots()
    sentiment_count.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # Why Products are Selling
    st.subheader("Why Products Are Selling?")

    positive_reviews = df[df['Sentiment']=="Positive"]

    st.write(
        positive_reviews[['Product','Review']]
    )

    st.info("""
    Positive reviews indicate why products are selling:
    - Excellent Quality
    - Good Performance
    - Value for Money
    - Fast Speed
    - Useful Features
    """)

    # Insights
    st.subheader("Business Insights")

    st.write(f"""
    ✔ Highest selling region is **{best_region}**

    ✔ Highest sales month is **{high_month}**

    ✔ Lowest sales month is **{low_month}**

    ✔ Positive customer reviews increase sales

    ✔ Negative reviews indicate areas for improvement
    """)