# streamlit_app.py
# Advanced Customer Segmentation Dashboard (K-Means + PCA)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# PAGE CONFIG
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# TITLE
st.title("📊 Customer Segmentation Dashboard")
st.write("End-to-end ML app using **K-Means + PCA** with business insights")

#  SIDEBAR    
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

k = st.sidebar.slider("Select number of clusters (K)", 2, 6, 3)

st.sidebar.subheader("Select Features")
use_income = st.sidebar.checkbox("Income", True)
use_age = st.sidebar.checkbox("Age", True)
use_wines = st.sidebar.checkbox("Wine Spending", True)
use_meat = st.sidebar.checkbox("Meat Spending", True)

# MAIN LOGIC
if uploaded_file is not None:
    # Load data (Kaggle dataset is tab-separated)
    df = pd.read_csv(uploaded_file, sep='\t')

    # Handle missing values
    if 'Income' in df.columns:
        df['Income'] = df['Income'].fillna(df['Income'].median())

    # Feature engineering
    if 'Year_Birth' in df.columns:
        df['Age'] = 2024 - df['Year_Birth']

    # Feature selection
    feature_cols = []
    if use_income and 'Income' in df.columns:
        feature_cols.append('Income')
    if use_age and 'Age' in df.columns:
        feature_cols.append('Age')
    if use_wines and 'MntWines' in df.columns:
        feature_cols.append('MntWines')
    if use_meat and 'MntMeatProducts' in df.columns:
        feature_cols.append('MntMeatProducts')

    if len(feature_cols) < 2:
        st.warning("Please select at least 2 features for clustering")
    else:
        X = df[feature_cols]

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # Silhouette Score
        sil_score = silhouette_score(X_scaled, df['Cluster'])

        # ---------------- METRICS ----------------
        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", round(sil_score, 3))
        col2.metric("Total Customers", df.shape[0])
        col3.metric("Number of Segments", k)

        # ---------------- PCA ----------------
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = df['Cluster']

        # ---------------- CLUSTER NAMING ----------------
        st.subheader("🏷️ Cluster Naming")
        cluster_names = {}
        for i in range(k):
            cluster_names[i] = st.text_input(f"Name for Cluster {i}", f"Segment {i}")

        df['Customer_Segment'] = df['Cluster'].map(cluster_names)
        pca_df['Customer_Segment'] = df['Customer_Segment']

        # ---------------- VISUALIZATION ----------------
        st.subheader("📈 PCA Cluster Visualization")
        fig, ax = plt.subplots(figsize=(7, 5))
        for segment in pca_df['Customer_Segment'].unique():
            subset = pca_df[pca_df['Customer_Segment'] == segment]
            ax.scatter(subset['PC1'], subset['PC2'], alpha=0.5, s=30, label=segment)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # ---------------- CLUSTER SUMMARY ----------------
        st.subheader("📋 Cluster Summary")
        summary = df.groupby('Customer_Segment')[feature_cols].mean().round(2)
        st.dataframe(summary)

        # ---------------- BUSINESS INSIGHTS ----------------
        st.subheader("💡 Auto Business Insights")
        for seg, row in summary.iterrows():
            st.write(f"• **{seg}** customers show average values: {row.to_dict()}")

        # ---------------- DOWNLOAD ----------------
        st.subheader("⬇️ Download Result")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Clustered Dataset",
            data=csv,
            file_name="customer_segmentation_output.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Upload a CSV file from the sidebar to start")
