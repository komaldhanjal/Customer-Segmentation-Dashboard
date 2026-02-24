# Customer-Segmentation-Dashboard

An end-to-end Machine Learning application built with Streamlit, Scikit-Learn, and Pandas to perform customer segmentation using K-Means Clustering and Principal Component Analysis (PCA). This project allows users to upload marketing data, identify distinct customer segments, and extract automated business insights.
Features
Interactive Controls: Adjustable slider to select the number of clusters (K) between 2 and 6.

Feature Selection: Users can choose specific features for clustering, including Income, Age, and spending on Wines or Meat products.

Data Preprocessing:

Automatic handling of missing values in the 'Income' column using median imputation.

Feature engineering to calculate 'Age' from 'Year_Birth'.

Feature scaling using StandardScaler to ensure uniform cluster influence.
Machine Learning & Visualization:

K-Means Clustering: Groups customers based on selected behavioral and demographic features.

Silhouette Score: Real-time metric to evaluate the quality and consistency of the clusters.

PCA Visualization: Reduces high-dimensional data to two principal components for a 2D scatter plot visualization of segments.

Business Tools:

Cluster Naming: Dynamic text inputs to assign business-friendly names to each identified segment.

🛠️ Tech Stack
Frontend: Streamlit

Data Analysis: Pandas, NumPy

Machine Learning: Scikit-Learn (KMeans, PCA, StandardScaler)

Visualization: Matplotlib

Automated Insights: Generates a summary table and text-based descriptions of average traits for each segment.

Data Export: Option to download the final dataset with assigned cluster labels as a CSV file.
