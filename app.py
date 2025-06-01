import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px

# Load dataset
df = pd.read_csv("dataset.csv")

# Preprocessing
df_clean = df.dropna().copy()
le = LabelEncoder()
df_clean["location"] = le.fit_transform(df_clean["location"])
df_clean["has_online_delivery"] = df_clean["has_online_delivery"].map({'Yes': 1, 'No': 0})

features = ["location", "cost_for_two", "has_online_delivery"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean["cluster"] = kmeans.fit_predict(X_scaled)

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🍽️ **Restaurant Clustering**")
    menu = st.radio(
        "Navigasi",
        ["📘 Background", "📊 Data Visualization", "⭐ Clustering", "📂 Restaurants By Cluster", "👤 Profile"],
        label_visibility="collapsed",
        index=0
    )


if menu == "📘 Background":
    st.title("📘 Background")
    st.markdown("""
    Dashboard ini dibuat untuk mengelompokkan restoran berdasarkan fitur seperti lokasi, harga, dan layanan online.

    **Metode:** K-Means Clustering  
    **Tujuan:** Membantu analisis segmen restoran di berbagai daerah.
    """)

elif menu == "📊 Data Visualization":
    st.title("📊 Data Visualization")
    st.dataframe(df.head())

    st.write("Distribusi Rating:")
    fig1 = px.histogram(df, x="rating", nbins=20, color_discrete_sequence=["indianred"])
    st.plotly_chart(fig1)

    st.write("Biaya untuk Dua Orang per Lokasi:")
    fig2 = px.box(df, x="location", y="cost_for_two", points="outliers", color="location")
    st.plotly_chart(fig2)

elif menu == "⭐ Clustering":
    st.title("⭐ Clustering Result (KMeans)")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_clean["pca1"] = X_pca[:, 0]
    df_clean["pca2"] = X_pca[:, 1]

    fig3 = px.scatter(df_clean, x="pca1", y="pca2", color=df_clean["cluster"].astype(str),
                      hover_data=["nama_restoran", "location", "cost_for_two"],
                      color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig3)

elif menu == "📂 Restaurants By Cluster":
    st.title("📂 Restaurants By Cluster")
    selected_cluster = st.selectbox("Pilih Cluster:", sorted(df_clean["cluster"].unique()))
    st.dataframe(df_clean[df_clean["cluster"] == selected_cluster][["nama_restoran", "location", "cost_for_two", "rating"]])

elif menu == "👤 Profile":
    st.title("👤 Profile")
    st.markdown("""
    **Nama:** [Tasya Noor Azhila]  
    **NIM:** [2304030002]  
    **Mata Kuliah:** Data Mining  
    **Proyek:** Restaurant Clustering Dashboard
    """)

