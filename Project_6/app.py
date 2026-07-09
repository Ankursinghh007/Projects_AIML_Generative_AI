import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

st.set_page_config(page_title="K-Means Clustering", page_icon="🌸")

st.title("🌸 K-Means Clustering on Iris Dataset")

# Load Dataset
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Keep only Petal Features
df = df[['petal length (cm)', 'petal width (cm)']]

st.subheader("Dataset")
st.dataframe(df.head())

# Select Number of Clusters
k = st.slider("Select Number of Clusters", 2, 6, 3)

# Train Model
model = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = model.fit_predict(df)

st.subheader("Clustered Data")
st.dataframe(df.head())

# Plot
fig, ax = plt.subplots(figsize=(7,5))

colors = ["red", "blue", "green", "orange", "purple", "brown"]

for i in range(k):
    cluster = df[df["Cluster"] == i]
    ax.scatter(
        cluster["petal length (cm)"],
        cluster["petal width (cm)"],
        color=colors[i],
        label=f"Cluster {i}"
    )

# Centroids
centers = model.cluster_centers_

ax.scatter(
    centers[:, 0],
    centers[:, 1],
    color="black",
    marker="X",
    s=200,
    label="Centroids"
)

ax.set_xlabel("Petal Length")
ax.set_ylabel("Petal Width")
ax.legend()

st.pyplot(fig)

# Elbow Method
st.subheader("Elbow Method")

sse = []

for i in range(1, 10):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(df[['petal length (cm)', 'petal width (cm)']])
    sse.append(km.inertia_)

fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(range(1,10), sse, marker='o')
ax2.set_xlabel("Number of Clusters")
ax2.set_ylabel("SSE")
ax2.set_title("Elbow Method")

st.pyplot(fig2)
