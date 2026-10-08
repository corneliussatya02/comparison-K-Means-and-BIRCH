 Comparison of K-Means and BIRCH for Bike Sales Clustering

 Overview

This project presents a comparative analysis of the **K-Means** and **BIRCH** clustering algorithms for grouping bike sales data based on sales price segments.

The research uses a large-scale bike sales dataset obtained from Kaggle. The clustering process aims to identify different patterns and characteristics within bike sales transactions and compare the performance of K-Means and BIRCH based on clustering quality and computational time.

The project is implemented using Python and Streamlit, providing an interactive interface for data processing, clustering, evaluation, visualization, and comparison.

---

 Research Objectives

The main objectives of this research are:

- Compare the performance of K-Means and BIRCH clustering algorithms.
- Group bike sales transactions into several clusters based on selected attributes.
- Evaluate clustering quality using Silhouette Score and Davies-Bouldin Index.
- Compare the computational time required by both algorithms.
- Identify the characteristics of each resulting cluster.
- Provide insights into bike sales patterns and customer characteristics.

---

 Dataset

The dataset used in this research is a bike sales dataset obtained from Kaggle.

The clustering process uses five main attributes:

- **Price**
- **Quantity**
- **Bike_Model**
- **State** – obtained by mapping Store_Location into state information
- **Customer_Gender**

These attributes are used to identify patterns and differences among bike sales transactions.

---

 Clustering Algorithms

 K-Means

K-Means is a centroid-based clustering algorithm that divides data into a predefined number of clusters. The algorithm assigns each data point to the nearest cluster centroid and iteratively updates the centroid until the clustering process converges.

 BIRCH

BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) is a hierarchical clustering algorithm designed to efficiently process large datasets by constructing a clustering feature tree.

---

 Clustering Process

The general workflow of this project consists of:

1. Dataset collection
2. Data preprocessing
3. Attribute selection
4. Data transformation and mapping
5. Data normalization
6. K-Means clustering
7. BIRCH clustering
8. Cluster evaluation
9. Cluster visualization
10. Comparison of clustering results
11. Interpretation of cluster characteristics

The clustering process produces **four clusters** for both algorithms.

---

 Evaluation Metrics

Two main clustering evaluation metrics are used:

 Silhouette Score

Silhouette Score measures how similar each data point is to its own cluster compared with other clusters.

A higher Silhouette Score indicates better-defined and more cohesive clusters.

 Davies-Bouldin Index

Davies-Bouldin Index (DBI) measures the similarity between clusters.

A lower DBI indicates better separation between clusters.

 Processing Time

Processing time is also measured to compare the computational efficiency of K-Means and BIRCH.

---

 Results

The evaluation results are presented below:

| Algorithm | Silhouette Score | Davies-Bouldin Index | Processing Time |
|-----------|------------------:|---------------------:|----------------:|
| K-Means   | 0.1868 | 1.9284 | 0.708 seconds |
| BIRCH     | 0.1709 | 2.0431 | 0.658 seconds |

 Performance Comparison

Based on the evaluation results:

- **K-Means achieved a higher Silhouette Score (0.1868)** compared with BIRCH (0.1709).
- **K-Means achieved a lower Davies-Bouldin Index (1.9284)** compared with BIRCH (2.0431).
- **BIRCH was slightly faster**, with a processing time of 0.658 seconds compared with 0.708 seconds for K-Means.
- Based on Silhouette Score and Davies-Bouldin Index, **K-Means produced better clustering quality**.
- BIRCH demonstrated slightly better computational efficiency in terms of processing time.

---

 Cluster Characteristics

The clustering process successfully produced four clusters with relatively balanced data distributions for both algorithms.

Each cluster has different characteristics based on:

- Dominant bike model
- Dominant price
- Dominant state
- Dominant customer gender

 K-Means

The K-Means clusters were mainly characterized by bike models such as:

- BMX
- Cruiser
- Folding Bike

The dominant locations included:

- New York
- Arizona

The dominant customer gender varied across the resulting clusters, with both male and female customers represented.

 BIRCH

The BIRCH clusters were mainly characterized by:

- BMX
- Road Bike

The dominant locations included:

- Texas
- Arizona

The customer gender distribution also varied across the resulting clusters.

These differences indicate that K-Means and BIRCH can produce different segmentation patterns based on the underlying characteristics of the transaction data.

---

 Conclusion

Based on the research results, **K-Means performed better than BIRCH in terms of clustering quality** for the bike sales dataset.

K-Means achieved a Silhouette Score of **0.1868** and a Davies-Bouldin Index of **1.9284**, while BIRCH achieved a Silhouette Score of **0.1709** and a Davies-Bouldin Index of **2.0431**.

Although BIRCH had a slightly faster processing time (**0.658 seconds**) compared with K-Means (**0.708 seconds**), K-Means produced better clustering quality based on both evaluation metrics.

The four clusters generated by each algorithm also demonstrated different characteristics in terms of bike model, price, state, and customer gender. This indicates that both algorithms can be used to identify sales segmentation patterns, although they produce different clustering structures.

Overall, **K-Means is considered more suitable for this research based on clustering quality, while BIRCH provides a slight advantage in computational speed.**

---

 Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

 Application Features

- Dataset upload
- Dataset validation
- Data preprocessing
- Data transformation
- K-Means clustering
- BIRCH clustering
- Silhouette Score evaluation
- Davies-Bouldin Index evaluation
- Processing time comparison
- Cluster visualization
- Cluster characteristic analysis
- Results comparison
- Data/result download

---

 Project Structure

```text
comparison-K-Means-and-BIRCH/
│
├── ComparisonKmeansBIRCH.py
├── bike_sales_50k.csv
├── sepeda.jpeg
├── sepeda1.jpeg
├── sepeda2.jpeg
├── sepeda3.jpeg
├── sepeda4.jpeg
├── sepeda5.jpeg
└── README.md
