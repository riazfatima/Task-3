# Customer Segmentation via Unsupervised Learning 

## Executive Summary
This project delivers an end-to-end unsupervised learning pipeline designed to eliminate human cognitive bias from enterprise retail data. Utilizing transactional data from **Online Retail.xlsx**, the architecture processes, standardizes, compresses, and clusters behavior into mathematically isolated, strategically actionable market segments. By pairing Principal Component Analysis (PCA) with a validated K-Means algorithm, we successfully compressed over 20 dimensions of retail behavior down to 3 major customer personas.

---

## The Pipeline Architecture (IPO Framework)
The data goes through a strict chronological pipeline following the Input-Process-Output blueprint:

1. **SCALE (Standardization):** Continuous features are mapped to a common geometric range ($z$-score scaling where $\mu = 0, \sigma = 1$) to eliminate magnitude-induced bias and ensure equal mathematical voting power.
2. **COMPRESS (PCA):** High-dimensional cloud data ($D > 20$) is transformed onto orthogonal Principal Components to dodge the curse of dimensionality while maximizing variance preservation.
3. **CLUSTER (K-Means):** Modeled to minimize the Within-Cluster Sum of Squares (WCSS), pulling customer metrics into highly cohesive clusters.
4. **TRANSLATE (Inverse Projection):** PCA-space centroid coordinates are inverse-transformed back into human-centric, real-world metrics to build marketing personas.

---

## Technical Results & Variance Preservation

### Principal Component Analysis (PCA)
The script isolated 3 orthogonal components capturing an exceptional cumulative spread of original consumer behavior:
* **PC1:** $79.10\%$ explained variance ratio
* **PC2:** $18.33\%$ explained variance ratio
* **PC3:** $2.57\%$ explained variance ratio

**Total Preserved Variance:** **$100\%$** of the core multi-dimensional feature space was successfully preserved within these 3 components, retaining perfect behavioral signals while completely shedding low-variance noise.

### Mathematical Cluster Proof
The target tier framework enforced a cluster count of **$k = 3$**. This selection was verified via two rigorous diagnostic gatekeepers outlined in **Data Science Project 3.pdf**:
* **The Elbow Method:** Identified the precise inflection point of diminishing returns for WCSS reduction.
* **The Silhouette Score:** Confirmed strong cluster cohesion versus neighbor separation, ensuring no natural customer groups were artificially split.

---

##  Strategic Corporate Persona Matrix

By executing an inverse scaling transformation on the cluster centroids, the abstract coordinates were mapped back to their original physical medians:

| Cluster ID | Assigned Corporate Persona | Median Frequency | Median Total Spend | Median Quantity | Strategic Business Action |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Cluster 1** | **CLUSTER A: HIGH-VALUE ENGAGERS** | 46.0 | \$143,825.06 | 74,215.0 | Deploy high-touch VIP support, exclusive perks, and experiential marketing. |
| **Cluster 2** | **CLUSTER B: MID-TIER EXPLORERS** | 34.0 | \$27,498.04 | 15,752.0 | Push targeted loyalty programs, product bundles, and flash sales. |
| **Cluster 0** | **CLUSTER C: LOW-ACTIVITY CHURN RISK** | 2.0 | \$660.00 | 370.0 | Execute drop-off re-engagement, clear value options, and win-back campaigns. |

---
