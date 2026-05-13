import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

labels = pd.read_csv("data/class.tsv", header=None)
labels = labels.iloc[:, 0].reset_index(drop=True)

expr = pd.read_csv("data/filtered.tsv.gz", sep="\t")
expr.columns = expr.columns.astype(str)

columns_map = pd.read_csv("data/columns.tsv.gz", sep="\t")

xbp1_id = columns_map.loc[
    columns_map.iloc[:, 1] == "XBP1",
    columns_map.columns[0]
].values[0]

gata3_id = columns_map.loc[
    columns_map.iloc[:, 1] == "GATA3",
    columns_map.columns[0]
].values[0]

xbp1_id = str(xbp1_id)
gata3_id = str(gata3_id)

xbp1 = expr[xbp1_id]
gata3 = expr[gata3_id]

mask = labels == 1

plt.figure(figsize=(7, 6))
plt.scatter(gata3[mask], xbp1[mask], color="red", s=35, label="ER+")
plt.scatter(gata3[~mask], xbp1[~mask], color="black", s=35, label="ER-")
plt.xlabel("GATA3")
plt.ylabel("XBP1")
plt.title("Figure 1a-like Plot")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

gene_var = expr.var(axis=0)
top_genes = gene_var.sort_values(ascending=False).head(5000).index
expr_filtered = expr[top_genes]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(expr_filtered)

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

pc1 = X_pca[:, 0]

plt.figure(figsize=(9, 4))
plt.scatter(pc1, np.full(len(pc1), 2), c=["red" if x == 1 else "black" for x in labels], s=25, label="All")
plt.scatter(pc1[labels == 0], np.full(sum(labels == 0), 1), color="black", s=25, label="ER-")
plt.scatter(pc1[labels == 1], np.full(sum(labels == 1), 0), color="red", s=25, label="ER+")
plt.yticks([2, 1, 0], ["All", "ER-", "ER+"])
plt.xlabel("Projection onto PC1")
plt.title("Figure 1c-like Plot")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

variance = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(8, 5))
plt.bar(range(1, len(variance) + 1), variance)
plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance (%)")
plt.title("Explained Variance by PCs")
plt.xlim(0, 21)
plt.grid(alpha=0.3)
plt.show()

pc1_vals = X_pca[:, 0]
pc2_vals = X_pca[:, 1]

plt.figure(figsize=(7, 6))
plt.scatter(pc1_vals[labels == 1], pc2_vals[labels == 1], color="red", s=35, label="ER+")
plt.scatter(pc1_vals[labels == 0], pc2_vals[labels == 0], color="black", s=35, label="ER-")
plt.xlabel("Projection onto PC1")
plt.ylabel("Projection onto PC2")
plt.title("PCA Projection")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("Expression matrix shape:", expr.shape)
print("PCA output shape:", X_pca.shape)
print("Top 10 explained variances:")
print(variance[:10])
