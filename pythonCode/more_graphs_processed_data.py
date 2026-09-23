# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%
df = pd.read_csv(r"C:\Users\Velimir\iCloudDrive\Documents\Python for AI\Project\Large set\ufc_processed.csv")

# %%
# 1. Winner distribution

plt.figure(figsize=(6, 4))
df["winner"].value_counts().sort_index().plot(kind="bar")
plt.title("Winner Distribution")
plt.xlabel("Winner")
plt.ylabel("Count")
plt.xticks([0, 1], ["Blue", "Red"], rotation=0)
plt.show()

# %%
# 2. Winner distribution percentage

plt.figure(figsize=(6, 4))
(df["winner"].value_counts(normalize=True).sort_index() * 100).plot(kind="bar")
plt.title("Winner Distribution (%)")
plt.xlabel("Winner")
plt.ylabel("Percentage")
plt.xticks([0, 1], ["Blue", "Red"], rotation=0)
plt.show()

# %%
# 3. Correlation heatmap

plt.figure(figsize=(14, 10))
corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")
plt.show()

# %%
# 4. Correlation with winner

winner_corr = df.corr(numeric_only=True)["winner"].drop("winner").sort_values()

plt.figure(figsize=(8, 12))
winner_corr.plot(kind="barh")
plt.title("Feature Correlation with Winner")
plt.xlabel("Correlation")
plt.ylabel("Feature")
plt.show()

# %%
# 5. Distribution of age difference

plt.figure(figsize=(7, 4))
plt.hist(df["age_diff"], bins=30)
plt.title("Distribution of Age Difference")
plt.xlabel("Age Difference")
plt.ylabel("Count")
plt.show()

# %%
# 6. Distribution of wins difference

plt.figure(figsize=(7, 4))
plt.hist(df["wins_total_diff"], bins=30)
plt.title("Distribution of Wins Difference")
plt.xlabel("Wins Difference")
plt.ylabel("Count")
plt.show()

# %%
# 7. Distribution of losses difference

plt.figure(figsize=(7, 4))
plt.hist(df["losses_total_diff"], bins=30)
plt.title("Distribution of Losses Difference")
plt.xlabel("Losses Difference")
plt.ylabel("Count")
plt.show()

# %%
# 8. Distribution of SLpM difference

plt.figure(figsize=(7, 4))
plt.hist(df["SLpM_total_diff"], bins=30)
plt.title("Distribution of Significant Strikes Landed per Minute Difference")
plt.xlabel("SLpM Difference")
plt.ylabel("Count")
plt.show()

# %%
# 9. Distribution of SApM difference

plt.figure(figsize=(7, 4))
plt.hist(df["SApM_total_diff"], bins=30)
plt.title("Distribution of Significant Strikes Absorbed per Minute Difference")
plt.xlabel("SApM Difference")
plt.ylabel("Count")
plt.show()

# %%
# 10. Boxplot: Age difference by winner

plt.figure(figsize=(7, 4))
plt.boxplot([
    df[df["winner"] == 0]["age_diff"],
    df[df["winner"] == 1]["age_diff"]
])
plt.xticks([1, 2], ["Blue Wins", "Red Wins"])
plt.title("Age Difference by Winner")
plt.ylabel("Age Difference")
plt.show()

# %%
# 11. Boxplot: Wins difference by winner

plt.figure(figsize=(7, 4))
plt.boxplot([
    df[df["winner"] == 0]["wins_total_diff"],
    df[df["winner"] == 1]["wins_total_diff"]
])
plt.xticks([1, 2], ["Blue Wins", "Red Wins"])
plt.title("Wins Difference by Winner")
plt.ylabel("Wins Difference")
plt.show()

# %%
# 12. Boxplot: SLpM difference by winner

plt.figure(figsize=(7, 4))
plt.boxplot([
    df[df["winner"] == 0]["SLpM_total_diff"],
    df[df["winner"] == 1]["SLpM_total_diff"]
])
plt.xticks([1, 2], ["Blue Wins", "Red Wins"])
plt.title("SLpM Difference by Winner")
plt.ylabel("SLpM Difference")
plt.show()

# %%
# 13. Boxplot: SApM difference by winner

plt.figure(figsize=(7, 4))
plt.boxplot([
    df[df["winner"] == 0]["SApM_total_diff"],
    df[df["winner"] == 1]["SApM_total_diff"]
])
plt.xticks([1, 2], ["Blue Wins", "Red Wins"])
plt.title("SApM Difference by Winner")
plt.ylabel("SApM Difference")
plt.show()

# %%
# 14. PCA plot

X = df.drop(columns=["winner"])
y = df["winner"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    alpha=0.6
)

plt.title("PCA Projection of Processed UFC Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Winner")
plt.show()

# %%
# 15. PCA explained variance

plt.figure(figsize=(6, 4))
plt.bar(
    ["PC1", "PC2"],
    pca.explained_variance_ratio_
)

plt.title("PCA Explained Variance")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.show()