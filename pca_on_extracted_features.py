import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the features CSV
features_file = "./Output/extracted_features.csv"
features_df = pd.read_csv(features_file)

# Select numerical columns for PCA
numerical_features = features_df.select_dtypes(include=["float64", "int64"]).drop(
    columns=["soc", "cycle"], errors="ignore"  # Exclude non-continuous columns
)

# Standardize the features (mean=0, variance=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)

# Perform PCA
n_components = 2  # Number of principal components to keep
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(scaled_features)

# Create a DataFrame for PCA results
pca_columns = [f"PC{i+1}" for i in range(n_components)]
pca_df = pd.DataFrame(data=principal_components, columns=pca_columns)

# Optionally add metadata columns back for reference
pca_df["soc"] = features_df["soc"]
pca_df["cycle"] = features_df["cycle"]
pca_df["battery"] = features_df["battery"]

# Save PCA results to a new CSV
pca_output_file = "./Output/pca_features.csv"
pca_df.to_csv(pca_output_file, index=False)
print(f"PCA results saved to {pca_output_file}")

# Visualize the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_df["PC1"], pca_df["PC2"], c=pca_df["soc"],
            cmap="viridis", edgecolor="k")
plt.colorbar(label="State of Charge (SoC)")
plt.title("PCA Results: PC1 vs PC2")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
plt.show()

# Print explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
