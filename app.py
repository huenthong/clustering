import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import silhouette_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title
st.title('Obesity Risk Clustering App')

# Load the scaled cleaned CSV file
df = pd.read_csv('pca_df.csv')

# Preprocessing Steps from your code
# Step 1: Create a BMI feature
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# Step 2: Categorize Age into bins
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Adolescent', 'Adult', 'Elderly'])

# Step 3: Create an interaction feature between FAVC and CAEC
df['FAVC_CAEC'] = df['FAVC'] + "_" + df['CAEC']

# Step 4: Create a Healthy Habits Score
df['Healthy_Score'] = df['FCVC'] + df['FAF'] - df['FAVC'].apply(lambda x: 1 if x == 'yes' else 0)

# Step 5: Convert binary categorical features to 1/0
df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'yes': 1, 'no': 0})
df['FAVC'] = df['FAVC'].map({'yes': 1, 'no': 0})

# Step 6: Drop redundant columns after creating engineered features
df.drop(columns=['Height', 'Weight'], inplace=True)

# Removing outliers based on Z-score (numerical features excluding 'Age')
numerical_features_excl_age = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI', 'Healthy_Score']
for feature in numerical_features_excl_age:
    df = df[(np.abs(stats.zscore(df[feature])) < 3)]

# Encoding categorical variables
nominal_features = ['Gender', 'FAVC_CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'CAEC'] 
ordinal_features = ['NObeyesdad', 'Age_Group']  # Ordinal features

# One-Hot Encoding nominal categorical features
onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_nominal = onehot_encoder.fit_transform(df[nominal_features])
encoded_nominal_df = pd.DataFrame(encoded_nominal, columns=onehot_encoder.get_feature_names_out(nominal_features))
encoded_nominal_df.reset_index(drop=True, inplace=True)

# Label Encoding the ordinal features
label_encoder = LabelEncoder()
df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad'])
df['Age_Group'] = label_encoder.fit_transform(df['Age_Group'])

# Drop original nominal columns and reset the index of the original DataFrame
df_numerical_only = df.drop(columns=nominal_features).reset_index(drop=True)

# Concatenate encoded nominal features with the rest of the DataFrame
df_encoded = pd.concat([df_numerical_only, encoded_nominal_df], axis=1)

# Dropping columns with weak correlation with target variable
high_corr_features = ['FAVC_CAEC_yes_Always', 'FAVC_CAEC_no_Frequently', 'CALC_no', 'CALC_Frequently', 'MTRANS_Walking']
weak_corr_features = ['FAVC', 'FCVC', 'NCP', 'CH2O', 'TUE', 'Gender_Male', 'SMOKE_yes', 'SCC_yes']
columns_to_drop = list(set(high_corr_features + weak_corr_features))
df_reduced = df_encoded.drop(columns=columns_to_drop)

# Drop further columns based on correlation
further_columns_to_drop = ['FAVC_CAEC_yes_Sometimes', 'CAEC_Sometimes', 'Healthy_Score', 'CAEC_Frequently']
clean_df_final_minimized = df_reduced.drop(columns=further_columns_to_drop)

# Step 1: Compute the correlation matrix
corr_matrix_minimized = clean_df_final_minimized.corr()

# Step 2: Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clean_df_final_minimized)

# Convert the scaled data back into a DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=clean_df_final_minimized.columns)

# Streamlit Sidebar for model selection and clustering parameters
st.sidebar.header("Clustering Model Settings")
model_choice = st.sidebar.selectbox(
    "Select a clustering model",
    ["KMeans", "MeanShift", "DBSCAN", "Gaussian Mixture Model (GMM)", "Agglomerative Clustering", "Spectral Clustering"]
)

# Sidebar parameters for each model
if model_choice == "KMeans":
    n_clusters = st.sidebar.slider("Number of clusters (K)", 2, 10, 3)
elif model_choice == "MeanShift":
    bandwidth = st.sidebar.slider("Bandwidth", 0.1, 5.0, 1.0)
elif model_choice == "DBSCAN":
    eps = st.sidebar.slider("EPS (Neighborhood Distance)", 0.1, 5.0, 0.5)
    min_samples = st.sidebar.slider("Min Samples", 1, 10, 5)
elif model_choice == "Gaussian Mixture Model (GMM)":
    n_components = st.sidebar.slider("Number of components", 2, 10, 3)
elif model_choice == "Agglomerative Clustering":
    n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
elif model_choice == "Spectral Clustering":
    n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)

# Apply PCA for dimensionality reduction (e.g., first 5 components)
pca = PCA(n_components=5)
pca_df = pca.fit_transform(scaled_df)

# Clustering model based on user selection
if model_choice == "KMeans":
    model = KMeans(n_clusters=n_clusters, random_state=42)
elif model_choice == "MeanShift":
    model = MeanShift(bandwidth=bandwidth)
elif model_choice == "DBSCAN":
    model = DBSCAN(eps=eps, min_samples=min_samples)
elif model_choice == "Gaussian Mixture Model (GMM)":
    model = GaussianMixture(n_components=n_components, random_state=42)
elif model_choice == "Agglomerative Clustering":
    model = AgglomerativeClustering(n_clusters=n_clusters)
elif model_choice == "Spectral Clustering":
    model = SpectralClustering(n_clusters=n_clusters, random_state=42)

# Fit the model and get labels
labels = model.fit_predict(pca_df)

# Calculate silhouette score
if len(np.unique(labels)) > 1:
    silhouette_avg = silhouette_score(pca_df, labels)
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")
else:
    st.write("Silhouette Score: Not applicable (only one cluster detected).")

# Add cluster labels to the DataFrame
df['Cluster'] = labels

# Show cluster sizes (number of records in each cluster)
st.write("Cluster Sizes:")
st.write(df['Cluster'].value_counts())

# Show PCA scatter plot with cluster labels
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=pca_df[:, 0], y=pca_df[:, 1], hue=labels, palette="Set1", ax=ax)
ax.set_title(f"{model_choice} - PCA Visualization")
st.pyplot(fig)

# Show the mean and median statistics for each cluster
st.write("Mean statistics for each cluster:")
st.write(df.groupby('Cluster').mean())

st.write("Median statistics for each cluster:")
st.write(df.groupby('Cluster').median())

# Display PCA explained variance
explained_variance = pca.explained_variance_ratio_
st.write("PCA Explained Variance:")
st.bar_chart(explained_variance)

# Display cumulative explained variance
cumulative_variance = np.cumsum(explained_variance)
st.write("Cumulative Explained Variance:")
st.line_chart(cumulative_variance)


