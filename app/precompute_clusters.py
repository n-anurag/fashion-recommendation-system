import pickle
import numpy as np
from sklearn.cluster import KMeans

# Load the feature vectors (pre-extracted embeddings)
feature_list = np.array(pickle.load(open('./shoppinglyx/embeddings.pkl', 'rb')))
filenames = pickle.load(open('./shoppinglyx/filenames.pkl', 'rb'))

# Perform K-Means clustering
k = 10  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_assignments = kmeans.fit_predict(feature_list)  # Assign each image to a cluster

# Save the cluster assignments and centroids for future use
pickle.dump(cluster_assignments, open('./shoppinglyx/cluster_assignments.pkl', 'wb'))
pickle.dump(kmeans.cluster_centers_, open('./shoppinglyx/cluster_centroids.pkl', 'wb'))

print("K-Means clustering completed and saved.")
