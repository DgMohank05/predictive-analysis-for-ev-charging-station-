# ev_app/clustering.py
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt

def cluster_stations(df, eps=0.1, min_samples=5):
    """
    Cluster stations using DBSCAN algorithm based on lat and lon.
    """
    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(df[['lat', 'lon']])
    df['cluster'] = db.labels_

    # Plot the clusters
    plt.scatter(df['lon'], df['lat'], c=df['cluster'], cmap='viridis')
    plt.title('EV Charging Station Clusters')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar()
    plt.show()

    return df
