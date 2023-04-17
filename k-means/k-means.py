import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from data_process.generate_Vt import generate_xt
from sklearn import preprocessing


def data_normalisation():
    normalised_data = preprocessing.normalize(generate_xt())
    return normalised_data


class GenerateKmeans:
    def __init__(self):
        self.clusters = 5
        self.init = 10

    def generate_clusters(self) -> pd.DataFrame:
        kmeans = KMeans(init='random', n_clusters=self.clusters, n_init=self.init)
        kmeans.fit(data_normalisation())
        data_clustered = pd.DataFrame(generate_xt(), index=generate_xt().index,
                                      columns=generate_xt().columns)
        data_clustered["clusters"] = kmeans.labels_
        return data_clustered

    # assuming clusters =5 based on elbow method
    # clusters are appended to non-normalised xt

    def generate_centroid(self) -> np.array:
        kmeans = KMeans(init='random', n_clusters=self.clusters, n_init=self.init)
        kmeans.fit(data_normalisation())
        return kmeans.cluster_centers_
