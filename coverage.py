import numpy as np
import pandas as pd
#from scaling import RobustStandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
SEED=42

class Coverage:

    def __init__(self, samples_df:pd.DataFrame, total_df:pd.DataFrame):
        """
        Compute coverage between of simulated data space by user samples data
        :param samples_df: user samples data (design flow forced data at framing)
        :param total_df: simulated data to populate space based on min max data given by user
        """
        self.samples_df = samples_df
        self.total_df = total_df.sample(1000, random_state=SEED)
        self.normalized_samples_df = pd.DataFrame()
        self.normalized_total_df = pd.DataFrame()
        self.coverage = 0
        self.scaler = MinMaxScaler()
        self.scale_data()

    def scale_data(self):
        """
        scale samples and total data
        :return: None
        """
        all_df = pd.concat([self.samples_df, self.total_df]).reset_index(drop=True)
        self.scaler.fit(all_df)
        self.normalized_total_df = pd.DataFrame(self.scaler.transform(self.total_df))
        self.normalized_samples_df = pd.DataFrame(self.scaler.transform(self.samples_df))

    def compute_distances(self):
        """
        computes distances between simulated data and user data
        :return: distances as a list
        """
        all_distances = cdist(self.normalized_total_df.astype(float).values, self.normalized_samples_df.astype(float).values, metric='euclidean')
        min_distances = np.min(all_distances, axis=1)
        return min_distances.tolist()

    def get_coverage(self, threshold=None):
        """
        :param threshold: limit distance authorized for data coverage computation, default=None
        :return: coverage in percentage
        """
        distances = self.compute_distances()
        self.total_df['distances'] = distances
        if threshold :
            filtered_total_df = self.total_df[self.total_df['distances'] <= threshold]
        else :
            filtered_total_df = self.total_df.copy()
        self.coverage = round((filtered_total_df.shape[0] / self.total_df.shape[0]) * 100, 2)
        self.coverage = int(self.coverage) if str(self.coverage).endswith(".0") else 0 if self.coverage < 0 else self.coverage
        return self.coverage