from torch.utils.data import Dataset
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import torch


class Data(Dataset):

    def __init__(self, data):
        # Collect samples, both cat and dog and store pairs of (filepath, label) in a simple list.
        self._samples = data

    def __getitem__(self, index):
        # Access the stored path and label for the correct index
        example, label = self._samples[index]

        return example, label

    def __len__(self):
        """Total number of samples"""
        return len(self._samples)

    def get_sample_by_id(self, id_):
        id_index = [path.stem for (path, _) in self._samples].index(id_)
        return self[id_index]


class BlackBoxWrapper():
    def __init__(self, model,
                 num_players,
                 scaler, device):
        self.model = model
        self.num_players = num_players
        self.scaler = scaler
        self.device = device

    def __call__(self, x, S):
        '''
        Evaluate a black-box model.
        Args:
          x: input examples.
          S: coalitions.
        '''
        x = self.scaler.transform(x)
        x = x * S

        x = self.scaler.inverse_transform(x)

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = x.reshape((x.shape[0], self.num_players, 1))
        values = self.model(x)

        return values


def oversample(data, labels, oversampled_class, num_folds):
    oversampled_data = []
    oversampled_labels = []

    for i in range(len(data)):

        if labels.iloc[i] == oversampled_class:
            for r in range(num_folds):
                oversampled_data.append(data.iloc[[i]])
                oversampled_labels.append(labels.iloc[[i]])

        oversampled_data.append(data.iloc[[i]])
        oversampled_labels.append(labels.iloc[[i]])

    oversampled_data = pd.concat(oversampled_data)
    oversampled_labels = pd.concat(oversampled_labels)
    return oversampled_data, oversampled_labels


def transform_to_tensors(data, labels, adj_matrix):
    list_of_tensors = []
    for i in range(len(data)):
        array_to_use = np.array(data.iloc[[i]])[0]
        list_of_tensors.append((torch.FloatTensor(array_to_use).reshape(adj_matrix.shape[0], 1),
                                torch.tensor(labels.iloc[i], dtype=torch.float)))
    return list_of_tensors


def compute_adjacency_matrix(data, self_loop_weight, threshold):
    index_to_name = {i: n for i, n in enumerate(data.columns)}
    name_to_index = {n: i for i, n in enumerate(data.columns)}

    adj_matrix = np.zeros((len(data.columns), len(data.columns)), dtype=float)

    for col_1 in data.columns:
        for col_2 in data.columns:
            if col_1 == col_2:
                adj_matrix[name_to_index[col_1], name_to_index[col_2]] = self_loop_weight
            else:
                corr, _ = pearsonr(data[col_1], data[col_2])
                if np.isnan(corr) or abs(corr) < threshold:
                    adj_matrix[name_to_index[col_1], name_to_index[col_2]] = 0
                else:
                    adj_matrix[name_to_index[col_1], name_to_index[col_2]] = corr

    return adj_matrix, index_to_name, name_to_index


def min_max_normalize(input_data, training_data):
    normalized_data = (input_data - training_data.min()) / (training_data.max() - training_data.min())
    return normalized_data


def plot_roc(clf, X, Y):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    RocCurveDisplay.from_estimator(clf, X, Y)
    plt.show()
