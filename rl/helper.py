import random
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


class IQR:
    def __init__(self, data, iqr_multiplier=1.5):
        self.data = data
        self.q1 = np.percentile(data, 25)
        self.q3 = np.percentile(data, 75)
        self.iqr = self.q3 - self.q1
        self.lower_bound = self.q1 - iqr_multiplier * self.iqr
        self.upper_bound = self.q3 + iqr_multiplier * self.iqr

    def is_outlier(self, value):
        return value < self.lower_bound or value > self.upper_bound

    def range(self):
        return self.lower_bound, self.upper_bound


class ModifiedZScore:
    def __init__(self, data, threshold=3.5):
        self.data = data
        self.median = np.median(data)
        self.mad = np.median(np.abs(data - self.median))
        self.lower_bound = self.median - threshold / 0.6745 * self.mad
        self.upper_bound = self.median + threshold / 0.6745 * self.mad

    def is_outlier(self, value):
        return value < self.lower_bound or value > self.upper_bound

    def range(self):
        return self.lower_bound, self.upper_bound


def plot_tsne(data, save_path=None):
    """
    使用 t-SNE 可视化数据
    :param data: 数据集，格式为 [(样本1, 标签1), (样本2, 标签2), ...]
    """
    # sample 500 data
    data = random.sample(data, min(1000, len(data)))
    X = np.array([x for x, _ in data])
    y = np.array([label for _, label in data])

    # 使用 t-SNE 降维到2维
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)

    # 可视化
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="tab20", alpha=0.5
    )
    plt.colorbar(scatter, label="Label")
    plt.title("t-SNE Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
