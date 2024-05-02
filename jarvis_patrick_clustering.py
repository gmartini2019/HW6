"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
import numpy as np
from scipy.spatial.distance import cdist
import random

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def adjusted_rand_index(labels_true, labels_pred):
    contingency_table = np.histogram2d(labels_true, labels_pred, bins=(np.unique(labels_true).size, np.unique(labels_pred).size))[0]

    sum_total = np.sum(contingency_table) * (np.sum(contingency_table) - 1) / 2

    sum_rows = np.sum([np.sum(row) * (np.sum(row) - 1) / 2 for row in contingency_table])

    sum_cols = np.sum([np.sum(col) * (np.sum(col) - 1) / 2 for col in contingency_table.T])

    sum_cells = np.sum([cell * (cell - 1) / 2 for cell in contingency_table.flatten()])

    numerator = sum_cells - sum_rows * sum_cols / sum_total
    denominator = 0.5 * (sum_rows + sum_cols) - sum_rows * sum_cols / sum_total

    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def compute_SSE(data, labels):
    """
    Calculate the sum of squared errors (SSE) for a clustering.

    Parameters:
    - data: numpy array of shape (n, 2) containing the data points
    - labels: numpy array of shape (n,) containing the cluster assignments

    Returns:
    - sse: the sum of squared errors
    """
    # ADD STUDENT CODE
    sse = 0.0
    for i in np.unique(labels):
        cluster_points = data[labels == i]
        cluster_center = np.mean(cluster_points, axis=0)
        sse += np.sum((cluster_points - cluster_center) ** 2)
    print(sse)
    return sse

def plot_clusters(data, labels, title):
    plt.figure(figsize=(8, 6))
    s = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.colorbar(label='Cluster label')
    plt.show()
    return s


def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
):
    k = params_dict['k']
    smin = params_dict['smin']

    distances = cdist(data, data, 'euclidean')

    neighbors = np.argsort(distances, axis=1)[:, 1:k+1]

    n_points = data.shape[0]
    cluster_labels = -np.ones(n_points, dtype=int)
    cluster_id = 0

    for i in range(n_points):
        if cluster_labels[i] == -1:  
            current_cluster = []
            for j in range(n_points):
                if cluster_labels[j] == -1:  
                    shared_neighbors = np.intersect1d(neighbors[i], neighbors[j])
                    if shared_neighbors.size >= smin:
                        current_cluster.append(j)
            if current_cluster:
                cluster_labels[current_cluster] = cluster_id
                cluster_id += 1

    sse = 0
    for k in np.unique(cluster_labels):
        cluster_points = data[cluster_labels == k]
        centroid = cluster_points.mean(axis=0)
        sse += np.sum((cluster_points - centroid) ** 2)
    
    ari = adjusted_rand_index(labels, cluster_labels)

    return cluster_labels, sse, ari


def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """



    data = np.load('question1_cluster_data.npy')
    labels = np.load('question1_cluster_labels.npy')

    data = data[:2500]
    labels = labels[:2500]

    data_segments = [data[500*i:500*(i+1)] for i in range(5)]
    label_segments = [labels[500*i:500*(i+1)] for i in range(5)]

    smins = [0.5, 3, 0.2, 1, 0.1, 0.01, 0.9, 4, 2, 0.25]
    k = [3,4,5,6,7,8]


    answers = {}
    answers["jarvis_patrick_function"] = jarvis_patrick

    groups = []

    testing_data = data_segments[0]
    testing_labels = label_segments[0]

    tested_smin = []
    tested_labels = []
    tested_ARI = []
    tested_SSE = []
    tested_k = []

    for index,smin in enumerate(smins):
        tested_smin.append(smin)
        random_k = random.choice(k)
        tested_k.append(random_k)
        computed_labels, sse, ari = jarvis_patrick(testing_data, testing_labels, {'smin':smin, 'k':random_k})

        tested_labels.append(computed_labels)
        tested_ARI.append(ari)
        tested_SSE.append(sse)
    
    best_ari = max(tested_ARI)
    index_of_best_smin = tested_ARI.index(best_ari)

    cluster_analysis = dict()
    cluster_analysis['smin'] = tested_smin[index_of_best_smin]
    cluster_analysis['ARI'] = best_ari
    cluster_analysis['SSE'] = tested_SSE[index_of_best_smin]

    groups.append(cluster_analysis)

    k = tested_k[index_of_best_smin]
    all_sses = []
    all_aris = []
    all_labels = []

    all_sses.append(tested_SSE[index_of_best_smin])
    all_aris.append(best_ari)
    all_labels.append(tested_labels[index_of_best_smin])


    for index,dataset in enumerate(data_segments[:-1]):
        cluster_analysis = dict()
        computed_labels, sse, ari = jarvis_patrick(dataset, label_segments[index], {'smin':tested_smin[index_of_best_smin], 'k':k})
        all_sses.append(sse)
        all_aris.append(ari)
        all_labels.append(computed_labels)

        cluster_analysis['smin'] = tested_smin[index_of_best_smin]
        cluster_analysis['ARI'] = ari
        cluster_analysis['SSE'] = sse

        groups.append(cluster_analysis)


    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]


    mean_sse = np.mean(all_sses)
    mean_ari = np.mean(all_aris)
    std_sse = np.std(all_sses)
    std_ari = np.std(all_aris)

    greatest_ari = max(all_aris)
    index_of_greatest_ari = all_aris.index(greatest_ari)
    
    # A single float
    answers["mean_ARIs"] = mean_ari

    # A single float
    answers["std_ARIs"] = std_ari

    # A single float
    answers["mean_SSEs"] = mean_sse

    # A single float
    answers["std_SSEs"] = std_sse


    smallest_sse = min(tested_SSE)
    index_of_smallest_sse = tested_SSE.index(smallest_sse)
    plot_ARI = plot_clusters(data_segments[index_of_greatest_ari], tested_labels[index_of_greatest_ari], f"Clusters with Largest ARI (SMIN={tested_smin[index_of_best_smin]})")

    plot_SSE = plot_clusters(testing_data, tested_labels[index_of_smallest_sse], f"Clusters with Smallest SSE (SMIN={tested_smin[index_of_smallest_sse]})")

    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE


    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)