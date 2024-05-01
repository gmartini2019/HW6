
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from scipy.cluster.vq import kmeans2

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def safe_mean(vectors):
    if vectors.size == 0:
        return np.zeros(vectors.shape[1])
    else:
        return vectors.mean(axis=0)

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

def plot_eigenvalues(eigenvalues, title):
    plt.figure(figsize=(10, 6))
    s = plt.plot(eigenvalues, marker='o')
    plt.title(title)
    plt.xlabel("Index of Eigenvalue")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()
    return s

def adjusted_rand_index(labels_true, labels_pred):
    contingency_table = np.histogram2d(
        labels_true,
        labels_pred,
        bins=(np.unique(labels_true).size, np.unique(labels_pred).size),
    )[0]

    # Sum over rows and columns
    sum_combinations_rows = np.sum(
        [np.sum(nj) * (np.sum(nj) - 1) / 2 for nj in contingency_table]
    )
    sum_combinations_cols = np.sum(
        [np.sum(ni) * (np.sum(ni) - 1) / 2 for ni in contingency_table.T]
    )

    # Sum of combinations for all elements
    N = np.sum(contingency_table)
    sum_combinations_total = N * (N - 1) / 2

    # Calculate ARI
    ari = (
        np.sum([np.sum(n_ij) * (np.sum(n_ij) - 1) / 2 for n_ij in contingency_table])
        - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    ) / (
        (sum_combinations_rows + sum_combinations_cols) / 2
        - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    )

    return ari

def spectral(data, labels ,params_dict):
    n_clusters = params_dict['k']
    sigma = params_dict['sigma']

    pairwise_distances = squareform(pdist(data, 'euclidean'))
    adjacency_matrix = np.exp(-pairwise_distances ** 2 / (2.0 * sigma ** 2))

    D = np.diag(adjacency_matrix.sum(axis=1))
    L = D - adjacency_matrix  

    eigenvalues, eigenvectors = eigh(L)
    sorted_indices = np.argsort(eigenvalues)
    V = eigenvectors[:, sorted_indices[1:n_clusters + 1]]  

    V_normalized = V / np.linalg.norm(V, axis=1, keepdims=True)

    centroids, computed_labels = kmeans2(V_normalized, n_clusters, minit='points', iter=100)

    centroids = np.array([V_normalized[computed_labels == k].mean(axis=0) for k in range(n_clusters)])
    sse = sum(np.sum((V_normalized[computed_labels == k] - centroids[k]) ** 2) for k in range(n_clusters))

    ari = adjusted_rand_index(labels, computed_labels)
    return computed_labels, sse, ari, eigenvalues



def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}
    data = np.load('question1_cluster_data.npy')
    labels = np.load('question1_cluster_labels.npy')

    data = data[:5000]
    labels = labels[:5000]

    data_segments = [data[1000*i:1000*(i+1)] for i in range(5)]
    label_segments = [labels[1000*i:1000*(i+1)] for i in range(5)]

    sigmas = [0.09, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 0.8, 0.95, 2.5]

    groups = []

    testing_data = data_segments[0]
    testing_labels = label_segments[0]

    tested_sigmas = []
    tested_labels = []
    tested_ARI = []
    tested_SSE = []
    tested_eigens = []

    for sigma in sigmas:
        tested_sigmas.append(sigma)
        computed_labels, sse, ari, eigenvalues = spectral(testing_data, testing_labels, {'sigma':sigma, 'k':5})

        tested_labels.append(computed_labels)
        tested_ARI.append(ari)
        tested_SSE.append(sse)
        tested_eigens.append(eigenvalues)
    
    best_ari = max(tested_ARI)
    index_of_best_sigma = tested_ARI.index(best_ari)

    cluster_analysis = dict()
    cluster_analysis['sigma'] = tested_sigmas[index_of_best_sigma]
    cluster_analysis['ARI'] = best_ari
    cluster_analysis['SSE'] = tested_SSE[index_of_best_sigma]

    groups.append(cluster_analysis)

    




### CALCULATE BEST SIGMA ON ONE SLICE --> LARGEST ARI
### WITH THAT SIGMA, CALCULATE THE OTHER PARAMETERS FOR THE OTHER 4 SLICES

### MEAN ARI/STD ARI/ MEAN SSE/ STD SSE

### SPECTRAL ONLY SIGMA AND SSE VALUES


    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above



    all_sses = []
    all_aris = []
    all_eigens = []
    all_labels = []

    all_sses.append(tested_SSE[index_of_best_sigma])
    all_aris.append(best_ari)
    all_eigens.append(tested_eigens[index_of_best_sigma])
    all_labels.append(tested_labels[index_of_best_sigma])

    for index,dataset in enumerate(data_segments[:-1]):
        cluster_analysis = dict()
        computed_labels, sse, ari, eigenvalues = spectral(dataset, label_segments[index], {'sigma':tested_sigmas[index_of_best_sigma], 'k':5})
        all_sses.append(sse)
        all_aris.append(ari)
        all_eigens.append(eigenvalues)
        all_labels.append(computed_labels)

        cluster_analysis['sigma'] = tested_sigmas[index_of_best_sigma]
        cluster_analysis['ARI'] = ari
        cluster_analysis['SSE'] = sse

        groups.append(cluster_analysis)

    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]['SSE']

    greatest_ari = max(all_aris)
    index_of_greatest_ari = all_aris.index(greatest_ari)



    mean_ARIs = np.mean(all_aris)
    print(mean_ARIs)

    mean_SSEs = np.mean(all_sses)
    print(mean_SSEs)

    std_ARIs = np.std(all_aris)
    print(std_ARIs)

    std_SSEs = np.std(all_sses)
    print(std_SSEs)


    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.
    smallest_sse = min(tested_SSE)
    index_of_smallest_sse = tested_SSE.index(smallest_sse)
    plot_ARI = plot_clusters(data_segments[index_of_greatest_ari], tested_labels[index_of_greatest_ari], f"Clusters with Largest ARI (Sigma={tested_sigmas[index_of_best_sigma]})")

    plot_SSE = plot_clusters(testing_data, tested_labels[index_of_best_sigma], f"Clusters with Smallest SSE (Sigma={tested_sigmas[index_of_best_sigma]})")


    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.

    eigenvalues_of_interest = all_eigens[index_of_greatest_ari]
    plot_eig = plot_eigenvalues(eigenvalues_of_interest, "Eigenvalues for Configuration with Largest ARI")
   
    answers["eigenvalue plot"] = plot_eig
    answers["mean_ARIs"] = mean_ARIs
    answers["std_ARIs"] = std_ARIs
    answers["mean_SSEs"] = mean_SSEs
    answers["std_SSEs"] = std_SSEs
    return answers



# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)