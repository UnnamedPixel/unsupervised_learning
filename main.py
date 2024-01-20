import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import cluster
import time
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as shc
import os


path = os.path.dirname(__file__).replace("\\", "/") + "/artificial/"


def define_path(new_path):
    global path
    path = new_path


def load_dataset(ds_name):
    databrut = arff.loadarff(open(path + str(ds_name), 'r'))
    datanp = [[x[0], x[1]] for x in databrut[0]]
    true_label = [[x[2]] for x in databrut[0]]
    # Affichage en 2D
    # Extraire chaque valeur de features pour en faire une liste
    # Ex pour f0 = [ - 0.499261 , -1.51369 , -1.60321 , ...]
    # Ex pour f1 = [ - 0.0612356 , 0.265446 , 0.362039 , ...]
    # tous les elements de la premiere colonne
    return datanp, true_label


def display_dataset(datanp):
    f0 = [element[0] for element in datanp]
    # tous les elements de la deuxieme colonne
    f1 = [element[1] for element in datanp]
    plt.scatter(f0, f1, s=8)
    plt.title(" Donnees initiales ")
    plt.show()


def display_dataset_w_label(datanp, label, clustering_method=""):
    f0 = [element[0] for element in datanp]
    # tous les elements de la deuxieme colonne
    f1 = [element[1] for element in datanp]
    plt.scatter(f0, f1, c=label, s=8)
    plt.title(f" Donnees apres clustering {clustering_method}")
    plt.show()

# SCORE


def get_silhouette_scr(data, predicted_label):
    silhouette_scr = metrics.silhouette_score(data, predicted_label)
    return silhouette_scr


def get_davies_bouldin_scr(data, predicted_label):
    davies_bouldin_scr = metrics.davies_bouldin_score(data, predicted_label)
    return davies_bouldin_scr


def get_calinski_harabasz_scr(data, predicted_label):
    calinski_harabasz_scr = metrics.calinski_harabasz_score(data, predicted_label)
    return calinski_harabasz_scr


def get_metric_score(datapoints, label, metric):
    if metric == "silhouette":
        return get_silhouette_scr(datapoints, label)
    elif metric == "davies_bouldin":
        return get_davies_bouldin_scr(datapoints, label)
    elif metric == "calinski_harabasz":
        return get_calinski_harabasz_scr(datapoints, label)
    else:
        raise ValueError(f"Métrique inconnue : {metric}")
        
def display_dendogramme(data):
    linked_mat = shc.linkage(data, 'single')
    plt.figure(figsize=(12, 12))
    shc.dendrogram(linked_mat,
                   orientation='top',
                   distance_sort='descending',
                   show_leaf_counts=False)
    plt.show()

# %% KMEANS CLUSTERING


def Kmeans(datanp, nb_cluster):
    tps_start = time.time()

    # create model
    model = cluster.KMeans(n_clusters=nb_cluster, init='k-means++',n_init='auto')

    # train model using data
    model.fit(datanp)
    tps_end = time.time()

    iteration = model.n_iter_
    predicted_labels = model.labels_
    runtime = tps_end - tps_start

    return predicted_labels, iteration, runtime


datanp, true_label = load_dataset("xclara.arff")
display_dataset(datanp)

for nb_cluster in range(3, 6):
    predicted_label, iteration, runtime = Kmeans(datanp, nb_cluster)
    get_silhouette_scr(datanp, predicted_label)
    get_davies_bouldin_scr(datanp, predicted_label)
    get_calinski_harabasz_scr(datanp, predicted_label)
    display_dataset_w_label(datanp, predicted_label)
    print(" nb clusters = ", nb_cluster, " , nb iter = ", iteration, " ,...",
          "runtime = ", round(runtime * 1000, 2), " ms ,")


datanp, true_label = load_dataset("banana.arff")
display_dataset(datanp)

for nb_cluster in range(4, 6):
    predicted_label, iteration, runtime = Kmeans(datanp, nb_cluster)
    get_silhouette_scr(datanp, predicted_label)
    get_davies_bouldin_scr(datanp, predicted_label)
    get_calinski_harabasz_scr(datanp, predicted_label)
    display_dataset_w_label(datanp, predicted_label)
    print(" nb clusters = ", nb_cluster, " , nb iter = ", iteration, " ,...",
          "runtime = ", round(runtime * 1000, 2), " ms ,")

datanp, true_label = load_dataset("2d-10c.arff")
display_dataset(datanp)
for nb_cluster in range(8, 10):
    predicted_label, iteration, runtime = Kmeans(datanp, nb_cluster)
    get_silhouette_scr(datanp, predicted_label)
    get_davies_bouldin_scr(datanp, predicted_label)
    get_calinski_harabasz_scr(datanp, predicted_label)
    display_dataset_w_label(datanp, predicted_label)
    print(" nb clusters = ", nb_cluster, " , nb iter = ", iteration, " ,...",
          "runtime = ", round(runtime * 1000, 2), " ms ,")

datanp, true_label = load_dataset("diamond9.arff")
display_dataset(datanp)
for nb_cluster in range(8, 10):
    predicted_label, iteration, runtime = Kmeans(datanp, nb_cluster)
    get_silhouette_scr(datanp, predicted_label)
    get_davies_bouldin_scr(datanp, predicted_label)
    get_calinski_harabasz_scr(datanp, predicted_label)
    display_dataset_w_label(datanp, predicted_label)
    print(" nb clusters = ", nb_cluster, " , nb iter = ", iteration, " ,...",
          "runtime = ", round(runtime * 1000, 2), " ms ,")


datanp, true_label = load_dataset("2d-4c.arff")
display_dataset(datanp)
for nb_cluster in range(3, 6):
    predicted_label, iteration, runtime = Kmeans(datanp, nb_cluster)
    get_silhouette_scr(datanp, predicted_label)
    get_davies_bouldin_scr(datanp, predicted_label)
    get_calinski_harabasz_scr(datanp, predicted_label)
    display_dataset_w_label(datanp, predicted_label)
    print(" nb clusters = ", nb_cluster, " , nb iter = ", iteration, " ,...",
          "runtime = ", round(runtime * 1000, 2), " ms ,")

# %% AGGLOMERATIF CLUSTERING


def clustering_agglomeratif_w_th(databrut, linkage, threshold) -> None:
    # set distance_threshold ( 0 ensures we compute the full tree )

    tps1 = time.time()
    model = cluster.AgglomerativeClustering(
        distance_threshold=threshold, linkage=linkage, n_clusters=None)
    model = model.fit(datanp)
    tps2 = time.time()
    predicted_labels = model.labels_
    nb_cluster_estimated = model.n_clusters_
    leaves = model.n_leaves_

    print(" nb clusters = ", nb_cluster_estimated, " , nb feuilles = ", leaves,
          " runtime = ", round((tps2 - tps1) * 1000, 2), " ms ")
    runtime = tps2 - tps1
    return predicted_labels, nb_cluster_estimated, leaves, runtime


def clustering_agglomeratif_w_nb_cluster(databrut, linkage, n_clusters) -> None:
    # set distance_threshold ( 0 ensures we compute the full tree )

    tps1 = time.time()
    model = cluster.AgglomerativeClustering(
        linkage=linkage, n_clusters=n_clusters)
    model = model.fit(datanp)
    tps2 = time.time()
    predicted_labels = model.labels_
    nb_cluster_estimated = model.n_clusters_
    leaves = model.n_leaves_

    print(" nb clusters = ", nb_cluster_estimated, " , nb feuilles = ", leaves,
          " runtime = ", round((tps2 - tps1) * 1000, 2), " ms ")
    runtime = tps2 - tps1
    return predicted_labels, nb_cluster_estimated, leaves, runtime


datanp, true_label = load_dataset("2d-4c.arff")
display_dataset(datanp)
display_dendogramme(datanp)

predicted_label, nb_cluster_estimated, n_noise_, runtime = clustering_agglomeratif_w_th(
    datanp, 'single', 0.5)
display_dataset_w_label(datanp, predicted_label)

predicted_label, nb_cluster_estimated, n_noise_, runtime = clustering_agglomeratif_w_nb_cluster(
    datanp, 'single', 3)
display_dataset_w_label(datanp, predicted_label)


# %% DBSCAN CLUSTERING

def DBSCAN(datanp, min_samples, eps):
    tps1 = time.time()
    db = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(datanp)
    tps2 = time.time()

    predicted_labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    nb_cluster_estimated = len(set(predicted_labels)) - \
        (1 if -1 in predicted_labels else 0)
    n_noise_ = list(predicted_labels).count(-1)

    print("Estimated number of clusters: %d" % nb_cluster_estimated)
    print("Estimated number of noise points: %d" % n_noise_)

    runtime = tps2 - tps1

    return predicted_labels, nb_cluster_estimated, n_noise_, runtime

def compute_NearestNeighboors(datanp,n_neighbors):
    # Distances k plus proches voisins
    # Donnees dans X
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(datanp)
    distances, indices = neigh.kneighbors(datanp)
    # retirer le point " origine "
    newDistances = np.asarray([np.average(distances[i][1:]) for i in range(0,
                                                                           distances.shape[0])])
    trie = np.sort(newDistances)
    plt.title(" Plus proches voisins ( 5 ) ")
    plt.plot(trie)
    plt.show()



def find_best_k(datapoints, k_values=list(range(2, 11)), metrics=["silhouette", "davies_bouldin", "calinski_harabasz"], verbose=True):

    scr_silh = []
    scr_db = []      
    scr_ch = []

    for k in k_values:
        predicted_labels, iteration, runtime = Kmeans(datapoints, k)
        # Calcul des scores pour chaque métrique choisie
        if len(set(predicted_labels)) < 2:
            print(f"Skipping k={k} as it resulted in only one cluster.")
            for metric in metrics:
                if metric == "silhouette":
                    scr_silh.append(0)
                elif metric == "davies_bouldin":
                    scr_db.append(0)
                    
                elif metric == "calinski_harabasz":
                    scr_ch.append(0)
            continue
        for metric in metrics:
            if metric == "silhouette":
                scr_silh.append(get_silhouette_scr(datapoints, predicted_labels))
            elif metric == "davies_bouldin":
                scr_db.append(get_davies_bouldin_scr(datapoints, predicted_labels))
                
            elif metric == "calinski_harabasz":
                scr_ch.append(get_calinski_harabasz_scr(datapoints, predicted_labels))
                
    best_k_silhouette = k_values[np.argmax(scr_silh)]
    best_k_davies_bouldin = k_values[np.argmin(scr_db)]
    best_k_calinski_harabasz = k_values[np.argmax(scr_ch)]

    if verbose:
        if best_k_silhouette == best_k_davies_bouldin == best_k_calinski_harabasz:
            print(f"Meilleur k : {best_k_silhouette} (unanimité)")
        else:
            if best_k_silhouette == best_k_davies_bouldin:
                print(f"Meilleur k (Silhouette et Davies-Bouldin) : {best_k_silhouette}")
            elif best_k_silhouette == best_k_calinski_harabasz:
                print(f"Meilleur k (Silhouette et Calinski-Harabasz) : {best_k_silhouette}")
            elif best_k_davies_bouldin == best_k_calinski_harabasz:
                print(f"Meilleur k (Davies-Bouldin et Calinski-Harabasz) : {best_k_davies_bouldin}")
            else:
                print(f"Meilleur k (Silhouette) : {best_k_silhouette}")
                print(f"Meilleur k (Davies-Bouldin) : {best_k_davies_bouldin}")
                print(f"Meilleur k (Calinski-Harabasz) : {best_k_calinski_harabasz}")
                
    if verbose:
        # Normaliser les scores
        max_silh = max(scr_silh)
        max_db = max(scr_db)
        max_ch = max(scr_ch)
        
        scr_silh_normalized = [score / max_silh for score in scr_silh]
        scr_db_normalized = [score / max_db for score in scr_db]
        scr_ch_normalized = [score / max_ch for score in scr_ch]
        
        # Afficher les courbes normalisées
        plt.plot(k_values, scr_silh_normalized, label='Silhouette')
        plt.plot(k_values, scr_db_normalized, label='Davies-Bouldin')
        plt.plot(k_values, scr_ch_normalized, label='Calinski-Harabasz')
        plt.xlabel('Nombre de clusters (k)')
        plt.ylabel('Scores normalisés')
        plt.legend()
        plt.show()
    
    
datanp, true_label = load_dataset("2d-4c.arff")
display_dataset(datanp)
predicted_label, nb_cluster_estimated, n_noise_, runtime = DBSCAN(
    datanp, 15, 5)
display_dataset_w_label(datanp, predicted_label)

predicted_label, nb_cluster_estimated, n_noise_, runtime = DBSCAN(
    datanp, 25, 15)
display_dataset_w_label(datanp, predicted_label)

find_best_k(datanp)
# %% TESTs

def list_files_with_specific_extension(folder_path, specific_extension):
    files_with_extension = []
    for file in os.listdir(folder_path):
        _, extension = os.path.splitext(file)
        if extension == f".{specific_extension}":
            files_with_extension.append(file)
    return files_with_extension

def group_files_by_name(folder_path):
    file_groups = {}
    for file in os.listdir(folder_path):
        filename, extension = os.path.splitext(file)
        base_name = filename.rstrip('1234567890')
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(file)
    
    return list(file_groups.values())

# Utilisation de la fonction
dossier = os.path.dirname(__file__).replace("\\", "/") + "/artificial/"  # Remplacez ceci par le chemin de votre dossier
all_files = group_files_by_name(dossier)

for shape in all_files:
    for path_to_data in shape:
        print(path+path_to_data)