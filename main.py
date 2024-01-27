import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import cluster
import time
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as shc
import os
import pandas as pd

def define_path(new_path):
    global path
    path = new_path.replace("\\", "/")


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


def display_dataset(datanp,name=""):
    f0 = [element[0] for element in datanp]
    # tous les elements de la deuxieme colonne
    f1 = [element[1] for element in datanp]
    plt.scatter(f0, f1, s=8)
    plt.title(" Initial Data " + str(name))
    plt.show()


def display_dataset_w_label(datanp, label, message=""):
    f0 = [element[0] for element in datanp]
    # tous les elements de la deuxieme colonne
    f1 = [element[1] for element in datanp]
    plt.scatter(f0, f1, c=label, s=8)
    plt.title(f" Data after clustering '{message}'")
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
        
def display_dendogramme(data, message = ""):
    linked_mat = shc.linkage(data, 'single')
    plt.title(f"Dendogramme of data {message}")
    shc.dendrogram(linked_mat,
                   orientation='top',
                   distance_sort='descending',
                   show_leaf_counts=False)
    plt.show()

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
    print(" nb clusters = ", 3, " , nb iter = ", iteration, " ,...",
          "runtime = ", round(runtime * 1000, 2), " ms ,")

    return predicted_labels, iteration, runtime

def clustering_agglomeratif_w_th(databrut, linkage, threshold) -> None:
    # set distance_threshold ( 0 ensures we compute the full tree )
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(
        distance_threshold=threshold, linkage=linkage, n_clusters=None)
    model = model.fit(databrut)
    tps2 = time.time()
    predicted_labels = model.labels_
    nb_cluster_estimated = model.n_clusters_
    leaves = model.n_leaves_

    print(" nb clusters = ", nb_cluster_estimated, " , nb feuilles = ", leaves,
          " runtime = ", round((tps2 - tps1) * 1000, 2), " ms ")
    runtime = tps2 - tps1
    return predicted_labels, nb_cluster_estimated, leaves, runtime


def clustering_agglomeratif_w_nb_cluster(datanp, linkage, n_clusters, metric='euclidean') -> None:
    # set distance_threshold ( 0 ensures we compute the full tree )

    tps1 = time.time()
    model = cluster.AgglomerativeClustering(
        linkage=linkage, n_clusters=n_clusters,  metric =  metric)
    model = model.fit(datanp)
    tps2 = time.time()
    predicted_labels = model.labels_
    nb_cluster_estimated = model.n_clusters_
    leaves = model.n_leaves_

    print(" nb clusters = ", nb_cluster_estimated, " , nb feuilles = ", leaves,
          " runtime = ", round((tps2 - tps1) * 1000, 2), " ms ")
    runtime = tps2 - tps1
    return predicted_labels, nb_cluster_estimated, leaves, runtime

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
            k_to_return = best_k_silhouette
        else:
            if best_k_silhouette == best_k_davies_bouldin:
                k_to_return = best_k_silhouette
                print(f"Meilleur k (Silhouette et Davies-Bouldin) : {best_k_silhouette}")
            elif best_k_silhouette == best_k_calinski_harabasz:
                k_to_return = best_k_silhouette
                print(f"Meilleur k (Silhouette et Calinski-Harabasz) : {best_k_silhouette}")
            elif best_k_davies_bouldin == best_k_calinski_harabasz:
                k_to_return = best_k_davies_bouldin
                print(f"Meilleur k (Davies-Bouldin et Calinski-Harabasz) : {best_k_davies_bouldin}")
            else:
                k_to_return = None
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
    
    return k_to_return
 
def load_dataset_from_txt(ds_name):
    # Charger les données depuis le fichier txt
    global path
    df = pd.read_csv(path + str(ds_name), sep='\s+', header=None, names=['x', 'y'])    # Convertir les données en numpy arrays
    datanp = df[['x', 'y']].to_numpy()

    return datanp

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

# %% KMEANS CLUSTERING
path = os.path.dirname(__file__).replace("\\", "/") + "/dataset_p1/"
define_path(path)


define_path(path)

plt.figure(figsize=(12, 12))  # Set the overall figure size to 12x12


datanp, true_label = load_dataset("xclara.arff")

# Kmeans clustering for the first dataset
plt.subplot(2, 2, 1)
display_dataset(datanp, "xclara.arff")


# Plotting for the first dataset
plt.subplot(2, 2, 2)
predicted_label, iteration, runtime = Kmeans(datanp, 3)

get_silhouette_scr(datanp, predicted_label)
get_davies_bouldin_scr(datanp, predicted_label)
get_calinski_harabasz_scr(datanp, predicted_label)
display_dataset_w_label(datanp, predicted_label, f"Kmeans with K = {3}")

# Load and display the second dataset (banana.arff)
datanp, true_label = load_dataset("banana.arff")


# Kmeans clustering for the second dataset
predicted_label, iteration, runtime = Kmeans(datanp, 2)

# Plotting for the second dataset
plt.subplot(2, 2, 3)
display_dataset(datanp, "banana.arff")
plt.subplot(2, 2, 4)

get_silhouette_scr(datanp, predicted_label)
get_davies_bouldin_scr(datanp, predicted_label)
get_calinski_harabasz_scr(datanp, predicted_label)
display_dataset_w_label(datanp, predicted_label, f"Kmeans with K = {2}")

# Show the plots
plt.tight_layout()
plt.show()
    
# %% Kmeans depend des conditions initiales + forme circulaire
name = "zelnik5.arff"
datanp, true_label = load_dataset(name)

# Define the number of rows and columns for the subplot grid
rows = 3
columns = 3

# Create a figure with a specified size
plt.figure(figsize=(15, 15))

# Initial subplot with the original dataset
plt.subplot(rows, columns, 1)
display_dataset(datanp, name)
plt.title("Original Dataset")

for i in range(1, rows * columns):
    plt.subplot(rows, columns, i + 1)
    
    # Perform K-means clustering for each iteration
    predicted_label, iteration, runtime = Kmeans(datanp, 4)
    
    # Display clustering results and metrics
    get_silhouette_scr(datanp, predicted_label)
    get_davies_bouldin_scr(datanp, predicted_label)
    get_calinski_harabasz_scr(datanp, predicted_label)
    
    display_dataset_w_label(datanp, predicted_label, f"Kmeans with K = {4}, iteration {i}")
    plt.title(f"Iteration {i}\nClusters: {4}, Runtime: {round(runtime * 1000, 2)} ms")
    
    print(f"Iteration {i}: nb clusters = 4, nb iter = {iteration}, runtime = {round(runtime * 1000, 2)} ms")

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the final plot
plt.show()


# %% AGGLOMERATIF CLUSTERING : exemple des méthodes

datanp, true_label = load_dataset("2d-4c.arff")

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
display_dataset(datanp, "2d-4c.arff")

plt.subplot(2, 2, 2)
predicted_label, nb_cluster_estimated, n_noise_, runtime = clustering_agglomeratif_w_th(
    datanp, 'single', 0.5)
display_dataset_w_label(datanp, predicted_label, "clustering agglomeratif with threshold = 0.5")

plt.subplot(2, 2, 3)
predicted_label, nb_cluster_estimated, n_noise_, runtime = clustering_agglomeratif_w_th(
    datanp, 'single', 5)
display_dataset_w_label(datanp, predicted_label, "clustering agglomeratif with threshold = 5")

plt.subplot(2, 2, 4)
predicted_label, nb_cluster_estimated, n_noise_, runtime = clustering_agglomeratif_w_nb_cluster(
    datanp, 'single', 4)
display_dataset_w_label(datanp, predicted_label, "clustering agglomeratif with 4 clusters")

plt.tight_layout()
plt.show()

# %% AGGLOMERATIF CLUSTERING - limite du threshold 
path = os.path.dirname(__file__).replace("\\", "/") + "/dataset_p1/"
define_path(path)
name = "sizes4.arff"
datanp, true_label = load_dataset(name)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
display_dataset(datanp, name)
seuil = 0.6
predicted_label, nb_cluster_estimated, n_noise_, runtime = clustering_agglomeratif_w_th(
    datanp, 'single', seuil)
display_dataset_w_label(datanp, predicted_label, f"clustering agglomeratif with seuil à {seuil}")

plt.subplot(1, 2, 2)
seuil = 2
predicted_label, nb_cluster_estimated, n_noise_, runtime = clustering_agglomeratif_w_th(
    datanp, 'single', seuil)
display_dataset_w_label(datanp, predicted_label, f"clustering agglomeratif with seuil à {seuil}")

plt.tight_layout()
plt.show()


# %% AGGLOMERATIF CLUSTERING - outliers
plt.figure(figsize=(12, 6))


path = os.path.dirname(__file__).replace("\\", "/") + "/dataset_p1/"
define_path(path)
name = "sizes4.arff"
datanp, true_label = load_dataset(name)
plt.subplot(2, 2, 1)

display_dataset(datanp, name)
plt.subplot(2, 2, 2)

display_dendogramme(datanp, name)
predicted_label, nb_cluster_estimated, n_noise_, runtime = clustering_agglomeratif_w_nb_cluster(
    datanp, 'single', 4)
plt.subplot(2, 2, 3)

display_dataset_w_label(datanp, predicted_label, "clustering agglomeratif with 4 clusters")
plt.show()

# %% AGGLOMERATIF CLUSTERING - marche sur les formes convexes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

path = os.path.dirname(__file__).replace("\\", "/") + "/dataset_p1/"
define_path(path)
name = "banana.arff"
nb_cluster = 2
datanp, true_label = load_dataset(name)
display_dataset(datanp, name)
#display_dendogramme(datanp, name)
predicted_label, nb_cluster_estimated, n_noise_, runtime = clustering_agglomeratif_w_nb_cluster(
    datanp, 'single', nb_cluster)

plt.subplot(1, 2, 2)

display_dataset_w_label(datanp, predicted_label, f"clustering agglomeratif with {nb_cluster} clusters")
plt.show()

# %% METRIQUES


plt.figure(figsize=(12, 6))


datanp, true_label = load_dataset("2d-4c.arff")
plt.subplot(1, 2, 1)

display_dataset(datanp,"2d-4c.arff")
plt.subplot(1, 2, 2)

find_best_k(datanp)

plt.tight_layout()
plt.show()

# %% Partie 2



current_file  = os.path.dirname(__file__).replace("\\", "/") + "/dataset_p2/"

all_files = list_files_with_specific_extension(current_file,"txt")


define_path(current_file)

# %% essai des métriques et compa des méthodes

data1 = all_files[1]
dataset = load_dataset_from_txt(data1)

plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
display_dataset(dataset, f"{data1}")

plt.subplot(2, 2, 2)
best_k = find_best_k(dataset, k_values=list(range(2, 20)))
predicted_label, nb_cluster_estimated, n_noise_, runtime = clustering_agglomeratif_w_nb_cluster(
    dataset, 'single', best_k)
plt.subplot(2, 2, 3)

display_dataset_w_label(dataset, predicted_label, f"clustering agglomératif with K = {best_k}")

plt.subplot(2, 2, 4)
predicted_label, iteration, runtime = Kmeans(dataset, best_k)
display_dataset_w_label(dataset, predicted_label, f"Kmeans with K = {best_k}")

plt.tight_layout()
plt.show()

# %% clustering agglomératif via le threshold est sensible aux variances des clusters

data1 = all_files[5]
dataset = load_dataset_from_txt(data1)

plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
display_dataset(dataset, f"{data1}")

plt.subplot(2, 2, 2)
th = 15
predicted_label, nb_cluster_estimated, n_noise_, runtime = clustering_agglomeratif_w_th(
    dataset, 'single', th)
display_dataset_w_label(dataset, predicted_label, f"clustering agglomeratif with threshold={th}")

plt.subplot(2, 2, 3)
th = 400
predicted_label, nb_cluster_estimated, n_noise_, runtime = clustering_agglomeratif_w_th(
    dataset, 'single', th)
display_dataset_w_label(dataset, predicted_label, f"clustering agglomeratif with threshold={th}")

plt.subplot(2, 2, 4)
th = 2500
predicted_label, nb_cluster_estimated, n_noise_, runtime = clustering_agglomeratif_w_th(
    dataset, 'single', th)
display_dataset_w_label(dataset, predicted_label, f"clustering agglomeratif with threshold={th}")

plt.tight_layout()
plt.show()



plt.figure()
plt.subplot(2, 1, 1)
find_best_k(dataset)

# Second subplot
plt.subplot(2, 1, 2)
predicted_label, iteration, runtime = Kmeans(dataset, 8)
display_dataset_w_label(dataset, predicted_label, f"Kmeans with K = {8}")

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# %% runtime 
plt.figure(figsize=(12, 6))


current_file  = os.path.dirname(__file__).replace("\\", "/") + "/dataset_p2/"

all_files = list_files_with_specific_extension(current_file,"txt")
define_path(current_file)
plt.subplot(2, 2, 1)

dataset = load_dataset_from_txt("y1.txt")
display_dataset(dataset, "y1.txt")
predicted_label, iteration, runtime = Kmeans(dataset, 1)
plt.subplot(2, 2, 2)

display_dataset_w_label(dataset, predicted_label, f"Kmeans with K = {1}, runtime = {round(runtime * 1000, 2)} sec")

predicted_label, iteration, leaves, runtime = clustering_agglomeratif_w_nb_cluster(
    dataset, 'single', 1)

plt.subplot(2, 2, 3)

display_dataset_w_label(dataset, predicted_label, f"clustering agglomeratif with K = {1}, runtime = {round(runtime * 1000, 2)} sec")
plt.tight_layout()
plt.show()