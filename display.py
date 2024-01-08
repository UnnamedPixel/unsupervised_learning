import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import cluster
import time
from sklearn import metrics
import scipy.cluster.hierarchy as shc

"""
# Parser un fichier de donnees au format arff
# data est un tableau d 'exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 featuresfrom sklearn import metrics ( dimension 2 )
# Ex : [[ - 0.499261 , -0.0612356 ] ,
# [ - 1.51369 , 0.265446 ] ,
# [ - 1.60321 , 0.362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster.On retire cette information
path = "./artificial/"
databrut = arff.loadarff(open(path + "xclara.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]

true_label = [[x[2]] for x in databrut[0]]

# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0.499261 , -1.51369 , -1.60321 , ...]
# Ex pour f1 = [ - 0.0612356 , 0.265446 , 0.362039 , ...]
# tous les elements de la premiere colonne
f0 = [element[0] for element in datanp]
# tous les elements de la deuxieme colonne
f1 = [element[1] for element in datanp]

plt.scatter(f0, f1, s=8)
plt.title(" Donnees initiales ")
plt.show()

#
# Les donnees sont dans datanp ( 2 dimensions )
# f0 : valeurs sur la premiere dimension
# f1 : valeur sur la deuxieme dimension
#
print(" Appel KMeans pour une valeur fixee de k ")
tps1 = time.time()
for nb_cluster in range(2, 5):
    k = nb_cluster
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    silhouette_scr = metrics.silhouette_score(datanp, labels)
    davies_bouldin_scr = metrics.davies_bouldin_score(datanp, labels)
    calinski_harabasz_scr = metrics.calinski_harabasz_score(datanp, labels)
    iteration = model.n_iter_
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(" Donnees apres clustering Kmeans ")
    plt.show()
    print(" nb clusters = ", k, " , nb iter = ", iteration, " ,...",
          "runtime = ", round((tps2 - tps1) * 1000, 2), " ms ,")
    print("silhouette score is", silhouette_scr)
    print("davies_bouldin_scr score is", davies_bouldin_scr)
    print("calinski_harabasz_scr score is", calinski_harabasz_scr)
'
# ------------------------------------------------------------------------------------------------------------------


databrut = arff.loadarff(open(path + "banana.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]

true_label = [[x[2]] for x in databrut[0]]

# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0.499261 , -1.51369 , -1.60321 , ...]
# Ex pour f1 = [ - 0.0612356 , 0.265446 , 0.362039 , ...]
# tous les elements de la premiere colonne
f0 = [element[0] for element in datanp]
# tous les elements de la deuxieme colonne
f1 = [element[1] for element in datanp]

plt.scatter(f0, f1, s=8)
plt.title(" Donnees initiales ")
plt.show()

    #
    # Les donnees sont dans datanp ( 2 dimensions )
    # f0 : valeurs sur la premiere dimension'
    # f1 : valeur sur la deuxieme dimension
    #
print(" Appel KMeans pour une valeur fixee de k ")
tps1 = time.time()
for nb_cluster in range(2, 5):
    k = nb_cluster
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    silhouette_scr = metrics.silhouette_score(datanp, labels)
    davies_bouldin_scr = metrics.davies_bouldin_score(datanp, labels)
    calinski_harabasz_scr = metrics.calinski_harabasz_score(datanp, labels)
    iteration = model.n_iter_
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(" Donnees apres clustering Kmeans ")
    plt.show()
    print(" nb clusters = ", k, " , nb iter = ", iteration, " ,...",
          "runtime = ", round((tps2 - tps1) * 1000, 2), " ms ,")
    print("silhouette score is", silhouette_scr)
    print("davies_bouldin_scr score is", davies_bouldin_scr)
    print("calinski_harabasz_scr score is", calinski_harabasz_scr)

# ------------------------------------------------------------------------------------------------------------------


databrut = arff.loadarff(open(path + "2d-10c.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]

true_label = [[x[2]] for x in databrut[0]]

# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0.499261 , -1.51369 , -1.60321 , ...]
# Ex pour f1 = [ - 0.0612356 , 0.265446 , 0.362039 , ...]
# tous les elements de la premiere colonne
f0 = [element[0] for element in datanp]
# tous les elements de la deuxieme colonne
f1 = [element[1] for element in datanp]

plt.scatter(f0, f1, s=8)
plt.title(" Donnees initiales ")
plt.show()

    #
    # Les donnees sont dans datanp ( 2 dimensions )
    # f0 : valeurs sur la premiere dimension
    # f1 : valeur sur la deuxieme dimension
    #
print(" Appel KMeans pour une valeur fixee de k ")
tps1 = time.time()
for nb_cluster in range(2, 5):
    k = nb_cluster
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    silhouette_scr = metrics.silhouette_score(datanp, labels)
    davies_bouldin_scr = metrics.davies_bouldin_score(datanp, labels)
    calinski_harabasz_scr = metrics.calinski_harabasz_score(datanp, labels)
    iteration = model.n_iter_
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(" Donnees apres clustering Kmeans ")
    plt.show()
    print(" nb clusters = ", k, " , nb iter = ", iteration, " ,...",
          "runtime = ", round((tps2 - tps1) * 1000, 2), " ms ,")
    print("silhouette score is", silhouette_scr)
    print("davies_bouldin_scr score is", davies_bouldin_scr)
    print("calinski_harabasz_scr score is", calinski_harabasz_scr)

# ------------------------------------------------------------------------------------------------------------------

databrut = arff.loadarff(open(path + "diamond9.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]

true_label = [[x[2]] for x in databrut[0]]

# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0.499261 , -1.51369 , -1.60321 , ...]
# Ex pour f1 = [ - 0.0612356 , 0.265446 , 0.362039 , ...]
# tous les elements de la premiere colonne
f0 = [element[0] for element in datanp]
# tous les elements de la deuxieme colonne
f1 = [element[1] for element in datanp]

plt.scatter(f0, f1, s=8)
plt.title(" Donnees initiales ")
plt.show()


#
# Les donnees sont dans datanp ( 2 dimensions )
# f0 : valeurs sur la premiere dimension
# f1 : valeur sur la deuxieme dimension
#
print(" Appel KMeans pour une valeur fixee de k ")
tps1 = time.time()
for nb_cluster in range(2, 10):
    k = nb_cluster
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    silhouette_scr = metrics.silhouette_score(datanp, labels)
    davies_bouldin_scr = metrics.davies_bouldin_score(datanp, labels)
    calinski_harabasz_scr = metrics.calinski_harabasz_score(datanp, labels)
    iteration = model.n_iter_
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(" Donnees apres clustering Kmeans ")
    plt.show()
    print(" nb clusters = ", k, " , nb iter = ", iteration, " ,...",
          "runtime = ", round((tps2 - tps1) * 1000, 2), " ms ,")
    print("silhouette score is", silhouette_scr)
    print("davies_bouldin_scr score is", davies_bouldin_scr)
    print("calinski_harabasz_scr score is", calinski_harabasz_scr)
    
# ------------------------------------------------------------------------------------------------------------------


databrut = arff.loadarff(open(path + "2d-4c.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]

true_label = [[x[2]] for x in databrut[0]]

# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0.499261 , -1.51369 , -1.60321 , ...]
# Ex pour f1 = [ - 0.0612356 , 0.265446 , 0.362039 , ...]
# tous les elements de la premiere colonne
f0 = [element[0] for element in datanp]
# tous les elements de la deuxieme colonne
f1 = [element[1] for element in datanp]

plt.scatter(f0, f1, s=8)
plt.title(" Donnees initiales ")
plt.show()

    #
    # Les donnees sont dans datanp ( 2 dimensions )
    # f0 : valeurs sur la premiere dimension
    # f1 : valeur sur la deuxieme dimension
    #
print(" Appel KMeans pour une valeur fixee de k ")
tps1 = time.time()
for nb_cluster in range(2, 5):
    k = nb_cluster
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    silhouette_scr = metrics.silhouette_score(datanp, labels)
    davies_bouldin_scr = metrics.davies_bouldin_score(datanp, labels)
    calinski_harabasz_scr = metrics.calinski_harabasz_score(datanp, labels)
    iteration = model.n_iter_
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(" Donnees apres clustering Kmeans ")
    plt.show()
    print(" nb clusters = ", k, " , nb iter = ", iteration, " ,...",
          "runtime = ", round((tps2 - tps1) * 1000, 2), " ms ,")
    print("silhouette score is", silhouette_scr)
    print("davies_bouldin_scr score is", davies_bouldin_scr)
    print("calinski_harabasz_scr score is", calinski_harabasz_scr)


    
# ------------------------------------------------------------------------------------------------------------------
databrut = arff.loadarff(open(path + "donut2.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]

true_label = [[x[2]] for x in databrut[0]]

# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0.499261 , -1.51369 , -1.60321 , ...]
# Ex pour f1 = [ - 0.0612356 , 0.265446 , 0.362039 , ...]
# tous les elements de la premiere colonne
f0 = [element[0] for element in datanp]
# tous les elements de la deuxieme colonne
f1 = [element[1] for element in datanp]

plt.scatter(f0, f1, s=8)
plt.title(" Donnees initiales ")
plt.show()


#
# Les donnees sont dans datanp ( 2 dimensions )
# f0 : valeurs sur la premiere dimension
# f1 : valeur sur la deuxieme dimension
#
print(" Appel KMeans pour une valeur fixee de k ")
tps1 = time.time()
for nb_cluster in range(2, 5):
    k = nb_cluster
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    silhouette_scr = metrics.silhouette_score(datanp, labels)
    davies_bouldin_scr = metrics.davies_bouldin_score(datanp, labels)
    calinski_harabasz_scr = metrics.calinski_harabasz_score(datanp, labels)
    iteration = model.n_iter_
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(" Donnees apres clustering Kmeans ")
    plt.show()
    print(" nb clusters = ", k, " , nb iter = ", iteration, " ,...",
          "runtime = ", round((tps2 - tps1) * 1000, 2), " ms ,")
    print("silhouette score is", silhouette_scr)
    print("davies_bouldin_scr score is", davies_bouldin_scr)
    print("calinski_harabasz_scr score is", calinski_harabasz_scr)
"""

tps2 = time.time()
# Parser un fichier de donnees au format arff
# data est un tableau d 'exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 featuresfrom sklearn import metrics ( dimension 2 )
# Ex : [[ - 0.499261 , -0.0612356 ] ,
# [ - 1.51369 , 0.265446 ] ,
# [ - 1.60321 , 0.362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster.On retire cette information


def clustering_agglomeratif_w_th(databrut, linkage, threshold) -> None:
    datanp = [[x[0], x[1]] for x in databrut[0]]
    # true_label = [[x[2]] for x in databrut[0]]
    # Donnees dans datanp
    print(" Dendrogramme 'single 'donnees initiales ")
    linked_mat = shc.linkage(datanp, 'single')
    plt.figure(figsize=(12, 12))
    shc.dendrogram(linked_mat,
                   orientation='top',
                   distance_sort='descending',
                   show_leaf_counts=False)
    plt.show()

    # set distance_threshold ( 0 ensures we compute the full tree )
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(
        distance_threshold=threshold, linkage=linkage, n_clusters=None)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    k = model.n_clusters_
    leaves = model.n_leaves_
    # tous les elements de la premiere colonne
    f0 = [element[0] for element in datanp]
    # tous les elements de la deuxieme colonne
    f1 = [element[1] for element in datanp]
    # Affichage clustering
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(" Resultat du clustering ")
    plt.show()
    print(" nb clusters = ", k, " , nb feuilles = ", leaves,
          " runtime = ", round((tps2 - tps1) * 1000, 2), " ms ")


def clustering_agglomeratif_w_nb_cluster(databrut, linkage, n_clusters=1) -> None:
    datanp = [[x[0], x[1]] for x in databrut[0]]
    # true_label = [[x[2]] for x in databrut[0]]
    # Donnees dans datanp
    f0 = [element[0] for element in datanp]
    # tous les elements de la deuxieme colonne
    f1 = [element[1] for element in datanp]
    print(" Dendrogramme 'single 'donnees initiales ")
    linked_mat = shc.linkage(datanp, 'single')
    plt.figure(figsize=(12, 12))
    shc.dendrogram(linked_mat,
                   orientation='top',
                   distance_sort='descending',
                   show_leaf_counts=False)
    plt.show()
    k = n_clusters

    # set the number of clusters
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(
        linkage=linkage, n_clusters=n_clusters)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    kres = model.n_clusters_
    leaves = model.n_leaves_
    # Affichage clustering
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(" Resultat du clustering ")
    plt.show()
    print(" nb clusters = ", k, " , nb feuilles = ", leaves,
          " runtime = ", round((tps2 - tps1) * 1000, 2), " ms ")


def clustering_DBSCAN(databrut, min_samples, eps) -> None:
    datanp = [[x[0], x[1]] for x in databrut[0]]
    # true_label = [[x[2]] for x in databrut[0]]
    # Donnees dans datanp
    f0 = [element[0] for element in datanp]
    # tous les elements de la deuxieme colonne
    f1 = [element[1] for element in datanp]

    tps1 = time.time()
    db = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(datanp)
    tps2 = time.time()

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    # Affichage clustering
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(" Resultat du clustering ")
    plt.show()
    print(" nb clusters estimated = ", n_clusters_, " , nb feuilles = ", "leaves",
          " runtime = ", round((tps2 - tps1) * 1000, 2), " ms ")


path = "./artificial/"
databrut = arff.loadarff(open(path + "2d-4c.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]

true_label = [[x[2]] for x in databrut[0]]
"""
clustering_agglomeratif_w_th(databrut, 'single', 0.5)
clustering_agglomeratif_w_nb_cluster(databrut, 'single', 3)
"""
# ------------------------------------------------------------------------------------------------------------
clustering_DBSCAN(databrut, 15, 5)

clustering_DBSCAN(databrut, 15, 5)