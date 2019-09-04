import pandas as pd 
import os 
import numpy as np
from numpy import linalg as LA
import random

def read_data(path, filename):
	path_file = os.path.join(path, filename)
	df = pd.read_csv(path_file)
	# df = df.sample(frac = 1)
	return df

def extract_label_and_feature(df):
	X = df.iloc[:, :-1]
	y = df.iloc[:, -1]
	return y.values, X.values

def centrer_init(X, n_cluster):
	c, r = np.shape(X)
	K_init = X[:n_cluster, :]
	return K_init

def re_assign_center(clusters):
	re_clusters = []
	for cluster in clusters:
		value_cluster = clusters[cluster]
		re_clusters.append(np.mean(value_cluster, axis = 0))
	re_cluster = np.concatenate([re_clusters], axis = 1)
	return re_cluster

def distance_change(centroids_old, centroids):
	distances = []
	for it in range(len(centroids)):
		distance = LA.norm(centroids[it] - centroids_old[it])
		distances.append(distance)
	return (np.mean(distances))

def fit(X, k = 4, condition_break = 0.000001, iteractions = 300):
	r, c = np.shape(X)
	centroids_init = centrer_init(X, k)
	centroids = centroids_init
	iteraction = 0
	while(iteraction <= iteractions):
		cluster = {}
		for x in X:
			distances = []
			label = 1
			for centroid in centroids:
				distance = LA.norm(centroid - x)
				distances.append((distance, label))
				label += 1
			distances.sort()
			min_distance = distances[0]
			cluster.setdefault(min_distance[1], []).append(x)
		centroids_old = centroids
		centroids = re_assign_center(cluster)
		distance_check = distance_change(centroids_old, centroids)
		if distance_check <= condition_break:
			break
		iteraction += 1
	return centroids

from sklearn.cluster import KMeans

if __name__ == '__main__':
	path = 'data'
	filename = 'iris.csv'
	df = read_data(path, filename)
	y, X = extract_label_and_feature(df)
	print('author: QuocTong \n', fit(X, k = 3))
	kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
	print('author: sklearn\n', kmeans.cluster_centers_)