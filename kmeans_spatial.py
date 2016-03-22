'''
function: kmeans_spatial
author: erikleenylen
last update: 3/21/16

how to use: program is to be run from shell, or pasted into Jupyter Notebook with Python2 and played around with

purpose of program: display kmeans clusters analysis of idealized drone locations given felony locations 
(by latitude and longitude) and desired number of clusters (default here is 22)

read more about the tools used:
http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

'''

import csv
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import dateutil
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.cluster.vq import kmeans2, whiten
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
%matplotlib inline


# load the load on Felony locations in NYC
# if you are using a different data set, specify path and filename here
df = pd.read_csv('Felony.csv') 

# let's do this just for Manhattan, though could in principle work for any borough
dfmanhattan=df[df['Borough']=='Manhattan']

# scale lat-lon, since degrees are not equal to one another
scalefactor = 1.317

x = dfmanhattan['Longitude'][:17000]
y = dfmanhattan["Latitude"][:17000]*scalefactor

# specify number of points on which to train the clusters
npoints = -1
points = np.array(zip(x[:npoints],y[:npoints]))
X = points # making a copy for easy use

# SPECIFY KMEANS PARAMETERS
n_clusters = 22
num_init = 300 # number of initializations
init_type = 'k-means++' # 'random' can also be used

# CREATE THE KMEANS MODEL
k_means = KMeans(init=init_type, n_clusters=n_clusters, n_init=num_init)

# FIT THE MODEL TO THE DATA POINTS
k_means.fit(points)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)

rescaled_k_means_cluster_centers=np.array([[_[0],_[1]/scalefactor] for _ in k_means_cluster_centers])
kmeansdfplus = pd.DataFrame(rescaled_k_means_cluster_centers,columns=['lon','lat'])
kmeansdfplus.to_pickle('kmeansdfplus.pkl') # save


fig = plt.figure(figsize=(5,10))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
#colors = ['#4EACC5', '#FF9C34', '#4E9A06']
colors = [(np.random.rand(),np.random.rand(),np.random.rand()) for __ in range(n_clusters)]

ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6)
ax.set_title('KMeans estimate of drone bases',style = 'italic')
ax.set_xticks(())
ax.set_yticks(())

plt.savefig('KMeans for '+str(n_clusters)+' clusters '+init_type+' manhattan w precincts.png')
plt.show()

# end program