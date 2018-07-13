#DATA VISUALISATION MODULE

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from functions import top_feats_per_cluster

class Visual:
  def __init__(self, X, labels, features, clf):
    '''Instantiates class Visual. Takes sparse matrix, classified labels, feature names and classifier as arguments.'''
    self.X = X				
    self.X_dense = X.todense()	
    self.labels = labels
    self.features = features
    self.clf = clf
    self.PCAinit()

  def PCAinit(self):
    '''Make 2D coordinates from the sparse matrix using PCA.'''
    self.pca = PCA(n_components=2).fit(self.X_dense)
    self.coords = self.pca.transform(self.X_dense)
	
  def raw_plot(self):
    '''Print raw data plot.'''
    plt.scatter(self.coords[:, 0], self.coords[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Raw Data Visualisation")
    plt.show()

  def cluster_plot(self):
    '''Print clustered plot.'''
    #Colour needs to be at least the length of the n_clusters.
    label_colors = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", 
                "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]
    colors = [label_colors[i] for i in self.labels]
    plt.scatter(self.coords[:, 0], self.coords[:, 1], c=colors)
    #Plot cluster centers
    centroids = self.clf.cluster_centers_
    centroid_coords = self.pca.transform(centroids)
    plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200, linewidths=2, c='#444d60')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clustered Data Visualisation")
    plt.show()

  def bargraph(self):
    '''Print bar graph for top terms per cluster.'''
    dfs = top_feats_per_cluster(self.X,self.labels,self.features,0.1,25)
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
      ax = fig.add_subplot(1, len(dfs), i+1)
      ax.spines["top"].set_visible(False)
      ax.spines["right"].set_visible(False)
      ax.set_frame_on(False)
      ax.get_xaxis().tick_bottom()
      ax.get_yaxis().tick_left()
      ax.set_xlabel("Tf-Idf Score", labelpad=16, fontsize=14)
      ax.set_title("Cluster = " + str(df.label), fontsize=16)
      ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
      ax.barh(x, df.score, align='center', color='#7530FF')
      ax.set_yticks(x)
      ax.set_ylim([-1, x[-1]+1])
      yticks = ax.set_yticklabels(df.features)
      plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
