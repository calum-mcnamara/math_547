#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:20:52 2020

@author: Calum
"""

import numpy as np
import networkx as nx
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d
import scipy
from matplotlib.pyplot import figure
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE


def build_fig(title="", axis_off=False, size=(10, 8), dpi=200, 
              y_lab="", x_lab=""):
    """A function to build a matplotlib figure. Primary
    goal is to sandardize the easy stuff.
    Args:
        - title (str): the title of the plot
        - axis_off (bool): should the axis be printed?
        - size (tuple): how big should the plot be?
        - y_lab (str): y axis label
        - x_lab (str): x axis label
    Returns:
        fig (plt.figure)
    """
    fig = plt.figure(figsize=size, 
                     facecolor='w',
                     dpi=dpi)
    fig.suptitle(title, fontsize=15)
    plt.xlabel(x_lab, fontsize=15)
    plt.ylabel(y_lab, fontsize=15)
    
    if axis_off:
        plt.axis('off')
    return fig

def plot_heatmap(arr, cmap="coolwarm", **kwargs):
    """ A function to plot a heatmap.
    Args:
        - arr (np.array): a 2d array
        - cmap (string): color map to pass to sns
        - **kwargs (dict): additional keyword arguments to pass 
            the figure building functions
    Returns:
        - ax (plt.axes._subplots.AxesSubplot)
    """
    fig = build_fig(**kwargs)
    sns.heatmap(arr, cmap=cmap)
    
    
def plot_3d_eigenvectors(labels, v, k=3, **kwargs):
    """A function to plot (3d) the results of a clustering 
    alforithm
    
    Args:
        - labels (list): the int values of the labels
        - k (int): the number of clusters
        - v (np.array): with shape (n, 3)
        
    Returns:
        plot
    """

    fig = build_fig(**kwargs)
    ax = plt.axes(projection='3d')
    
    x = v[:,0]
    y = v[:,1]
    z = v[:,2]
    shift_labels = labels+1
    
    colors = ['Red', 'Blue', 'Green', 'Yellow', 'Orange', "Gray"]

    ax.scatter3D(x, y, z, c=shift_labels, 
                 cmap=matplotlib.colors.ListedColormap(colors[:k]))
    return plt

    
def plot_tnse(labels, points, **kwargs):
    """A function to plot cluster results in 2d space using tsne
    
    Args:
        - labels (np.array): array of int
        - points (nD np.array): an n-dimensional array
        
    Returns:
        - ax (plt.axes._subplots.AxesSubplot)
    """
    fig = build_fig(y_lab="TSNE-2", x_lab="TSNE-1", **kwargs)
    
    embedded = TSNE(n_components=2).fit_transform(points)
    
    shifted_labels = labels + 1
    
    sns.scatterplot(x=embedded[:,0], y=embedded[:,1], 
                    hue=shifted_labels, palette='Set1')
    
    return plt
    
    
def plot_A_clusters(A, labels, cmap='plasma', **kwargs):
    """A function to plot heatmap of adjancey mat
    
    Args:
        - A (np.array): the adjancency matrix
        - labels (np.array): array of int
        - points (nD np.array): an n-dimensional array
        
    Returns:
        - ax (plt.axes._subplots.AxesSubplot)
    """
    _A = np.zeros(A.shape)
        
    for idx, label in enumerate(labels):
        _A[:, idx] = label
    fig = build_fig(**kwargs)
    sns.heatmap(_A, cmap=cmap)