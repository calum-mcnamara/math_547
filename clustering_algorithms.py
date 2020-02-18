#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:21:54 2020

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


def load_data(filepath="data:/data_mat.csv"):
    """A function to load the data matrix
    
    Args:
        - filepath (str): filepath of the data matrix
        
    Returns
        mat (2d np.array): a matrix with the data
    """
    return np.genfromtxt(filepath, delimiter=',')


def compute_D(W):
    """A function to compute D, given W
    
    Args:
        - W (2d np.array): adjency matrix
        
    Returns:
        - D (2d dioagonal np.array)
    """
    D = np.zeros(W.shape)
    
    for i in range(0, len(W)):
        D[i,i] = np.sum(W[:,i])
        
    return D


def normlize_rows(eigenvectors):
    """A fucntion to normalize eigenvectors
    row-wise across n eigenvectors
    
    Args:
        - eigenvectors (np.array): the n eigenvectors
    
    Returns:
        - T (np.array): eigenvectors normalized
    """
    T = np.zeros(eigenvectors.shape)
    for idx, row in enumerate(eigenvectors):
        T[idx] = abs(row / np.linalg.norm(row, ord=1))
    return T



def sp_clustering_1(k=3):
    """A function to compute the spectral
    clustering using the first method.
    
    1. Get adjencency matrix W
    2. Compute degree matrix D
    3. Compute L (unnormalized) 
    4. Get first k eigenvectors from L
    5. Let k eigenvectors be columns of U
    6. TODO
    7. Cluster U using k-means
    
    Args:
        - k (int): the number of clusters
        
    Returns:
        - clustrer labels
    """
    A = load_data()
    D = compute_D(A)
    L = D - A
    
    w, v = np.linalg.eig(L)
    
    v = v[:,0:k]
    
    cluster = KMeans(n_clusters=k).fit(v)
    return cluster.labels_, v


def sp_clustering_2(k=3):
    """A function to compute the spectral
    clustering using the first method.
    
    1. Get adjencency matrix W
    2. Compute degree matrix D
    3. Compute L (unnormalized) 
    4. Get first k eigenvectors from the generalized 
        `solution Lu = lambda Du`
    5. Let k eigenvectors be columns of U
    6. TODO
    7. Cluster U using k-means
    
    Args:
        - k (int): the number of clusters
        
    Returns:
        - clustrer labels
    """
    A = load_data()
    D = compute_D(A)
    L = D - A
    w, v = scipy.linalg.eigh(L, D, eigvals_only=False)
    v = v[:,0:k]
    
    cluster = KMeans(n_clusters=k).fit(v)
    return cluster.labels_, v


def sp_clustering_3(k=3):
    """A function to compute the spectral
    clustering using the first method.
    
    1. Get adjencency matrix W
    2. Compute degree matrix D
    3. Compute notmalized L :
        L = D^-1/2 L D^-1/2 
    5. Let k eigenvectors be columns of U J
    6. Normlize rows of J
    7. Cluster J using k-means
    
    Args:
        - k (int): the number of clusters
        
    Returns:
        - clustrer labels
    """
    A = load_data()
    _L = scipy.sparse.csgraph.laplacian(A, normed=True) 
    w, v = scipy.linalg.eigh(_L)
    _v = normlize_rows(v[:,0:k])
    
    cluster = KMeans(n_clusters=k).fit(_v)

    return cluster.labels_, _v


    
    
    
def coopers_algorithm(k=3):
    """"""