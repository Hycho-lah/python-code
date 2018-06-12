
# biclustering.py
# HW2, Computational Genomics, Fall 2018
# andrewid:

# Developed and implemented a bi-clustering algorithm that uses bipartite graphs for bi-clustering. A bi-cluster is a cluster containing a subset of the experiments and a subset of the genes. In this problem we will not allow overlap between the bi-clusters, though other methods allow such overlap.
# The script can be run with command line: python biclustering.py alphaCycle.txt
# discretize should return an array of size n × m.
# bicluster and bicluster revised both should return an array of size M × 2, where M is the number of edges in the detected bicluster. First column corresponds to gene node ids and second column corresponds to time node ids.

import sys
import numpy as np

# Do not change this function signature

#data = np.loadtxt("alphaCycle.txt")
#gene_ids = np.loadtxt("alphaGenes.txt")
#gene_ids = tuple(open("alphaGenes.txt", 'r'))

#return an array of size n x m
def discretize(data):
    """ Returns discretized version of the input data"""
    n,m = np.shape(data)
    discretized_data = np.zeros((n, m))
    for r in range(0,n):
        for c in range(0,m):
            v = data[r,c]
            if v >= 0.9:
                discretized_data[r,c] = 1
            elif v <= (-0.9):
                discretized_data[r, c] = -1
            else:
                discretized_data[r, c] = 0
    return discretized_data

#discrete_data= discretize(data)

#define any helper function you need
#function for creating two matrices for activation and repression given one discretized gene expression matrix
def separate_expression(data):
    data = data.astype(int)
    n,m = data.shape
    data_activation = np.zeros((n, m))
    data_repression = np.zeros((n, m))
    for i in range(0,n):
        for j in range(0,m):
            if data[i,j] == int(1):
                data_activation[i,j] = int(1)
            elif data[i,j] == int(-1):
                data_repression[i,j] = int(-1)
    return data_activation, data_repression

#input array and a value to find, return positions of the array with the value.
def find_element_pos(array,value):
    array_pos = []
    for i in range(0,len(array)):
        if array[i] == value:
            array_pos = np.append(array_pos,i)
    return array_pos

#input an array and positions, check if each position of that array has a the corresponding value
def check_element_pos(array,array_pos,value):
    match = True
    for p in array_pos:
        if array[int(p)] != value:
            match = False
    return match

#given gene_id and array positions (active time points), return a set of edges
def get_edges(array_pos, gene_id):
    edges = np.zeros((2, 2))
    for p in array_pos:
        edges = np.concatenate((edges,[[gene_id,p]]), axis=0)
    edges = edges[2:,:]
    return edges

def calculate_edges(cluster):
    num_edges = len(cluster)
    return num_edges

#given an array of positions, find all possible subsets and store in array
def subset_pos(array_pos):
    # base case
    if len(array_pos) == 0:
        return [[]]
    # the input set is not empty, divide and conquer!
    h, t = array_pos[0], array_pos[1:]
    ss_excl_h = subset_pos(t)
    ss_incl_h = [([h] + ss) for ss in ss_excl_h]
    subset_poses = ss_incl_h + ss_excl_h
    return subset_poses

# Do not change this function signature

#Let's say we have n genes and m time nodes.
# First, for each of the genes find the subset of time nodes that they are connected to.
# Then find the set of genes that are also connected to the same set of time nodes in same way (either activated or repressed, because we want to have activated and repressed genes in separate biclusters).
# That gives a complete subgraph. Now you have to find the maximal complete subgraph, i.e, the complete subgraph that has the highest number of edges.
# To find that you need to iterate through all the genes and all the possible subset of time points


#return an array of size M x 2, M being the number of edges in the detected bicluster
def bicluster(data):
    """Takes input discrete data, returns the largest bicluster.
    The return array is of size M *2, where M is the number of edges 
    in the bicluster. Each row corresponds to an edge, first column is 
    the gene node ids and the second column is the time node ids.
    """
    n,m = data.shape #n represents the genes, m represents the time points
    #create matrices for actication and repression
    data_activation, data_repression = separate_expression(data)
    cluster_activation_max = []
    for g in range(0,n):
        array_pos = find_element_pos(data_activation[g,:], 1)
        subset_poses = subset_pos(array_pos)
        for poses in subset_poses:
            cluster_activation = np.zeros((2, 2)) #each cluster has two columns: first column represents gene node id, second column represents time node id
            #print(get_edges(array_pos, g + 1))
            cluster_activation = np.concatenate((cluster_activation, get_edges(poses, g)), axis=0) #add the edges for this target gene
            for r in range(0,n):
                if r != g:
                    if check_element_pos(data_activation[r,:], poses, 1):
                        cluster_activation = np.concatenate((cluster_activation, get_edges(poses, r)), axis=0)#add edges of genes with same set of time points activated
            cluster_activation = cluster_activation[2:,:]
            if calculate_edges(cluster_activation) > calculate_edges(cluster_activation_max):
                cluster_activation_max = cluster_activation


    cluster_repression_max = []
    for g in range(0, n):
        array_pos = find_element_pos(data_repression[g, :], -1)
        subset_poses = subset_pos(array_pos)
        for poses in subset_poses:
            cluster_repression = np.zeros((2, 2))  # each cluster has two columns: first column represents gene node id, second column represents time node id
            cluster_repression = np.concatenate((cluster_repression, get_edges(poses, g)), axis=0)  # add the edges for this target gene
            for r in range(0, n):
                if r != g:
                    if check_element_pos(data_repression[r, :], poses, -1):
                        cluster_repression = np.concatenate((cluster_repression, get_edges(poses, r)), axis=0)  # add edges of genes with same set of time points repressed
            cluster_repression = cluster_repression[2:, :]
            if calculate_edges(cluster_repression) > calculate_edges(cluster_repression_max):
                cluster_repression_max = cluster_repression

    max_cluster = []
    if calculate_edges(cluster_activation_max) > calculate_edges(cluster_repression_max):
        max_cluster = cluster_activation_max
    elif calculate_edges(cluster_activation_max) < calculate_edges(cluster_repression_max):
        max_cluster = cluster_repression_max
    max_cluster = max_cluster.astype(int)
    return max_cluster

#print(bicluster(discrete_data))
# Do not change this function signature
def bicluster_revised(data):
    """Takes input discrete data, returns the largest bicluster.
    The return array is of size M *2, where M is the number of edges 
    in the bicluster. Each row corresponds to an edge, first column is 
    the gene node ids and the second column is the time node ids.
    """

    n, m = data.shape  # n represents the genes, m represents the time points
    # create matrices for actication and repression
    data_activation, data_repression = separate_expression(data)
    cluster_activation_max = []
    for g in range(0, n):
        array_pos = find_element_pos(data_activation[g, :], 1)
        cluster_activation = np.zeros((2,2))  # each cluster has two columns: first column represents gene node id, second column represents time node id
        # print(get_edges(array_pos, g + 1))
        cluster_activation = np.concatenate((cluster_activation, get_edges(array_pos, g)),axis=0)  # add the edges for this target gene
        for r in range(0, n):
            if r != g:
                if check_element_pos(data_activation[r, :], array_pos, 1):
                    cluster_activation = np.concatenate((cluster_activation, get_edges(array_pos, r)), axis=0)  # add edges of genes with same set of time points activated
        cluster_activation = cluster_activation[2:, :]
        unique_timepoints = len(np.unique(cluster_activation[:,1]))
        if (calculate_edges(cluster_activation) > calculate_edges(cluster_activation_max)) & (unique_timepoints > 1):
            cluster_activation_max = cluster_activation

    cluster_repression_max = []
    for g in range(0, n):
        array_pos = find_element_pos(data_repression[g, :], -1)
        cluster_repression = np.zeros((2,2))  # each cluster has two columns: first column represents gene node id, second column represents time node id
        cluster_repression = np.concatenate((cluster_repression, get_edges(array_pos, g)),axis=0)  # add the edges for this target gene
        for r in range(0, n):
            if r != g:
                if check_element_pos(data_repression[r, :], array_pos, -1):
                    cluster_repression = np.concatenate((cluster_repression, get_edges(array_pos, r)), axis=0)  # add edges of genes with same set of time points repressed
        cluster_repression = cluster_repression[2:, :]
        unique_timepoints = len(np.unique(cluster_repression[:, 1]))
        if (calculate_edges(cluster_repression) > calculate_edges(cluster_repression_max)) & (unique_timepoints > 1):
            cluster_repression_max = cluster_repression

    max_cluster = []
    if calculate_edges(cluster_activation_max) > calculate_edges(cluster_repression_max):
        max_cluster = cluster_activation_max
    elif calculate_edges(cluster_activation_max) < calculate_edges(cluster_repression_max):
        max_cluster = cluster_repression_max
    max_cluster = max_cluster.astype(int)
    return max_cluster

#print(bicluster(discretize(data)))
#print(bicluster_revised(discretize(data)))

if __name__=="__main__":
    data=np.loadtxt(sys.argv[1])

    disc_data=discretize(data)
    BiClu=bicluster(disc_data)
    BiCluR=bicluster_revised(disc_data)
    

