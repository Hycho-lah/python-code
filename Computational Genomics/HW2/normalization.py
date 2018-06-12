# normalization.py
# HW2, Computational Genomics, Fall 2018
# andrewid: eye

# Code contains functions of three normalization techniques: RPKM, TPM, and size factor

import sys
import numpy as np

#test_counts = np.loadtxt("ReadCounts.txt")
#test_lengths = np.loadtxt("GeneLengths.txt")
# Do not change this function signature
#raw_counts would be a numpy.ndarray with 13918 rows and 49 columns --> n = 13918 genes, m = 49 samples
#gene_lengths would be a numpy.ndarray with 13918 rows


# RPKM(G) = R/L*10^9/M
def rpkm(raw_counts,gene_lengths):
    """Find the normalized counts for raw_counts
    Returns: a matrix of same size as raw_counts
    """
    n,m = np.shape(raw_counts)
    RPM = np.zeros((n, m))
    output = np.zeros((n, m))
    #iterate through row (or genes) of raw_counts
    for s in range (0,m):
        M = sum(raw_counts[:, s])/1000000 # per million scaling factor
        RPM[:, s] = raw_counts[:, s]/M
        for g in range(0,n):
            L = gene_lengths[g]
            rpkm_val = (RPM[g,s]/L)*1000
            output[g,s] = rpkm_val
    return output


# Do not change this function signature
def tpm(raw_counts,gene_lengths):
    """Find the normalized counts for raw_counts
    Returns: a matrix of same size as raw_counts
    """
    n, m = np.shape(raw_counts)
    RPK = np.zeros((n, m))
    output = np.zeros((n, m))
    # iterate through row (or genes) of raw_counts
    #Step 1 divide the read counts for each gene by its length (in kilobases). This results in a matrix of RPK values
    for g in range(0, n):
        M = len(raw_counts[g, :])  # usually 49
        L = gene_lengths[g]
        RPK[g,:] = (raw_counts[g, :]/L)*1000
    #Step 2 count up all RPK values in a sample and divide this number by 1,000,000.
    # this is the per million scaling factor (M) for that sample set.
    for s in range(0,m):
        M = sum(RPK[:,s])/1000000
        #Step 3 divide each RPK value by scaling factor
        for g in range(0,n):
            output[g,s] = RPK[g,s]/M
    return output


# define any helper function here

#calculate geometric mean of an array
def geometric_mean(gene_counts):
    m = len(gene_counts)
    g=1
    for v in range(0,m):
        g = g*gene_counts[v]
    geomean = g**(1/m)
    return geomean

# Do not change this function signature


def size_factor(raw_counts):
    """Find the normalized counts for raw_counts
    Returns: a matrix of same size as raw_counts
    """
    n, m = np.shape(raw_counts)
    output = np.zeros((n, m))
    for s in range(0,m):
        sample_counts = raw_counts[:,s] # obtain an array of the read counts for a sample
        ratio_counts = []
        for g in range(0,n):
            gi = geometric_mean(raw_counts[g,:])#geometric mean for each gene
            if gi != 0:
                k = sample_counts[g]/gi
                #ratio_counts[g] = k
                ratio_counts = np.append(ratio_counts,k)
        sf = np.median(ratio_counts)
        output[:,s] = raw_counts[:,s]/sf
    return output

if __name__=="__main__":
    raw_counts=np.loadtxt(sys.argv[1])
    gene_lengths=np.loadtxt(sys.argv[2])

    rpkm1=rpkm(raw_counts,gene_lengths)
    tpm1=tpm(raw_counts,gene_lengths)
    size_factor1=size_factor(raw_counts)
    
    
