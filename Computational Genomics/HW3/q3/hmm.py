#When DNA undergoes replication in the S phase of the cell cycle, the replication does not start at the same time for all regions along the DNA. Techniques have been developed to reveal the replication timing of DNA through a microarray assay. It is believed that the patterns in replication timing that these assays reveal could be correlated with gene expression, the 3D organization of the chromosome structure, as well as evolution of DNA sequences.
# This code is for modelling replication timing data using Hidden Markov Model (HMM)
# Model the DNA replication timing by a 2-state HMM. In this HMM, one state indicates that the replication occurs early, while the other state indicates that it occurs later. In the given data, higher replication timing signal indicates earlier replication.

import time
import csv
import numpy as np
import warnings
import math
from hmmlearn import hmm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# There are annoying sklearn DeprecationWarnings, we silence them here.
def warn(*args, **kwargs):
    pass
warnings.warn = warn


def get_fitted_hmm(X, n_components):
	""" Fit a Gaussian Hidden Markov Model to the data X. Use "full" covariance, and n_iter=100.
	Returns the fitted model.
	"""
	HMM = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100)
	print(HMM)
	HMM.fit(X)
	model = HMM
	return model

def plot_predictions(positions, replication_timing_data, predicted_states, path):
	""" Plots both the replication timing signal and the predicted states on the same plot
	positions - numpy array of chromosome positions
	replication_timing_data - the RT signal for every position
	predicted_states - the predicted state for every position
	path - file path to save plot (must end with '.png')
	"""
	plt.figure()
	plt.scatter(positions, replication_timing_data, c='b', s=0.5)
	state_colors = ['orange', 'magenta']
	cm = LinearSegmentedColormap.from_list('orange_magenta', state_colors, N=2)
	orange_patch = mpatches.Patch(color='orange', label='State=0')
	magenta_patch = mpatches.Patch(color='magenta', label='State=1')
	blue_patch = mpatches.Patch(color='blue', label='RT value')
	plt.scatter(positions, predicted_states, c=predicted_states, cmap=cm, s=0.5)
	plt.title("Replication timing signal and predicted states vs chromosome position")
	plt.xlabel("Chromosome position")
	plt.legend(handles=[blue_patch, orange_patch, magenta_patch])
	plt.savefig(path)


def main():
	X1 = np.load('chrom2_koren_et_al.npy')
	X2 = np.load('chrom2_ryba_et_al.npy')

	RT_signals = X1[:,1]
	#S = np.zeros((len(RT_signals), 2))
	S = RT_signals.reshape(-1, 1)
	model = get_fitted_hmm(S, 2)
	predicted_states = model.predict(S)
	print(model.monitor_)
	print(model.monitor_.converged)
	plot_predictions(X1[:,0], RT_signals, predicted_states, 'HMM_koren.png')

	positions = np.zeros(len(X2[:, 0]))
	for i in range(0,len(X2[:, 0])):
		positions[i] = math.ceil((X2[i, 1]-X2[i, 0])/2.00) + X2[i, 0]
	RT_signals = X2[:, 2]
	S = RT_signals.reshape(-1, 1)
	model = get_fitted_hmm(S, 2)
	predicted_states = model.predict(S)
	print(model.monitor_)
	print(model.monitor_.converged)
	plot_predictions(positions, RT_signals, predicted_states, 'HMM_ryba.png')

if __name__ == "__main__":
	main()
