# Pipeine for classification of RNA-seq data from cancer tumors

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import svm
#import sklearn.metrics.auc


def load_data():
	X = np.load('X.npy')
	y = np.load('y.npy')
	return X, y

def binarize(y, label='COAD'):
	""" Convert the multi-labeled y into just binary labels of 1 if
	the sample is labeled as label, and 0 otherwise.
	Note the returned vector should have a data type of int, not string
	"""
	binarized = []
	print("binarizing labels")
	for i in range(0,len(y)):
		if y[i] == label:
			binarized.append(1)
		else:
			binarized.append(0)
	return binarized

def preprocess(X):
	""" Apply preprocessing steps to the data
	Such as:
		Whitening
		L2 normalization
		etc
	(We won't use this in this assignment)
	"""
	return X

def reduce(X, dims=2):
	""" Reduce the dimensions of X down to dims using PCA
	X has shape (n, d)
	Returns: The reduced X of shape (n, dims)
			 The fitted PCA model used to reduce X
	"""
	print("reducing dimensions using PCA")
	pca = PCA(n_components = 2)
	X_reduced = pca.fit_transform(X)  # rows are PC
	return X_reduced, pca

def get_classifier(X, y):
	print("training classifier")
	n, m = X.shape  # number of rows(801) and columns(20531)

	# Create linear regression object
	regr = LogisticRegression()
	regr.fit(X, y)
	classifier = regr
	return classifier

def get_classifier2(X, y):
	print("training classifier")

	SVM = svm.SVC(kernel='rbf', C=0.00001,gamma=3,probability=True) #gamma Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
	#SVM = svm.SVC(kernel='poly'probability=True)
	SVM.fit(X, y)
	classifier = SVM
	return classifier

# metrics.auc(fpr, tpr)
#You can use any sklearn classier (LogisticRegression, SVC, etc).
# You cannot use any sklearn metrics (accuracy, f1 score, precision, recall, sensitivity, etc).
# You must write your own code to calculate things like false positive rate and f-score
def plot_ROC_curve(y_true, y_score, path):
	""" Plot and save the ROC curve by varying a threshold on y_score.
	Assume that the positive class is labeled truly as 1, and that if given a threshold value,
	we can classify the examples with: (y_score > threshold).astype(int)
	Return:
		The area under your curve (AUROC)
		The threshold value to use that would give you the highest F-score. Remember that each
		  point in an ROC curve has an associated F-score
	"""

	#create list of thresholds
	#thresholds = y_score
	#thresholds = [thresholds[0]-1] + thresholds + [thresholds[-1]+1]

	thresholds = np.arange(min(y_score)-1, max(y_score)+1, 0.0001)
	TPR_list = []
	FPR_list = []
	f_score_list = []
	for thr in thresholds:
		#convert y_score to y_pred
		y_pred = [[0, 1][x > thr] for x in y_score]# returns list = [[FALSE, TRUE][IF CONDITION] FOR a row in LIST]

		n11_TP = sum([[0, 1][y_pred[i] == 1 and y_true[i] == 1] for i in range(len(y_pred))])
		n00_TN = sum([[0, 1][y_pred[i] == 0 and y_true[i] == 0] for i in range(len(y_pred))])
		n10_FN = sum([[0, 1][y_pred[i] == 0 and y_true[i] == 1] for i in range(len(y_pred))])
		n01_FP = sum([[0, 1][y_pred[i] == 1 and y_true[i] == 0] for i in range(len(y_pred))])

		tpr = n11_TP / (n11_TP + n10_FN)  # TP rate = TP/(TP+FN)
		fpr = n01_FP / (n00_TN + n01_FP)  # FP rate = FP/(TN+FP)
		if (n11_TP == 0) & (n01_FP ==0):
			precision = 0
		else:
			precision = n11_TP / (n11_TP + n01_FP)  # precision = TP/(TP+FP)
		recall = tpr
		if (precision == 0) & (recall ==0):
			f_score = 0
		else:
			f_score = 2 * (precision * recall) / (precision + recall)

		TPR_list.append(tpr)
		FPR_list.append(fpr)
		f_score_list.append(f_score)

	fig, ax = plt.subplots()

	sorted_pairs = sorted((i, j) for i, j in zip(FPR_list, TPR_list))
	new_FPR_list, new_TPR_list = zip(*sorted_pairs)

	# Plot data
	ax.plot(new_FPR_list, new_TPR_list)
	ax.set_xlabel("FPR")
	ax.set_ylabel("TPR")
	plt.savefig(path)

	#calculate AUROC
	#auc = metrics.auc(FPR_list, TPR_list, reorder = True)
	auc = metrics.auc(new_FPR_list, new_TPR_list)
	#find f_score_optimal_thr
	pos_optimal_f_score = f_score_list.index(max(f_score_list))
	f_score_optimal_thr = thresholds[pos_optimal_f_score]

	return auc, f_score_optimal_thr

def plot_decision_boundary(clf, X, y, path, thr=0.5):
	""" Plot the two dimensional data X (y is the labels and is used to color the points),
	along with the decision boundary of the classifier clf, using the threshold of trh.
	Save the plot to path.
	"""
	fig, ax = plt.subplots()
	# Plot data
	ax.scatter(X[:,0], X[:,1], c=y)
	ax.set_xlabel("X1")
	ax.set_ylabel("X2")
	# Plot decision boundary
	x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
	y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
	x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
	grid = np.c_[x_mesh.ravel(), y_mesh.ravel()]
	z = clf.predict_proba(grid)[:, 1].reshape(x_mesh.shape) #note this obtains the second column, which represents prob of being 1
	ax.contour(x_mesh, y_mesh, z, levels=[-1,thr,1], colors='k')
	plt.savefig(path)

def main():
	# Your final submitted code should contain all the code to produce the figures and metrics
	# in your report (i.e., I should be able to run "python clf.py" and it will save all the plots,
	# and also print out things like the AUC values we ask for (no strict formatting rules))

	# All of the below is an example, just to show you how some of the code works. You might
	# not need all of it and you can erase/modify/add to it as you need to answer the questions.

	X, y = load_data()
	# X = preprocess(X)
	# Binarizing data for Colon cancer
	y_bin = binarize(y, label='COAD')
	# In case you want to plot with all the labels, not just binary labels
	# You can use y_multi below, it encodes all 5 labels as integers
	_, y_multi = np.unique(y, return_inverse=True)
	X, _ = reduce(X) # we don't really need the fitted pca model (2nd return value) here
	# Example call to train LR for binary classification:
	clf = get_classifier(X, y_bin)
	plot_decision_boundary(clf, X, y_bin, 'DB_0.2.png', thr=0.2)
	plot_decision_boundary(clf, X, y_bin, 'DB_0.5.png', thr=0.5)
	plot_decision_boundary(clf, X, y_bin, 'DB_0.9.png', thr=0.9)
	# Or to plot with different colors for all labels:
	# plot_decision_boundary(clf, X, y_multi, 'example.png', thr=0.5)
	y_score = clf.predict_proba(X)[:,1]
	auc, f_score_optimal_thr = plot_ROC_curve(y_bin,y_score,'ROC_curve.png')
	print('auc=', auc)
	print('F-score optimal threshold=',f_score_optimal_thr)

	y_bin_LUAD = binarize(y, label='LUAD')
	clf2 = get_classifier2(X, y_bin_LUAD)
	y_score_LUAD = clf2.predict_proba(X)[:, 1]
	auc_LUAD, f_score_optimal_thr_LUAD = plot_ROC_curve(y_bin_LUAD, y_score_LUAD, 'ROC_curve_LUAD.png')
	print('auc=', auc_LUAD)
	print('F-score optimal threshold=', f_score_optimal_thr_LUAD)

if __name__ == "__main__":
	main()
