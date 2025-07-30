import sys
from numpy import array, loadtxt, argsort
from numpy.linalg import eig
import numpy

def covariance_matrix(X, bias=False):
	"""
	Computes covariance matrix.

	Parameters
	----------
	X : an array of size (n, k)
		input data
	bias: bool, optional
		If True, then the normalization should be n, otherwise it is n - 1.
		Default value is False.

	Returns
	-------
	C : an array of size (k, k)
		covariance matrix
	"""

	# place your code here
	if bias:
		return 1 / (X.shape[0]) * (X - numpy.mean(X, axis=0)).T.dot(X - numpy.mean(X, axis=0))
	else: 
		return 1 / (X.shape[0] - 1) * (X - numpy.mean(X, axis=0)).T.dot(X - numpy.mean(X, axis=0))


def pca(X):
	"""
	Computes PCA with 2 components

	Parameters
	----------
	X : an array of size (n, k)
		input data

	Returns
	-------
	v1 : an array of size n
		ith element = first principal component of the ith data point
	v2 : an array of size n
		ith element = second principal component of the ith data point
	"""

	v1 = None
	v2 = None

	# place your code here
	cm = covariance_matrix(X)
	eigvals, eigvecs = eig(cm)
	sorted_indices = argsort(eigvals)[::-1]
	eigvecs = eigvecs[:, sorted_indices]
	v1 = X @ eigvecs[:, 0]
	v2 = X @ eigvecs[:, 1]
	return v1, v2


def main(argv):
	X = loadtxt(argv[1])
	print(covariance_matrix(X))
	print(pca(X))



# This allows the script to be used as a module and as a standalone program
if __name__ == "__main__": 
	if len(sys.argv) != 2:
		print('usage: python %s filename' % sys.argv[0])
	else:
		main(sys.argv)
