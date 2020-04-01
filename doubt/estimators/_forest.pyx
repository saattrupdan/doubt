# cython: language_level=3
# -*- coding: utf-8 -*-

from anytree import Node
import scipy.optimize as opt
import numpy as np
from joblib import Parallel, delayed

DTYPE = np.float64

def _splitting_loss(double thres, double[:, :] X, double[:] y, int feat):
    ''' Calculate the mean squared loss of a split. '''

    # Get the number of rows
    cdef Py_ssize_t nrows = X.shape[0]
    cdef int i

    # Get the target values of the left and right split
    cdef double[:] yleft = np.array(
        [y[i] for i in range(nrows) if X[i, feat] <= thres],
        dtype = DTYPE
    )
    cdef double[:] yright = np.array(
        [y[i] for i in range(nrows) if X[i, feat] > thres],
        dtype = DTYPE
    )

    cdef float left_val
    cdef float right_val

    # If all feature values are below or above the threshold
    # then output infinite loss
    if yleft.size == 0 or yright.size == 0: return float('inf')

    # Calculate mean squared loss on both child nodes, so that
    # minimising these corresponds to ensuring that the target
    # values within each child are as similar as possible
    left_val = np.mean(np.square(yleft - np.mean(yleft)))
    right_val = np.mean(np.square(yright - np.mean(yright)))

    return left_val + right_val

def _find_threshold(double[:, :] X, double[:] y, int feat_idx):
    ''' Given a feature, find the optimal split threshold for it. '''

    cdef Py_ssize_t nrows = X.shape[0]
    cdef double initial_guess
    cdef double threshold
    cdef double loss

    # Initial guess for the optimal threshold, which is required by
    # `scipy.opt.minimize`
    initial_guess = np.mean(X[:, feat_idx])

    # Find the threshold that minimises the splitting loss
    result = opt.minimize(_splitting_loss, x0 = initial_guess, 
                          args = (X, y, feat_idx))

    # Get the resulting threshold and the associated loss, the latter
    # of which is used to compare thresholds across multiple features
    threshold = result.x[0]
    loss = result.fun

    return [feat_idx, threshold, loss]

def _branch(double[:, :] X, double[:] y, int min_samples_leaf,
            int pos = -1, parent = None):
    ''' Recursive function that computes the next two child nodes. '''

    # Get the number of rows and features in the data set
    cdef Py_ssize_t nrows = X.shape[0]
    cdef Py_ssize_t nfeats = X.shape[1]

    cdef double[:, :] X0
    cdef double[:, :] X1
    cdef double[:] y0
    cdef double[:] y1

    cdef double[:, :] result_array
    cdef double[:] proj_result_array

    cdef int feat
    cdef double thres

    cdef int i
    cdef int min_loss_idx

    cdef str name

    # Compute the best thresholds for all the features
    result_array = np.array([_find_threshold(X, y, i) for i in range(nfeats)],
                            dtype = DTYPE)

    # Pull out the feature and threshold with the smallest loss
    min_loss_idx = np.argmin(result_array[:, 2])
    proj_result_array = result_array[min_loss_idx]
    feat = int(proj_result_array[0])
    thres = proj_result_array[1]

    # Get the target values of the left and right split
    y0 = np.array([y[i] for i in range(nrows) if X[i, feat] <= thres],
                     dtype = DTYPE)
    y1 = np.array([y[i] for i in range(nrows) if X[i, feat] > thres],
                     dtype = DTYPE)

    # Get the feature values of the left and right split
    X0 = np.array([X[i, :] for i in range(nrows) if X[i, feat] <= thres],
                     dtype = DTYPE)
    X1 = np.array([X[i, :] for i in range(nrows) if X[i, feat] > thres],
                     dtype = DTYPE)

    # If we have reached a leaf node then store information about
    # the target values and stop the recursion
    if len(np.unique(y0)) < min_samples_leaf or \
        len(np.unique(y1)) < min_samples_leaf:

        name = (f'[{np.min(y):,.0f}; {np.max(y):,.0f}]\n'
                f'n = {nrows}\n'
                f'n_unique = {len(np.unique(y))}')
        Node(name, n = nrows, parent = parent, vals = y, pos = pos)
        return 0

    # Define the current node, which by the above conditional can't
    # be a leaf node
    name = f'Is feature {feat} < {thres:.2f}?'
    node = Node(name, n = nrows, parent = parent, feat = feat,
                thres = thres, pos = pos)

    # Continue the recursion on the child nodes
    _branch(X0, y0, pos = 0, parent = node, 
            min_samples_leaf = min_samples_leaf)
    _branch(X1, y1, pos = 1, parent = node, 
            min_samples_leaf = min_samples_leaf)

    # Return and recursive call, and also return the node if we're at the root
    if parent is None:
        return node

def _predict_one(root, double[:] x, double quantile = -1.):
    ''' Perform a prediction for a single input. '''

    cdef int node_feat
    cdef double node_thres
    cdef double[:] node_vals

    node = root
    while not node.is_leaf:
        node_feat = node.feat
        node_thres = node.thres
        left, right = sorted(node.children, key = lambda x: x.pos)
        node = left if x[node_feat] <= node_thres else right

    node_vals = node.vals
    if quantile == -1:
        return np.mean(node_vals)
    else:
        return np.quantile(node_vals, quantile)

def _predict(root, double[:, :] X, double quantile = -1.):
    ''' Predict the response values of a given feature matrix. '''

    cdef int onedim = (len(X.shape) == 1)
    cdef double[:] result
    cdef double[:] x

    if onedim == 1: X = np.expand_dims(X, 0)

    result = np.array([_predict_one(root, x, quantile) for x in X],
                      dtype = DTYPE)

    if onedim == 1:
        return result[0] 
    else:
        return result
