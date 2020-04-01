# cython: language_level=3
# -*- coding: utf-8 -*-

import scipy.optimize as opt
from anytree import Node

import numpy as np
cimport numpy as np
np.import_array()

# Datatype to be used in numpy arrays
DTYPE = np.float64

# Assign corresponding compile-time type
ctypedef np.float64_t DTYPE_t

def _splitting_loss(double thres, np.ndarray[DTYPE_t, ndim = 2] X, 
                    np.ndarray[DTYPE_t] y, int feat):
    ''' Calculate the mean squared loss of a split. '''

    # Get the number of rows
    cdef Py_ssize_t nrows = X.shape[0]

    # Get the target values of the left and right split
    cdef np.ndarray[DTYPE_t] yleft
    cdef np.ndarray[DTYPE_t] yright

    cdef float left_val = 0.
    cdef float right_val = 0.
    cdef float yleft_mean
    cdef float yright_mean
    cdef int i

    yleft = np.array([y[i] for i in range(nrows) if X[i, feat] <= thres],
                     dtype = DTYPE)
    yright = np.array([y[i] for i in range(nrows) if X[i, feat] > thres],
                      dtype = DTYPE)

    # If all feature values are below or above the threshold
    # then output infinite loss
    if yleft.size == 0 or yright.size == 0: return float('inf')

    # Calculate the means of yleft and yright
    yleft_mean = sum(yleft) / yleft.size
    yright_mean = sum(yright) / yright.size

    # Calculate mean squared loss on both child nodes, so that
    # minimising these corresponds to ensuring that the target
    # values within each child are as similar as possible
    for i in range(nrows):
        left_val += (yleft[i] - yleft_mean) ** 2
        right_val += (yright[i] - yright_mean) ** 2
    left_val /= yleft.size
    right_val /= yright.size

    return left_val + right_val

def _find_threshold(np.ndarray[DTYPE_t, ndim = 2] X, np.ndarray[DTYPE_t] y, 
                    int feat_idx):
    ''' Given a feature, find the optimal split threshold for it. '''

    cdef Py_ssize_t nrows = X.shape[0]
    cdef double initial_guess
    cdef double threshold
    cdef double loss

    # Initial guess for the optimal threshold, which is required by
    # `scipy.opt.minimize`
    initial_guess = sum(X[:, feat_idx]) / nrows

    # Find the threshold that minimises the splitting loss
    result = opt.minimize(_splitting_loss, x0 = initial_guess, 
                          args = (X, y, feat_idx))

    # Get the resulting threshold and the associated loss, the latter
    # of which is used to compare thresholds across multiple features
    threshold = result.x[0]
    loss = result.fun

    return [feat_idx, threshold, loss]

def _branch(np.ndarray[DTYPE_t, ndim = 2] X, np.ndarray[DTYPE_t] y, 
            int min_samples_leaf, int pos = -1, parent = None):
    ''' Recursive function that computes the next two child nodes. '''

    # Get the number of rows and features in the data set
    cdef Py_ssize_t nrows = X.shape[0]
    cdef Py_ssize_t nfeats = X.shape[1]

    cdef np.ndarray[DTYPE_t, ndim = 2] X0
    cdef np.ndarray[DTYPE_t, ndim = 2] X1
    cdef np.ndarray[DTYPE_t] y0
    cdef np.ndarray[DTYPE_t] y1

    cdef np.ndarray[DTYPE_t, ndim = 2] result_array

    cdef int feat
    cdef double thres
    cdef double loss

    cdef int i
    cdef int min_loss_idx

    cdef str name

    # Compute the best thresholds for all the features
    result_array = np.array([_find_threshold(X, y, i) for i in range(nfeats)],
                            dtype = DTYPE)

    # Pull out the feature and threshold with the smallest loss
    loss = min(result_array[:, 2])
    for i in range(nfeats):
        if result_array[i, 2] == loss:
            feat = int(result_array[i, 0])
            thres = result_array[i, 1]

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
    if len(set(y0)) < min_samples_leaf or len(set(y1)) < min_samples_leaf:

        name = (f'[{min(y):,.0f}; {max(y):,.0f}]\n'
                f'n = {nrows}\n'
                f'n_unique = {len(set(y))}')
        node = Node(name, n = nrows, parent = parent, vals = y, pos = pos)

    else:
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

    # Return the node if we're at the root
    if parent is None:
        return node

def _predict_one(root, np.ndarray[DTYPE_t] x, double quantile = -1.):
    ''' Perform a prediction for a single input. '''

    cdef int node_feat
    cdef int num_node_vals
    cdef int node_val_idx
    cdef double node_thres
    cdef np.ndarray[DTYPE_t] node_vals

    node = root
    while not node.is_leaf:
        node_feat = node.feat
        node_thres = node.thres
        left, right = sorted(node.children, key = lambda x: x.pos)
        node = left if x[node_feat] <= node_thres else right

    node_vals = node.vals
    if quantile == -1:
        return sum(node_vals) / len(node_vals)
    else:
        # Compute quantile
        node_vals = sorted(node_vals)
        num_node_vals = len(node_vals)
        if num_node_vals % 2 == 0:
            node_val_idx = int(num_node_vals * quantile)
            return (node_vals[node_val_idx] + node_vals[node_val_idx + 1]) / 2
        else:
            node_val_idx = int(num_node_vals * quantile)
            return node_vals[node_val_idx]

def _predict(root, np.ndarray[DTYPE_t, ndim = 2] X, double quantile = -1.):
    ''' Predict the response values of a given feature matrix. '''

    cdef Py_ssize_t nrows
    cdef int onedim
    cdef int i
    cdef np.ndarray[DTYPE_t] result

    nrows = X.shape[0]
    result = np.array(
        [_predict_one(root, X[i, :], quantile) for i in range(nrows)],
        dtype = DTYPE
    )

    return result
