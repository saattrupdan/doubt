''' Build quantile regression trees. '''

from ..data_structures import Stack, Queue

import scipy.optimize as opt
import numpy as np
import logging

logging.basicConfig(format = '%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def mean_squared_impurity(thres: float, x, y):
    ''' 
    Compute the mean of the variance of the child nodes given by the
    given threshold.
    '''
    left = y[x < thres]
    right = y[x >= thres]

    left_impurity = np.sum((left - left.mean()) ** 2) if left.size else 0.
    right_impurity = np.sum((right - right.mean()) ** 2) if right.size else 0.

    return left_impurity + right_impurity

class Node(object):
    ''' A node in a tree, also storing all its descendants. '''

    def __init__(self, mask, parent = None):
        self.mask = mask
        self.parent = parent

        self.left_child: Node
        self.right_child: Node

        self.impurity = float('inf') if parent is None else parent.impurity
        self.feat: int
        self.thres: float
        self.vals: np.ndarray

        # Every node has to prove that it's not a leaf
        self.is_leaf: bool = True 

    def compute_stats(self, X, y, min_samples_leaf: int):
        ''' 
        Compute the impurity and best feature and threshold to use
        for a split at the node, if it results in an impurity decrease.
        These are stored internally in the node.
        '''

        # Mask the rows in the input to reflect the samples remaining
        # at this node
        X = X[self.mask, :]
        y = y[self.mask, :]

        # Compute the impurity for the node
        self.impurity = np.sum((y - y.mean()) ** 2)

        # Loop over all the features to check for splits
        for feat in range(X.shape[1]):
            x = X[:, feat]

            # Define the search space for the threshold.
            # If we cannot enforce there to be at least ``min_samples_leaf``
            # many samples in each child node then skip this feature
            try:
                lower_bound = np.partition(x, min_samples_leaf)
                lower_bound = lower_bound[min_samples_leaf]
                upper_bound = np.partition(x, -min_samples_leaf)
                upper_bound = upper_bound[-min_samples_leaf]
            except ValueError:
                continue

            # If the x only has a single unique value then we cannot
            # split the node on that feature and we continue
            if lower_bound >= upper_bound: continue

            # Find the best threshold to split x
            result = opt.minimize(
                mean_squared_impurity, 
                x0 = lower_bound, 
                args = (x, y), 
                bounds = ((lower_bound, upper_bound),)
            )

            # Pull out the best threshold and corresponding impurity, and
            # if this impurity is an improvement then store the associated data
            thres = result.x[0]
            child_impurities = result.fun
            if child_impurities < self.impurity:
                self.is_leaf = False
                self.feat = feat
                self.thres = thres

        return self

    def get_all_descendants(self):
        ''' Fetch all the descendants of the node from the tree. '''
        stack = Stack([self])
        nodes = []
        while True:
            try:
                node = stack.pop()
                nodes.append(node)
            except IndexError:
                break
            if not node.is_leaf:
                stack.push(node.left_child)
                stack.push(node.right_child)
        return nodes

class Tree(object):
    ''' A collection of nodes organised in a tree structure. '''

    def __init__(self, root: Node):
        self.root = root
        self.nodes = root.get_all_descendants()

    def size(self):
        return len(self.nodes)

    def __len__(self):
        stack = Stack([(0, self.root)])
        max_depth = 0
        while True:
            try:
                depth, node = stack.pop()
            except IndexError:
                break

            if depth > max_depth: max_depth = depth
            if node.is_leaf: continue
            stack.push((depth + 1, node.left_child))
            stack.push((depth + 1, node.right_child))

        return max_depth

    def find_leaf(self, x):
        ''' Traverse the tree and find the leaf to which ``x`` belongs. '''
        node = self.root
        while True:
            if node.is_leaf:
                return node
            elif x[node.feat] < node.thres:
                node = node.left_child
            else:
                node = node.right_child
        return node

class TreeBuilder(object):
    ''' Builds quantile regression trees in a depth-first fashion. '''

    def __init__(self, min_samples_leaf: int = 5):
        self.min_samples_leaf = min_samples_leaf

    def build(self, X, y):
        root = Node(X[:, 0] == X[:, 0])
        queue = Queue([root])

        n = 0
        while True:
            try:
                node = queue.pop()
            except IndexError:
                break

            node.compute_stats(X, y, min_samples_leaf = self.min_samples_leaf)
            if not node.is_leaf:
                left_mask = node.mask & (X[:, node.feat] < node.thres)
                right_mask = node.mask & (X[:, node.feat] >= node.thres)

                left_child = Node(left_mask, parent = node)
                right_child = Node(right_mask, parent = node)

                node.left_child = left_child
                node.right_child = right_child

                queue.push(left_child)
                queue.push(right_child)

            else:
                node.is_leaf = True
                node.vals = y[node.mask]

        return Tree(root)
