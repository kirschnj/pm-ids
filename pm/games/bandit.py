import numpy as np
from pm.game import Game


# class Bandit(Game):
#
#     def __init__(self, X, id=None):
#         super().__init__(X.shape[0], X.shape[1], 1)
#         self._X = X
#         self._M = X.reshape(self.k, 1, self.d)
#         self._id = id
#
#     def get_actions(self):
#         return self._X
#
#     def get_observation_maps(self):
#         return self._M
#
#     def __str__(self):
#         if self._id is not None:
#             return self._id
#         return type(self).__name__


# TODO : another experiment idea
# class BanditMultiContext(Game):
#     def __init__(self, X, nb_contexts ,id=""):
#         self._X = X # nb_contexts * nb_actions
#         self._id = id
#         self._d = X.shape[1]
#         self._I = np.arange(len(X))
#         self._nb_contexts = nb_contexts
#
#     def get_d(self):
#         return self._d
#
#     def get_indices(self):
#         return self._I[]
#
#     def get_actions(self, indices):
#         return self._X[indices]
#
#     def get_observation_maps(self, indices):
#         return self.get_actions(indices).reshape(len(indices), 1, self._d)
#
#     # this looks a lot like pm.utils.difference_matrix
#     def get_cell_constraints(self, index):
#         C = np.zeros((len(self._I),self._d))
#         for i in self._I:
#             if i != index:
#                 C[i,:] = self._X[i,:] - self._X[index,:]
#
#         return C
#
#     def id(self):
#         return self._id
