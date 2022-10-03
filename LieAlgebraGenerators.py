# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:27:45 2022

@author: Holik
"""

# %%

from functools import reduce

import numpy as np

import math as m

from scipy import linalg as la

from scipy.linalg import logm, expm

from scipy import log, log2

import funciones_SP as f

import itertools

import sympy

from itertools import permutations

import itertools

# %%

N = 3 # Número de qubits

Tipo = "Permutational" # Permutational, Rotational, Werner

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1.0j], [1.0j, 0]])
Z = np.array([[1, 0], [0, -1]])

# %%
# -------------------------------------------------------------
# --------Generadores de simetría permutacional.---------------
# -------------------------------------------------------------

Vect1 = range(0, N)

# Permut = list(permutations(Vect))

GenPerm10 = [
    (1, 0, 2, 3, 4, 5, 6, 7, 8, 9),
    (0, 2, 1, 3, 4, 5, 6, 7, 8, 9),
    (0, 1, 3, 2, 4, 5, 6, 7, 8, 9),
    (0, 1, 2, 4, 3, 5, 6, 7, 8, 9),
    (0, 1, 2, 3, 5, 4, 6, 7, 8, 9),
    (0, 1, 2, 3, 4, 6, 5, 7, 8, 9),
    (0, 1, 2, 3, 4, 5, 7, 6, 8, 9),
    (0, 1, 2, 3, 4, 5, 6, 8, 7, 9),
    (0, 1, 2, 3, 4, 5, 6, 7, 9, 8),
    (9, 1, 2, 3, 4, 5, 6, 7, 8, 0),
]

GenPerm9 = [
    (1, 0, 2, 3, 4, 5, 6, 7, 8),
    (0, 2, 1, 3, 4, 5, 6, 7, 8),
    (0, 1, 3, 2, 4, 5, 6, 7, 8),
    (0, 1, 2, 4, 3, 5, 6, 7, 8),
    (0, 1, 2, 3, 5, 4, 6, 7, 8),
    (0, 1, 2, 3, 4, 6, 5, 7, 8),
    (0, 1, 2, 3, 4, 5, 7, 6, 8),
    (0, 1, 2, 3, 4, 5, 6, 8, 7),
    (8, 1, 2, 3, 4, 5, 6, 7, 0),
]

GenPerm5 = [
    (1, 0, 2, 3, 4),
    (0, 2, 1, 3, 4),
    (0, 1, 3, 2, 4),
    (0, 1, 2, 4, 3),
    (4, 1, 2, 3, 0),
]

GenPerm6 = [
    (1, 0, 2, 3, 4, 5),
    (0, 2, 1, 3, 4, 5),
    (0, 1, 3, 2, 4, 5),
    (0, 1, 2, 4, 3, 5),
    (0, 1, 2, 3, 5, 4),
    (5, 1, 2, 3, 4, 0),
]

GenPerm7 = [
    (1, 0, 2, 3, 4, 5, 6),
    (0, 2, 1, 3, 4, 5, 6),
    (0, 1, 3, 2, 4, 5, 6),
    (0, 1, 2, 4, 3, 5, 6),
    (0, 1, 2, 3, 5, 4, 6),
    (0, 1, 2, 3, 4, 6, 5),
    (6, 1, 2, 3, 4, 5, 0),
]

GenPerm8 = [
    (1, 0, 2, 3, 4, 5, 6, 7),
    (0, 2, 1, 3, 4, 5, 6, 7),
    (0, 1, 3, 2, 4, 5, 6, 7),
    (0, 1, 2, 4, 3, 5, 6, 7),
    (0, 1, 2, 3, 5, 4, 6, 7),
    (0, 1, 2, 3, 4, 6, 5, 7),
    (0, 1, 2, 3, 4, 5, 7, 6),
    (7, 1, 2, 3, 4, 5, 6, 0),
]

GenPerm4 = [(1, 0, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2), (3, 1, 2, 0)]

GenPerm3 = [(1, 0, 2), (0, 2, 1), (2, 1, 0)]

GenPerm2 = [(0, 1), (1, 0)]

string = "GenPerm" + str(int(N))

GenPerm = eval(string)

# Permions = list( f.P(Permut[i],N) for i in range(len(Permut))  )

PermionsRed = []

for i in range(len(GenPerm)):
    A = f.P(GenPerm[i], N)
    PermionsRed.append(A)
    
globals()[ "Lie_Algebra_Generators_Permutational_" +str(N) ] = PermionsRed  
   

# %%
# -------------------------------------------------------------
# --------Generadores de Werner.-------------------------------
# -------------------------------------------------------------

Lie_Algebra_Generators_Werner_2 = [
    np.kron(X, I) + np.kron(I, X),
    np.kron(Y, I) + np.kron(I, Y),
    np.kron(Z, I) + np.kron(I, Z)
]

Lie_Algebra_Generators_Werner_3 = [
    f.KroneckerMultiplyList([X, I, I])
    + f.KroneckerMultiplyList([I, X, I])
    + f.KroneckerMultiplyList([I, I, X]),
    f.KroneckerMultiplyList([Y, I, I])
    + f.KroneckerMultiplyList([I, Y, I])
    + f.KroneckerMultiplyList([I, I, Y]),
    f.KroneckerMultiplyList([Z, I, I])
    + f.KroneckerMultiplyList([I, Z, I])
    + f.KroneckerMultiplyList([I, I, Z])
]

Lie_Algebra_Generators_Werner_4 = [
    f.KroneckerMultiplyList([X, I, I, I])
    + f.KroneckerMultiplyList([I, X, I, I])
    + f.KroneckerMultiplyList([I, I, X, I])
    + f.KroneckerMultiplyList([I, I, I, X]),
    f.KroneckerMultiplyList([Y, I, I, I])
    + f.KroneckerMultiplyList([I, Y, I, I])
    + f.KroneckerMultiplyList([I, I, Y, I])
    + f.KroneckerMultiplyList([I, I, I, Y]),
    f.KroneckerMultiplyList([Z, I, I, I])
    + f.KroneckerMultiplyList([I, Z, I, I])
    + f.KroneckerMultiplyList([I, I, Z, I])
    + f.KroneckerMultiplyList([I, I, I, Z])
]

Lie_Algebra_Generators_Werner_5 = [
    f.KroneckerMultiplyList([X, I, I, I, I])
    + f.KroneckerMultiplyList([I, X, I, I, I])
    + f.KroneckerMultiplyList([I, I, X, I, I])
    + f.KroneckerMultiplyList([I, I, I, X, I])
    + f.KroneckerMultiplyList([I, I, I, I, X]),
    f.KroneckerMultiplyList([Y, I, I, I, I])
    + f.KroneckerMultiplyList([I, Y, I, I, I])
    + f.KroneckerMultiplyList([I, I, Y, I, I])
    + f.KroneckerMultiplyList([I, I, I, Y, I])
    + f.KroneckerMultiplyList([I, I, I, I, Y]),
    f.KroneckerMultiplyList([Z, I, I, I, I])
    + f.KroneckerMultiplyList([I, Z, I, I, I])
    + f.KroneckerMultiplyList([I, I, Z, I, I])
    + f.KroneckerMultiplyList([I, I, I, Z, I])
    + f.KroneckerMultiplyList([I, I, I, I, Z])
]

Lie_Algebra_Generators_Werner_6 = [
    f.KroneckerMultiplyList([X, I, I, I, I, I])
    + f.KroneckerMultiplyList([I, X, I, I, I, I])
    + f.KroneckerMultiplyList([I, I, X, I, I, I])
    + f.KroneckerMultiplyList([I, I, I, X, I, I])
    + f.KroneckerMultiplyList([I, I, I, I, X, I])
    + f.KroneckerMultiplyList([I, I, I, I, I, X]),
    f.KroneckerMultiplyList([Y, I, I, I, I, I])
    + f.KroneckerMultiplyList([I, Y, I, I, I, I])
    + f.KroneckerMultiplyList([I, I, Y, I, I, I])
    + f.KroneckerMultiplyList([I, I, I, Y, I, I])
    + f.KroneckerMultiplyList([I, I, I, I, Y, I])
    + f.KroneckerMultiplyList([I, I, I, I, I, Y]),
    f.KroneckerMultiplyList([Z, I, I, I, I, I])
    + f.KroneckerMultiplyList([I, Z, I, I, I, I])
    + f.KroneckerMultiplyList([I, I, Z, I, I, I])
    + f.KroneckerMultiplyList([I, I, I, Z, I, I])
    + f.KroneckerMultiplyList([I, I, I, I, Z, I])
    + f.KroneckerMultiplyList([I, I, I, I, I, Z])
]

Lie_Algebra_Generators_Werner_7 = [
    f.KroneckerMultiplyList([X, I, I, I, I, I, I])
    + f.KroneckerMultiplyList([I, X, I, I, I, I, I])
    + f.KroneckerMultiplyList([I, I, X, I, I, I, I])
    + f.KroneckerMultiplyList([I, I, I, X, I, I, I])
    + f.KroneckerMultiplyList([I, I, I, I, X, I, I])
    + f.KroneckerMultiplyList([I, I, I, I, I, X, I])
    + f.KroneckerMultiplyList([I, I, I, I, I, I, X]),
    f.KroneckerMultiplyList([Y, I, I, I, I, I, I])
    + f.KroneckerMultiplyList([I, Y, I, I, I, I, I])
    + f.KroneckerMultiplyList([I, I, Y, I, I, I, I])
    + f.KroneckerMultiplyList([I, I, I, Y, I, I, I])
    + f.KroneckerMultiplyList([I, I, I, I, Y, I, I])
    + f.KroneckerMultiplyList([I, I, I, I, I, Y, I])
    + f.KroneckerMultiplyList([I, I, I, I, I, I, Y]),
    f.KroneckerMultiplyList([Z, I, I, I, I, I, I])
    + f.KroneckerMultiplyList([I, Z, I, I, I, I, I])
    + f.KroneckerMultiplyList([I, I, Z, I, I, I, I])
    + f.KroneckerMultiplyList([I, I, I, Z, I, I, I])
    + f.KroneckerMultiplyList([I, I, I, I, Z, I, I])
    + f.KroneckerMultiplyList([I, I, I, I, I, Z, I])
    + f.KroneckerMultiplyList([I, I, I, I, I, I, Z])
]
# %%
# -------------------------------------------------------------
# --------Generadores de Rotaciones.---------------------------
# -------------------------------------------------------------

Lie_Algebra_Generators_Rotational_1 = [Z]

Lie_Algebra_Generators_Rotational_2 = [np.kron(Z, I) + np.kron(I, Z)]

Lie_Algebra_Generators_Rotational_3 = [
    f.KroneckerMultiplyList([Z, I, I])
    + f.KroneckerMultiplyList([I, Z, I])
    + f.KroneckerMultiplyList([I, I, Z])
]

Lie_Algebra_Generators_Rotational_4 = [
    f.KroneckerMultiplyList([Z, I, I, I])
    + f.KroneckerMultiplyList([I, Z, I, I])
    + f.KroneckerMultiplyList([I, I, Z, I])
    + f.KroneckerMultiplyList([I, I, I, Z])
]

Lie_Algebra_Generators_Rotational_5 = [
    f.KroneckerMultiplyList([Z, I, I, I, I])
    + f.KroneckerMultiplyList([I, Z, I, I, I])
    + f.KroneckerMultiplyList([I, I, Z, I, I])
    + f.KroneckerMultiplyList([I, I, I, Z, I])
    + f.KroneckerMultiplyList([I, I, I, I, Z])
]

Lie_Algebra_Generators_Rotational_6 = [
    f.KroneckerMultiplyList([Z, I, I, I, I, I])
    + f.KroneckerMultiplyList([I, Z, I, I, I, I])
    + f.KroneckerMultiplyList([I, I, Z, I, I, I])
    + f.KroneckerMultiplyList([I, I, I, Z, I, I])
    + f.KroneckerMultiplyList([I, I, I, I, Z, I])
    + f.KroneckerMultiplyList([I, I, I, I, I, Z]) 
]

Lie_Algebra_Generators_Rotational_7 = [
    f.KroneckerMultiplyList([Z, I, I, I, I, I, I])
    + f.KroneckerMultiplyList([I, Z, I, I, I, I, I])
    + f.KroneckerMultiplyList([I, I, Z, I, I, I, I])
    + f.KroneckerMultiplyList([I, I, I, Z, I, I, I])
    + f.KroneckerMultiplyList([I, I, I, I, Z, I, I])
    + f.KroneckerMultiplyList([I, I, I, I, I, Z, I]) 
    + f.KroneckerMultiplyList([I, I, I, I, I, I, Z]) 
]

# %%
# -------------------------------------------------------------
# --------Generadores de Rotaciones (Distintos ángulos).---------------------------
# -------------------------------------------------------------

Lie_Algebra_Generators_Rotational_Different_1 = [Z]

Lie_Algebra_Generators_Rotational_Different_2 = [np.kron(Z, I),
                                       np.kron(I, Z)]

Lie_Algebra_Generators_Rotational_Different_3 = [
    f.KroneckerMultiplyList([Z, I, I]),
    f.KroneckerMultiplyList([I, Z, I]),
    f.KroneckerMultiplyList([I, I, Z])
]

Lie_Algebra_Generators_Rotational_Different_4 = [
    f.KroneckerMultiplyList([Z, I, I, I]),
    f.KroneckerMultiplyList([I, Z, I, I]),
    f.KroneckerMultiplyList([I, I, Z, I]),
    f.KroneckerMultiplyList([I, I, I, Z])
]

Lie_Algebra_Generators_Rotational_Different_5 = [
    f.KroneckerMultiplyList([Z, I, I, I, I]),
    f.KroneckerMultiplyList([I, Z, I, I, I]),
    f.KroneckerMultiplyList([I, I, Z, I, I]),
    f.KroneckerMultiplyList([I, I, I, Z, I]),
    f.KroneckerMultiplyList([I, I, I, I, Z])
]

Lie_Algebra_Generators_Rotational_Different_6 = [
    f.KroneckerMultiplyList([Z, I, I, I, I, I]),
    f.KroneckerMultiplyList([I, Z, I, I, I, I]), 
    f.KroneckerMultiplyList([I, I, Z, I, I, I]), 
    f.KroneckerMultiplyList([I, I, I, Z, I, I]), 
    f.KroneckerMultiplyList([I, I, I, I, Z, I]),
    f.KroneckerMultiplyList([I, I, I, I, I, Z]) 
]

Lie_Algebra_Generators_Rotational_Different_7 = [
    f.KroneckerMultiplyList([Z, I, I, I, I, I, I]), 
    f.KroneckerMultiplyList([I, Z, I, I, I, I, I]), 
    f.KroneckerMultiplyList([I, I, Z, I, I, I, I]), 
    f.KroneckerMultiplyList([I, I, I, Z, I, I, I]), 
    f.KroneckerMultiplyList([I, I, I, I, Z, I, I]), 
    f.KroneckerMultiplyList([I, I, I, I, I, Z, I]), 
    f.KroneckerMultiplyList([I, I, I, I, I, I, Z]) 
]

# %%

# Grabar generators Lie

np.save("Lie_Algebra_Generators/Lie_Algebra_Generators_" + Tipo + "_" + str(int(N)) + ".npy", eval("Lie_Algebra_Generators_" + Tipo + "_" + str(int(N))  ))