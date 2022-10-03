# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:31:28 2022

@author: Holik
"""

# %%

import pickle

from datetime import datetime

import cvxpy as cp

from functools import reduce

from cvxopt import solvers, blas, matrix, spmatrix, spdiag, log, div

import numpy as np

import math as m

from scipy import linalg as la

from scipy.linalg import logm, expm

from scipy import log, log2

from scipy.stats import unitary_group

import funciones_SP as f

import itertools

import sympy

from itertools import permutations

import itertools

from scipy.sparse import csr_matrix, lil_matrix

from scipy import sparse

# %%

N = 3

Num_Cats = 10

num_perm = 10

Num_Perm = 1

Num_Wer2q = 100

# %%

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1.0j], [1.0j, 0]])
Z = np.array([[1, 0], [0, -1]])

sigmas=[I,X,Y,Z]

PaulisN = f.KroneckerNSets(sigmas,N)

ListasN = f.Lists(sigmas,N)

# %%

# ------------------------------------------------------------
# ---------Generamos distintos tipos de estados.--------------
# ------------------------------------------------------------

print("Calculando Estados Permutacionales")

# Estos son con simetría permutacional.

Set_Permutational = []

Zero = np.array([[1], [0]])

V0 = list(itertools.repeat(Zero, N))

One = np.array([[0], [1]])

# Armo unos estados simétricos.

VList = list(itertools.repeat(Zero, N))

VListOne = list(itertools.repeat(One, N))

Vect = f.KroneckerMultiplyList(VList)

VectOne = f.KroneckerMultiplyList(VListOne)

V_GHZ = (1 / np.sqrt(2)) * (Vect + VectOne)

W = V_GHZ.conj().T
RHO_GHZ = np.outer(V_GHZ, W)

Set_Permutational.append(RHO_GHZ)

Set_GHZ = [RHO_GHZ]

# for i in range(Num_Perm):
#     U = unitary_group.rvs(2**N)
#     V = np.dot(U,Vect)
#     W = V.conj().T
#     RHO5 = np.outer(V,W)
#     RHO = sum((1/2**N)*(np.trace(RHO5@a))*(f.Symmetrizer(b)) for (a,b) in zip(PaulisN,ListasN) )
#     Set_Permutational.append(RHO)

# np.save('States/Permutacionalmente_Invariantes_'+str(N)+'.npy', Set_Permutational)

# Set_Permutational = np.load('States/Permutacionalmente_Invariantes_'+str(N)+'.npy', allow_pickle = True)

# %%

# Cat-like states:
    
print("Calculando Estados Cat_Like_Mixed")    
    
Name = "Cat_Like_Mixed"    

Set_Cats_Mixed = []

P = []

Alpha = 0.5

for i in range(Num_Cats + 1):
    p = (0.5 / Num_Cats) * i
    Psi_p = np.sqrt(p) * Vect + np.sqrt(1 - p) * VectOne
    Psi_p_Dagger = Psi_p.conj().T
    RHO_p = Alpha * (1 / 2 ** N) * (np.eye(2 ** N)) + (1 - Alpha) * np.outer(
        Psi_p, Psi_p_Dagger
    )
    Set_Cats_Mixed.append(RHO_p)
    P.append(p)
    
np.save("States/" + Name + "_" + str(int(N)), Set_Cats_Mixed) 

np.save("States/" + Name  + "_Parameters" + "_" + str(int(N)), P) 
  
# %%

print("Calculando Estados Cat_Like")    

Name = "Cat_Like"

Set_Cats = []

P = []

for i in range(Num_Cats + 1):
    p = (0.5 / Num_Cats) * i
    Psi_p = np.sqrt(p) * Vect + np.sqrt(1 - p) * VectOne
    Psi_p_Dagger = Psi_p.conj().T
    RHO_p = np.outer(Psi_p, Psi_p_Dagger)
    Set_Cats.append(RHO_p)
    P.append(p)

np.save("States/" + Name + "_" + str(int(N)), Set_Cats)

np.save("States/" + Name  + "_Parameters" + "_" + str(int(N)), P)

# %%

# Muchos Werners de dos qubits:
 
print("Calculando Estados Werners")        
    
Name = "Werner_2q"    

Swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

Psym = (1 / 2) * (np.kron(I, I) + Swap)

Pas = (1 / 2) * (np.kron(I, I) - Swap)

Set_Werner2q = []

for i in range(Num_Wer2q + 1):
    Alpha = -1 + (2 / Num_Wer2q) * i
    RHO3 = (1 / (4 - 2 * Alpha)) * (np.kron(I, I) - Alpha * Swap)
    Set_Werner2q.append(RHO3)

np.save("States/" + Name + "_" + str(int(N)), Set_Cats)

