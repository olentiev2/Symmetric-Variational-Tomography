# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:22:55 2022

@author: Holik
"""

# %%
######################################################
#################### Libraries #######################
######################################################

import numpy as np

import funciones_SP as f 

import itertools

from itertools import permutations

# %%

N = 3

# %%

# -------------------------------------------------------------
# --------Matrices de Pauli y proyectores.---------------------
# -------------------------------------------------------------

# Paulis.

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1.0j], [1.0j, 0]])
Z = np.array([[1, 0], [0, -1]])

sigmas = [I, X, Y, Z]

# Lista de Paulis de N qubits.

PaulisN = f.KroneckerNSets_SP(sigmas, N)

# Proyectores

print("Calculando Proyectores")

Px = (I + X) / 2

Py = (I + Y) / 2

Pz = (I + Z) / 2

Proyectorcitos = [I, Px, Py, Pz]

ProyectorcitosB = [Px, Py, Pz]

ProyectorcitosN = f.KroneckerNSets_SP(Proyectorcitos, N)

ProyectorcitosN.remove(ProyectorcitosN[0])

# %%

# Mubs de 3 qubits

# Mubs = np.load('mubs_3q.npy')

# MubsObs = []

# for x in Mubs:
#     for i in range(len(x)):
#             temp = x[i]
#             MubsObs.append(temp)

# Proyectorcitos sin simetrizar, pero maximizando i#nformación (para permutacional).

# %%

Proyectorcitos_Normales = []

for comb in itertools.combinations_with_replacement(Proyectorcitos, N):
    L = list(comb)
    O = f.KroneckerMultiplyList_SP(L)
    Proyectorcitos_Normales.append(O)
with open("Observables/Proyectorcitos_Normales_" + str(N) + ".npz", "wb") as file:
    np.savez_compressed(file, *Proyectorcitos_Normales)

