# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 14:45:08 2021

@author: Richard
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%

from datetime import datetime

from functools import reduce

import numpy as np

import math as m

from scipy import linalg as la

from scipy.linalg import logm, expm

from scipy import log, log2

import funciones_SP as f

import itertools

import sympy

import pickle

from itertools import permutations

import itertools

from scipy.sparse import csr_matrix, lil_matrix

from scipy import sparse

N = 8

Tipo = "Permutational"

t_1 = datetime.now()

# %%

# -------------------------------------------------------------
# --------Paulis y proyectores.--------------------------------
# -------------------------------------------------------------

# Paulis.

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1.0j], [1.0j, 0]])
Z = np.array([[1, 0], [0, -1]])
sigmas = [I, X, Y, Z]

print("Calculando Proyectorcitos")

Px = (I + X) / 2

Py = (I + Y) / 2

Pz = (I + Z) / 2

Proyectorcitos = [I, Px, Py, Pz]

ProyectorcitosB = [Px, Py, Pz]

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

print("Calculando Permiones")

PermionsRed = []

for i in range(len(GenPerm)):
    A = f.P(GenPerm[i], N)
    PermionsRed.append(A)
# PermionsRed = np.load('PermionsRed_5.npy')

# PermionsRed = list( f.P(GenPerm[i],N) for i in range(len(GenPerm))  )

np.save("Figuras_Prueba/GenratorsLiePermutational_" + str(int(N)), PermionsRed)
# %%
# -------------------------------------------------------------
# --------Generadores de Werner.-------------------------------
# -------------------------------------------------------------

GeneratorsLieWer2 = [
    np.kron(X, I) + np.kron(I, X),
    np.kron(Y, I) + np.kron(I, Y),
    np.kron(Z, I) + np.kron(I, Z),
]

GeneratorsLieWer3 = [
    f.KroneckerMultiplyList([X, I, I])
    + f.KroneckerMultiplyList([I, X, I])
    + f.KroneckerMultiplyList([I, I, X]),
    f.KroneckerMultiplyList([Y, I, I])
    + f.KroneckerMultiplyList([I, Y, I])
    + f.KroneckerMultiplyList([I, I, Y]),
    f.KroneckerMultiplyList([Z, I, I])
    + f.KroneckerMultiplyList([I, Z, I])
    + f.KroneckerMultiplyList([I, I, Z]),
]

GeneratorsLieWer4 = [
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
    + f.KroneckerMultiplyList([I, I, I, Z]),
]
# %%
# -------------------------------------------------------------
# --------Generadores de Rotaciones.---------------------------
# -------------------------------------------------------------

GeneratorsLieRot = [Z]

GeneratorsLieRot2 = [np.kron(Z, I) + np.kron(I, Z)]

GeneratorsLieRot3 = [
    f.KroneckerMultiplyList([Z, I, I]),
    f.KroneckerMultiplyList([I, Z, I]),
    f.KroneckerMultiplyList([I, I, Z]),
]

# GeneratorsLieRot4 = [f.KroneckerMultiplyList([Z,I,I,I]),f.KroneckerMultiplyList([I,Z,I,I]),f.KroneckerMultiplyList([I,I,Z,I]),f.KroneckerMultiplyList([I,I,I,Z])]

GeneratorsLieRot4 = [
    f.KroneckerMultiplyList([Z, I, I, I])
    + f.KroneckerMultiplyList([I, Z, I, I])
    + f.KroneckerMultiplyList([I, I, Z, I])
    + f.KroneckerMultiplyList([I, I, I, Z])
]

# %%

print("Calculando base canónica...")

B = []
for i in range(2 ** N):
    for j in range(2 ** N):
        if i == j:
            dimensions = (2 ** N, 2 ** N)
            indexA = i
            indexB = i
            arr = lil_matrix(dimensions)
            arr[indexA, indexB] = 1.0
            temp = arr.tocsr()
            B.append(temp)
        elif i < j:
            dimensions = (2 ** N, 2 ** N)
            indexA = i
            indexB = j
            arr = lil_matrix(dimensions)
            arr[indexA, indexB] = 1.0
            arr[indexB, indexA] = 1.0
            temp1 = arr.tocsr()
            B.append(temp1)
        elif j < i:
            indexA = i
            indexB = j
            dimensions = (2 ** N, 2 ** N)
            arr = lil_matrix(dimensions, dtype=complex)
            arr[indexA, indexB] = -1j
            arr[indexB, indexA] = 1j
            temp2 = arr.tocsr()
            B.append(temp2)
CanonicalH = B

with open("Figuras_Prueba/Canonical_" + str(N) + ".npz", "wb") as file:
    np.savez_compressed(file, *CanonicalH)

del CanonicalH

# %%

print("Calculando Commutators...")

t_2 = datetime.now()

CanonicalH_Load = np.load(
    "Figuras_Prueba/Canonical_" + str(N) + ".npz", allow_pickle=True
)

GeneratorsLie = PermionsRed

loop = 0

for i in range(len(GeneratorsLie)):
    Lista_Cruda = []
    loop = loop + 1
    print(f"Loop: {loop}")
    for key in CanonicalH_Load:
        x = CanonicalH_Load[key].item()
        Com = (1j) * (
            csr_matrix(GeneratorsLie[i].astype(np.complex64)) @ x
            - x @ (csr_matrix(GeneratorsLie[i].astype(np.complex64)))
        )
        if Com.count_nonzero() == 0:
            print("Era cero")
            del Com
        else:
            Lista_Cruda.append(Com)
    print("Pickleando")
    Lista_Cruda = {pickle.dumps(x) for x in Lista_Cruda}
    Lista_Cruda = list(dict.fromkeys(Lista_Cruda))
    Lista_Cruda = list(pickle.loads(y) for y in Lista_Cruda)
    basis = []
    counter = 0
    l = len(Lista_Cruda)
    loop = i
    print("Iterando Gram-Schmidt")
    for v in Lista_Cruda:
        y = v.toarray()
        Products = [np.trace(np.matmul(y, (b).conj().T)) for b in basis]
        Relevant = [[Products[j], (basis[j])]
                    for j in range(len(basis)) if Products[j] != 0]
        w = y - sum(Relevant[k][0]*Relevant[k][1]
                    for k in range(len(Relevant)))
        counter = counter + 1
        print(f"Va por el vector: {counter}")
        print(f"Va por el loop: {loop}")
        print(f"Faltan: {l-counter}")
        print(f"Largo Base: {len(basis)}")
        if np.abs(np.trace(np.matmul(w, w.conj().T))) > 1e-14:
            basis.append(
                w/np.sqrt(np.trace(np.matmul(w, w.conj().T))))
            print(f"Largo Base: {len(basis)}")
    with open("Figuras_Prueba/Listas_Achuradas/Temp_1_" + str(i) + Tipo + "_" + str(N) + ".npz", "wb") as file:
        np.savez_compressed(file, *basis)
        print("Largo de la base provisoria")
        print(len(basis))
    del basis
    del Lista_Cruda

data_all = [dict(np.load("Figuras_Prueba/Listas_Achuradas/Temp_1_" + str(i) + Tipo +
                         "_" + str(N) + ".npz", allow_pickle=True)) for i in range(len(GeneratorsLie))]

data = []

for k in range(len(data_all)):
    Mong = data_all[k]
    for l in range(len(data_all[k])-1):
        Mong['L_' + str(k) + '_' + str(l)] = Mong.pop('arr_'+str(l))
    data.append(Mong)

del data_all

Commutators_Rough = {}

for j in range(len(data)):
    [Commutators_Rough.update({k: v}) for k, v in data[j].items()]

np.savez("Figuras_Prueba/Listas_Achuradas/Commutators_Rough_" +
         Tipo + "_" + str(N) + ".npz", **Commutators_Rough)

Commutators_Rough = np.load("Figuras_Prueba/Listas_Achuradas/Commutators_Rough_" +
                            Tipo + "_" + str(N) + ".npz", allow_pickle=True)

print("Picleando Lista")

Commutators_temp_1 = {pickle.dumps(
    Commutators_Rough[key]) for key in Commutators_Rough}

print("Haciendo el dict")

Commutators_temp_2 = list(dict.fromkeys(Commutators_temp_1))

print("Reobteniendo matrices")

CommutatorsA = list(pickle.loads(y) for y in Commutators_temp_2)

Commutators = []

for x in CommutatorsA:
    if (x).shape == (2**N, 2**N):
        Commutators.append(x)

print("Commutators Achurados")

print(len(Commutators))

Tipo = "Permutational"

with open("Figuras_Prueba/Commutators_" + Tipo + "_" + str(N) + ".npz", "wb") as file:
    np.savez_compressed(file, *Commutators)

# del Commutators

del CommutatorsA

del Commutators

t_3 = datetime.now()

print("Duration of process: {}".format(t_3 - t_2))

# %%

print("Calculando Commutators LI...")

t_4 = datetime.now()

Tipo = "Permutational"

print("Lanzando Commutators...")

Commutators_Load = np.load(
    "Figuras_Prueba/Commutators_" + Tipo + "_" + str(N) + ".npz", allow_pickle=True
)

print(len(Commutators_Load))

print("Extrayendo Commutators LI...")

CommutatorsLI = []

counter = 0

l = len(Commutators_Load)

for key in Commutators_Load:
    v = Commutators_Load[key]
    Products = [np.trace(np.matmul(v, (b).conj().T)) for b in CommutatorsLI]
    Relevant = [[Products[j], CommutatorsLI[j]]
                for j in range(len(CommutatorsLI)) if Products[j] != 0]
    w = v - sum(Relevant[k][0]*Relevant[k][1] for k in range(len(Relevant)))
    counter = counter + 1
    print(f"Va por el vector: {counter}")
    print(f"Faltan: {l-counter}")
    if np.abs(np.trace(np.matmul(w, w.conj().T))) > 1e-14:
        CommutatorsLI.append(w/np.sqrt(np.trace(np.matmul(w, w.conj().T))))

print("Largo de CommutatorsLI")

print(len(CommutatorsLI))

with open("Figuras_Prueba/CommutatorsLI_" + Tipo + "_" + str(N) + ".npz", "wb") as file:
    np.savez_compressed(file, *CommutatorsLI)

# t_2 = datetime.now()

# print("Duration_Commutators: {}".format(t_2 - t_1))

del CommutatorsLI


t_5 = datetime.now()

print("Duration of process: {}".format(t_5 - t_4))

# %%


t_6 = datetime.now()

print("Duration of process: {}".format(t_2 - t_1))

print("Lanzando CommutatorsLI")

CommutatorsLI_Load = np.load(
    "Figuras_Prueba/CommutatorsLI_" + Tipo + "_" + str(N) + ".npz", allow_pickle=True
)

CanonicalH_Load = np.load("Figuras_Prueba/Canonical_" +
                          str(N) + ".npz", allow_pickle=True)

print("Pasando a matrices...")

# Acá las hago matrices csr:

CommutatorsLI = [CommutatorsLI_Load[key] for key in CommutatorsLI_Load]

# CanonicalH = [ CanonicalH_Load[key].item()
# for key in CanonicalH_Load]

print("Largo CommutatorsLI")
print(len(CommutatorsLI_Load))

print("Largo Canonical")
print(len(CanonicalH_Load))

print("Calculando Base Total LI...")

PreBasis = CommutatorsLI

del CommutatorsLI

counter = 0

for key in CanonicalH_Load:
    if len(PreBasis) == len(CanonicalH_Load):
        print("No hace falta seguir")
        break
    else:
        x = (CanonicalH_Load[key].item()).toarray()
        Products = [np.trace(np.matmul(x, (b).conj().T)) for b in PreBasis]
        Relevant = [[Products[j], PreBasis[j]]
                    for j in range(len(PreBasis)) if Products[j] != 0]
        w = x - sum(Relevant[k][0]*Relevant[k][1]
                    for k in range(len(Relevant)))
        counter = counter + 1
        print(f"Va por el vector: {counter}")
        if np.abs(np.trace(np.matmul(w, w.conj().T))) > 1e-14:
            PreBasis.append(w / np.sqrt(np.trace(np.matmul(w, w.conj().T))))
            l = len(CanonicalH_Load) - len(PreBasis)
            print(f"Faltan: {l}")

print("Grabando PreBasis...")

Tipo = "Permutational"

with open("Figuras_Prueba/PreBasis_" + Tipo + "_" + str(N) + ".npz", "wb") as file:
    np.savez_compressed(file, *PreBasis)

del PreBasis


t_7 = datetime.now()

print("Duration of process: {}".format(t_7 - t_6))

# %%


print("Lanzando CommutatorsLI y PreBasis")

CommutatorsLI_Load = np.load(
    "Figuras_Prueba/CommutatorsLI_" + Tipo + "_" + str(N) + ".npz", allow_pickle=True
)

PreBasis_Load = np.load(
    "Figuras_Prueba/PreBasis_" + Tipo + "_" + str(N) + ".npz", allow_pickle=True
)

# CommutatorsLI = [CommutatorsLI_Load[key] for key in CommutatorsLI_Load]

PreBasis = [PreBasis_Load[key] for key in PreBasis_Load]

print(f"Largo de CommutatorsLI: {len(CommutatorsLI_Load)}")

print(f"Largo de PreBasis: {len(PreBasis)}")

print("Calculando BasisOrt...")

BasisOrt = list(PreBasis[k]
                for k in range(len(CommutatorsLI_Load), 2 ** (2 * N)))

Tipo = "Permutational"

print("Grabando BasisOrt:")

with open("Figuras_Prueba/BasisOrt_" + Tipo + "_" + str(N) + ".npz", "wb") as file:
    np.savez_compressed(file, *BasisOrt)

print("Largo de la base ortogonal:")

Largo = len(BasisOrt)

print(Largo)

t_8 = datetime.now()

print("Duration of process: {}".format(t_8 - t_1))
