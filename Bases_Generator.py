# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 14:45:08 2021

@author: Holik
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

N = 3

Tipo = "Permutational"

t_1 = datetime.now()

GeneratorsLie = np.load("Lie_algebra_Generators/Lie_Algebra_Generators_" + Tipo + "_" + str(int(N)) + ".npy")

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

with open("Orthogonal_Bases/Canonical_" + str(N) + ".npz", "wb") as file:
    np.savez_compressed(file, *CanonicalH)

del CanonicalH

# %%

print("Calculando Commutators...")

t_2 = datetime.now()

CanonicalH_Load = np.load(
    "Orthogonal_Bases/Canonical_" + str(N) + ".npz", allow_pickle=True
)

# GeneratorsLie = eval(Evaluar)

GeneratorsLie = np.load("Lie_Algebra_Generators/Lie_Algebra_Generators_" + Tipo + "_" + str(int(N)) + ".npy")

loop = 0

for i in range(len(GeneratorsLie)):
    Lista_Cruda = []
    loop = loop + 1
    print(f"Loop: {loop}")
    for key in CanonicalH_Load:
        x = CanonicalH_Load[key].item()
        Com = (1j) * (
            csr_matrix(GeneratorsLie[i]) @ x
            - x @ (csr_matrix(GeneratorsLie[i]))
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
        Products = [np.trace(np.matmul(y, (b.toarray()).conj().T))
                    for b in basis]
        Relevant = [[Products[j], (basis[j]).toarray()]
                    for j in range(len(basis)) if Products[j] != 0]
        w = y - sum(Relevant[k][0]*Relevant[k][1]
                    for k in range(len(Relevant)))
        counter = counter + 1
        print(f"Va por el vector: {counter}")
        print(f"Va por el loop: {loop}")
        print(f"Faltan: {l-counter}")
        if np.abs(np.trace(np.matmul(w, w.conj().T))) > 1e-14:
            basis.append(csr_matrix(
                w/np.sqrt(np.trace(np.matmul(w, w.conj().T)))))
            print(f"Largo Base: {len(basis)}")
    with open("Orthogonal_Bases/Listas_Achuradas/Temp_1_" + str(i) + Tipo + "_" + str(N) + ".npz", "wb") as file:
        np.savez_compressed(file, *basis)
        print("Largo de la base provisoria")
        print(len(basis))
    del basis
    del Lista_Cruda

data_all = [dict(np.load("Orthogonal_Bases/Listas_Achuradas/Temp_1_" + str(i) + Tipo +
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

np.savez("Orthogonal_Bases/Listas_Achuradas/Commutators_Rough_" +
         Tipo + "_" + str(N) + ".npz", **Commutators_Rough)

Commutators_Rough = np.load("Orthogonal_Bases/Listas_Achuradas/Commutators_Rough_" +
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
    if (x.item().toarray()).shape == (2**N, 2**N):
        Commutators.append(x.item())

print("Commutators Achurados")

print(len(Commutators))


with open("Orthogonal_Bases/Commutators_" + Tipo + "_" + str(N) + ".npz", "wb") as file:
    np.savez_compressed(file, *Commutators)

# del Commutators

del CommutatorsA

del Commutators


t_3 = datetime.now()

print("Duration of process: {}".format(t_3 - t_2))

# %%

print("Calculando Commutators LI...")


t_4 = datetime.now()

print("Lanzando Commutators...")

Commutators_Load = np.load(
    "Orthogonal_Bases/Commutators_" + Tipo + "_" + str(N) + ".npz", allow_pickle=True
)

print(len(Commutators_Load))

print("Extrayendo Commutators LI...")

CommutatorsLI = []

counter = 0

l = len(Commutators_Load)

for key in Commutators_Load:
    v = (Commutators_Load[key].item()).toarray()
    Products = [np.trace(np.matmul(v, (b.toarray()).conj().T))
                for b in CommutatorsLI]
    Relevant = [[Products[j], CommutatorsLI[j].toarray()]
                for j in range(len(CommutatorsLI)) if Products[j] != 0]
    w = v - sum(Relevant[k][0]*Relevant[k][1] for k in range(len(Relevant)))
    counter = counter + 1
    print(f"Va por el vector: {counter}")
    print(f"Faltan: {l-counter}")
    if np.abs(np.trace(np.matmul(w, w.conj().T))) > 1e-14:
        CommutatorsLI.append(csr_matrix(
            w/np.sqrt(np.trace(np.matmul(w, w.conj().T)))))
        print(f"Largo Base: {len(CommutatorsLI)}")

print("Largo de CommutatorsLI")

print(len(CommutatorsLI))

with open("Orthogonal_Bases/CommutatorsLI_" + Tipo + "_" + str(N) + ".npz", "wb") as file:
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
    "Orthogonal_Bases/CommutatorsLI_" + Tipo + "_" + str(N) + ".npz", allow_pickle=True
)

CanonicalH_Load = np.load("Orthogonal_Bases/Canonical_" +
                          str(N) + ".npz", allow_pickle=True)

print("Pasando a matrices...")

# Acá las hago matrices csr:

CommutatorsLI = [CommutatorsLI_Load[key].item() for key in CommutatorsLI_Load]

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
        Products = [np.trace(np.matmul(x, (b.toarray()).conj().T))
                    for b in PreBasis]
        Relevant = [[Products[j], PreBasis[j].toarray()]
                    for j in range(len(PreBasis)) if Products[j] != 0]
        w = x - sum(Relevant[k][0]*Relevant[k][1]
                    for k in range(len(Relevant)))
        counter = counter + 1
        print(f"Va por el vector: {counter}")
        if np.abs(np.trace(np.matmul(w, w.conj().T))) > 1e-14:
            PreBasis.append(csr_matrix(
                w / np.sqrt(np.trace(np.matmul(w, w.conj().T)))))
            l = len(CanonicalH_Load) - len(PreBasis)
            print(f"Faltan: {l}")

print("Grabando PreBasis...")

with open("Orthogonal_Bases/PreBasis_" + Tipo + "_" + str(N) + ".npz", "wb") as file:
    np.savez_compressed(file, *PreBasis)

del PreBasis


t_7 = datetime.now()

print("Duration of process: {}".format(t_7 - t_6))

# %%


print("Lanzando CommutatorsLI y PreBasis")

CommutatorsLI_Load = np.load(
    "Orthogonal_Bases/CommutatorsLI_" + Tipo + "_" + str(N) + ".npz", allow_pickle=True
)

PreBasis_Load = np.load(
    "Orthogonal_Bases/PreBasis_" + Tipo + "_" + str(N) + ".npz", allow_pickle=True
)

# CommutatorsLI = [CommutatorsLI_Load[key] for key in CommutatorsLI_Load]

PreBasis = [PreBasis_Load[key].item() for key in PreBasis_Load]

print(f"Largo de CommutatorsLI: {len(CommutatorsLI_Load)}")

print(f"Largo de PreBasis: {len(PreBasis)}")

print("Calculando BasisOrt...")

BasisOrt = list(PreBasis[k].toarray()
                for k in range(len(CommutatorsLI_Load), 2 ** (2 * N)))

print("Grabando BasisOrt:")

with open("Orthogonal_Bases/Bases/BasisOrt_" + Tipo + "_" + str(N) + ".npz", "wb") as file:
    np.savez_compressed(file, *BasisOrt)

print("Largo de la base ortogonal:")

Largo = len(BasisOrt)

print(Largo)

t_8 = datetime.now()

print("Duration of process: {}".format(t_8 - t_1))
