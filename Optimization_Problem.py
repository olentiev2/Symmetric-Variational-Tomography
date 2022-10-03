# -*- coding: utf-8 -*-
"""
Created on Tue May 24 12:44:57 2022

@author: Holik
"""
# %%
import matplotlib.pyplot as plt

import matplotlib

import statistics

from qutip import *

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

N = 2

Tipo = "Permutational"

Estados = "Cat_like"

Observables = "Proyectorcitos_Normales" # BasisOrt, ProyectorcitosNSymm , ProyectorcitosN, ProyectorcitosNSymm

# %%

#------------------------------------------------------------------
#----------------------Lanzamos cosas------------------------------
#------------------------------------------------------------------

Observables_elegidos_Load = np.load(
    "Observables/"+ Observables + "_" + str(N) + ".npz", allow_pickle=True
)

Observables_elegidos = [Observables_elegidos_Load[key].item().toarray() for key in Observables_elegidos_Load] 

Estados_Load = np.load(
    "States/"+ Estados + "_" + str(N) + ".npy", allow_pickle=True
)


BasisOrt_Load = np.load(
    "Orthogonal_Bases/Bases/BasisOrt_" + Tipo + "_" + str(N) + ".npz", allow_pickle=True
)

BasisOrt = [BasisOrt_Load[key] for key in BasisOrt_Load] 

print("Largo de la base ortogonal (dimensión del espacio)")
Largo = len(BasisOrt)
print(Largo)

# %%

# -----------------------------------------------------------------
# ---------Optimizamos.--------------------------------------------
# -----------------------------------------------------------------

print("Entrando en la optimización")

# -----------------------------------------------------------
# --Definimos variables y especificamos observables elegidos:
# -----------------------------------------------------------


a = cp.Variable(len(BasisOrt))

expr1 = sum(a[i] * BasisOrt[i] for i in range(len(BasisOrt))) 

#+ a[len(BasisOrt)]*(1 / 2 ** N) * (np.eye(2 ** N)) 

##### Para el ruido.##############################
mu = 0.18  # num  medio de fotones por pulso de laser
dc = 5 * (10 ** (-4))  # ir variando con valores del mismo orden
N_intentos = 5 * (10 ** 4)
##################################################

# %%

Fidelidades = []

Trace_Distances = []

Obtenidos = []

EstadosComp = []

Nro_de_Observables_Medidos = len(
    Observables_elegidos
)  # len(Observables_elegidos) si los queremos todos

Observables_Medidos = list(
    Observables_elegidos[i] for i in range(Nro_de_Observables_Medidos)
)

Observables_No_Medidos = list(
    Observables_elegidos[i]
    for i in range(Nro_de_Observables_Medidos, len(Observables_elegidos))
)

d = cp.Variable(len(Observables_Medidos))

Conjunto_Elegido = Estados_Load  # , Set_Permutational, Set_IBMQ

Nombre_Conjunto_Elegido = Estados

Nombre_Observables_Elegidos = Observables

counter = 0

counter_err = 0

Diff_viejo = []

# %%

from datetime import datetime

for X in Conjunto_Elegido:
    print("Calculando F0")
    F0 = list(
        np.trace(X @ Observables_Medidos[i]).astype(float)
        for i in range(len(Observables_Medidos)))
    F0 = np.array(F0)
    # mean and standard deviation (para ruido Gausssiano)
    media, sigma = 0, 0.01
    s = np.random.normal(media, sigma, len(F0))
    F1 = F0 + s
    print("Calculando frecs_exp con binord")
    n = np.random.binomial(N_intentos, 1 - np.exp(-mu * F0 - dc)) 
    frecs_exp = n / (mu * N_intentos)
    ValoresMedios = list(
        d[i] * frecs_exp[i]
        >= cp.abs(cp.trace(expr1 @ Observables_Medidos[i]) - frecs_exp[i])
        for i in range(len(Observables_Medidos)))
    Ineqs = list(d[i] >= 0 for i in range(len(Observables_Medidos)))
    constr1 = [expr1 >> 0, cp.trace(expr1) == 1] + ValoresMedios + Ineqs
    Variational = 100 * sum(d[i] for i in range(len(Observables_Medidos))) - 0.1*cp.log_det(expr1)     
    obj = cp.Minimize(Variational)
    t_1 = datetime.now()
    print("Resolviendo el problema")
    prob = cp.Problem(obj, constr1)
    prob.solve(
        solver=cp.SCS,
        verbose=False,
        eps=0.0001e-01,
        alpha=1,
        max_iters=1000,
        normalize=True,
        scale=0.1,
        acceleration_lookback=19,
        rho_x=1.00e-08,
        warm_start=True,
    )
    print("Recuperando el RHO y comparando")
    R = a.value
    t_2 = datetime.now()
    print("Duración de la Optimización: {}".format(t_2 - t_1)) 
    try:
        RM0 = sum(a.value[i] * BasisOrt[i] for i in range(len(BasisOrt)))
        RM = RM0 / np.trace(RM0)
        Fid = f.fidelity(X, RM)
        TD = tracedist(Qobj(RM), Qobj(X))
        Fidelidades.append(Fid)
        Trace_Distances.append(TD)
        Obtenidos.append(RM)
        EstadosComp.append(X)
        counter = counter + 1
        print(
            "Este converge---> " + str(counter) +
            "Fid:" + str(Fid) + "   TD:" + str(TD)
        )           
    except TypeError:
        print("Este no converge---> " + str(counter))
        counter_err = counter_err + 1
print("Grabando")

np.save(
    "Figuras_Prueba/Fidelidades_"
    + Nombre_Conjunto_Elegido
    + "_"
    + Nombre_Observables_Elegidos
    + "_"
    + str(int(N)),
    Fidelidades,
)

np.save(
    "Figuras_Prueba/Elegidos_" + Nombre_Conjunto_Elegido + "_" + str(int(N)),
    EstadosComp,
)

np.save(
    "Figuras_Prueba/Obtenidos_"
    + Nombre_Conjunto_Elegido
    + "_"
    + Nombre_Observables_Elegidos
    + "_"
    + str(int(N)),
    Obtenidos,
)

# %%

print("Prnteando cosas y haciendo histogramas")


print("Fidelidades")
print(Fidelidades)
print("Media")
print(statistics.mean(Fidelidades))

# -----------------------------------------------------------------
# ---------Armamos un histograma y grabamos cositas.---------------
# -----------------------------------------------------------------

hist, bin_edges = np.histogram(Fidelidades, bins=100)

np.save("Figuras_Prueba/Hist_" + str(int(N)), hist)


fig = plt.figure(figsize=[10, 8])

plt.bar(bin_edges[:-1], hist, width=0.001, color="#0504aa", alpha=0.7)
plt.xlim(min(bin_edges), max(bin_edges))
plt.grid(axis="y", alpha=0.75)
plt.xlabel("Value", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Frequency", fontsize=15)
plt.title("Histograma del mal", fontsize=15)
plt.show()
fig.savefig("Figuras_Prueba/Histogram_Cat_" + str(int(N)) + ".png")


plt.plot(Fidelidades, linestyle="", marker="o", label="Valores obtenidos")
plt.hlines(
    y=1,
    xmin=0,
    xmax=[len(Fidelidades)],
    colors="purple",
    linestyles="--",
    lw=2,
    label="Fidelidad uno",
)
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.show()
