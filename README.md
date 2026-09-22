# Symmetric Variational Quantum Tomography (Symmetric-VQT)

Core code containing functions needed to perform estimation of quantum states with arbitrary symmetries, in combination with the VQT technique (a variant of the MaxEnt technique). 

The theory of combining symmetries with MaxEnt was presented in references **[1, 2]**, and studied numerically in **[3]**. Subsequent improvements and numerical simulations were presented in **[4]**, and in **[5]**, the technique was used to benchmark quantum computers. Different versions and improvements of the functions presented here were used in **[3, 4, 5]**. The contents of this repository provide the implementation tools used across these studies. 

By combining the core functions presented here, users can apply parameterization that leverages arbitrary symmetries of otherwise unknown quantum states with estimation techniques other than VQT, such as Maximum Likelihood Estimation (MaxLik; see, for example, **[5]**).

> **Note on Code Status & Scope:**  
> This repository serves as a reference implementation of the core algorithms and methodology used throughout the published papers **[3, 4, 5]**. While the code successfully produced all reported results, it is provided as research-grade software and is not actively maintained as a production-ready package. The core mathematical and computational building blocks are present, allowing researchers and experienced users to adapt or reproduce the complete technique for their own workflows.

---

## 📁 Files & Usage

* **`Bases_Generator.py`**  
  Contains core functions to generate bases of symmetric subspaces used to reparameterize a density operator for state estimation. It takes as input the symmetries of the system communicated in terms of group generators. This leverages system symmetries to significantly reduce the number of observables needed for state reconstruction.

* **`Optimization_Problem.py`**  
  Contains the implementation for the optimization routine.

* **`funciones_SP.py`**  
  Contains auxiliary functions useful for matrix operations, tensor products, and related tasks.

---

## ⚙️ Requirements

The repository requires Python 3.x along with the following packages (latest versions recommended):

* `numpy`
* `scipy`
* `itertools` (standard library)
* `cvxpy`
* `cvxopt`
* `qutip`

You can install the external dependencies via `pip`:

```bash
pip install numpy scipy cvxpy cvxopt qutip
