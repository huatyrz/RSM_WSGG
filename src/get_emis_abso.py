"""
Just contain a function to get emissivity or absorptivity
"""
from typing import Tuple
import numpy as np
import pandas as pd
from scipy import constants
import tqdm
from typing import Iterable
# import os

C1 = 2*constants.pi*constants.h*constants.c**2*1e8
C2 = constants.h*constants.c/constants.k*1e2

# 吸收系数数据库路径和参数网格定义
ABSC_DB = "../AbscDB"
VOLUME_FRACTION_IN_DB = np.linspace(0., 1., 5)
SPECIES =["H2O", "NH3", "NH3"]

X_GRID = np.linspace(0, 1, 5)
T_GRID = np.arange(300.0, 3001.0, 100.0)

def neighbor_index(num: float, nums: Iterable[float])->Tuple[int]:
    """
    # ! This function will not check if the nums is sorted or not
    Suppose nums is a sorted array from small to large as:
    nums = [num_0, num_1, num_2, ..., num_n]

    return i, i+1 if num>=num_i and num<=num_(i+1)

    if the num is smaller than nums[0] or larger than nums[-1], the function
    will return 0, 1 or -2, -1

    Parameters
    ----------
    num : float
        _description_
    nums : Iterable[float]
        _description_

    Returns
    -------
    Tuple[int]
        _description_
    """
    if num<nums[0]:
        return 0, 1
    if num>nums[-1]:
        return len(nums)-2, len(nums)-1
    for i, _ in enumerate(nums):
        if num>=nums[i] and num<=nums[i+1]:
            return i, i+1


def neighbor_value(num: float, nums: Iterable[float])->Tuple[float]:
    """
    # ! This function will not check if the nums is sorted or not
    Return the corresponding value of the index get from the neighbor_index

    Parameters
    ----------
    num : float
        _description_
    nums : Iterable[float]
        _description_

    Returns
    -------
    Tuple[int]
        _description_
    """
    i, j = neighbor_index(num, nums)
    return nums[i], nums[j]


def add_array(array1, array2):
    """Add two array with different dimension

    Parameters
    ----------
    array1 : _type_
        _description_
    array2 : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    max_len = np.max([len(array1), len(array2)])
    if len(array1)!=max_len:
        array1 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(array1)), array1)
    if len(array2)!=max_len:
        array2 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(array2)), array2)
    return array1+array2

def minus_array(array1, array2):
    """Minus two array with different dimension

    Parameters
    ----------
    array1 : _type_
        _description_
    array2 : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    max_len = np.max([len(array1), len(array2)])
    if len(array1)!=max_len:
        array1 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(array1)), array1)
    if len(array2)!=max_len:
        array2 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(array2)), array2)
    return array1-array2

def i_b_eta(temperature:float, eta:float, n:float = 1): # pylint: disable=invalid-name
    """
    return the black body radiation intensity

    Parameters
    ----------
    temperature : float
        In Kelvin
    eta : float
        In cm-1
    n : float, optional

    Returns
    -------
    black body intensity : float
        in terms of wavenumber with dimension cm-1
    """
    e_b_lambda = C1*eta**3/(n**2*(np.exp(C2*eta/(n*temperature))-1))
    return e_b_lambda/np.pi


def get_emis_abso(p: float, Tg: float, Ts: float, pL: float, x1: float,
    x2: float, x3: float)->Tuple[float, float]:
    """A function used to calculate emissivity and absorptivity.

    Parameters
    ----------
    p : float
        pressure, in bar
    Tg : float
        tempereature for gas, in K
    Ts : float
        temperature for surface, in K
    pL : float
        pressure optical path length, in bar*cm
    x1 : float
        volume fraction for H2O
    x2 : float
        volume fraction for CO2
    x3 : float
        volume fraction for CO

    Returns
    -------
    Tuple[float, float]
        emis, abso
    """
    L = pL/p
    xs = [x1, x2, x3]
    kappas = [None for i in range(3)]
    etas = [None for i in range(3)]
    for i in range(3):
        kappas[i] = get_kappas(p, SPECIES[i], xs[i], Tg)* p * xs[i]
        # xl, xh = neighbor_value(xs[i], VOLUME_FRACTION_IN_DB) # l=low, h=high
        # kappas_l = np.load(f"./WSGG_H2O_NH3_AbscDB/{p:04.1f}/{SPECIES[i]}/{xl:.2f}_{int(Tg):04d}.npy")
        # kappas_h = np.load(f"./WSGG_H2O_NH3_AbscDB/{p:04.1f}/{SPECIES[i]}/{xh:.2f}_{int(Tg):04d}.npy")
        # kappas[i] = add_array(kappas_l, minus_array(kappas_h, kappas_l)*(xs[i]-xl)/(xh-xl))* p * xs[i]
    kappas = add_array(add_array(kappas[0], kappas[1]), kappas[2])
    etas = np.linspace(0.1, 15000.0, len(kappas))
    i_bs = i_b_eta(Ts, etas)
    i_bg = i_b_eta(Tg, etas)
    d_wn = etas[1]-etas[0]
    common_term = 1-np.exp(-kappas*L)
    emis = np.sum(i_bg*common_term)* \
                d_wn/(constants.sigma*Tg**4)*np.pi
    abso = np.sum(i_bs*common_term)* \
                d_wn/(constants.sigma*Ts**4)*np.pi
    return emis, abso


# def get_kappas(p: float, specie: str, x: float, T: float):
#     id_x = np.searchsorted(X_GRID, x, side='right') - 1
#     id_T = np.searchsorted(T_GRID, T, side='right') - 1
#     if x == X_GRID[id_x] and T == T_GRID[id_T]:
#         return np.load(f"../AbscDB/{p:04.1f}/{specie}/{x:.2f}_{int(T):04d}.npy")
#     elif x == X_GRID[id_x]:
#         Tl = T_GRID[id_T]
#         Tr = T_GRID[id_T + 1]
#         kl = np.load(f"../AbscDB/{p:04.1f}/{specie}/{x:.2f}_{int(Tl):04d}.npy")
#         kr = np.load(f"../AbscDB/{p:04.1f}/{specie}/{x:.2f}_{int(Tr):04d}.npy")
#         return add_array(kl, minus_array(kr, kl) * (T - Tl) / (Tr - Tl))
#     elif T == T_GRID[id_T]:
#         xl = X_GRID[id_x]
#         xr = X_GRID[id_x + 1]
#         kl = np.load(f"../AbscDB/{p:04.1f}/{specie}/{xl:.2f}_{int(T):04d}.npy")
#         kr = np.load(f"../AbscDB/{p:04.1f}/{specie}/{xr:.2f}_{int(T):04d}.npy")
#         return add_array(kl, minus_array(kr, kl) * (x - xl) / (xr - xl))
#     else:
#         xl = X_GRID[id_x]
#         xr = X_GRID[id_x + 1]
#         Tl = T_GRID[id_T]
#         Tr = T_GRID[id_T + 1]
#         kll = np.load(f"../AbscDB/{p:04.1f}/{specie}/{xl:.2f}_{int(Tl):04d}.npy")
#         klr = np.load(f"../AbscDB/{p:04.1f}/{specie}/{xl:.2f}_{int(Tr):04d}.npy")
#         krl = np.load(f"../AbscDB/{p:04.1f}/{specie}/{xr:.2f}_{int(Tl):04d}.npy")
#         krr = np.load(f"../AbscDB/{p:04.1f}/{specie}/{xr:.2f}_{int(Tr):04d}.npy")
#         kl = add_array(kll, minus_array(klr, kll) * (T - Tl) / (Tr - Tl))
#         kr = add_array(krl, minus_array(krr, krl) * (T - Tl) / (Tr - Tl))
#         return add_array(kl, minus_array(kr, kl) * (x - xl) / (xr - xl))

def to_same_grid(kappas):
    max_n = max([len(kappa) for kappa in kappas])
    for i, kappa in enumerate(kappas):
        n = len(kappa)
        if n < max_n:
            kappas[i] = np.interp(np.linspace(0, 1, max_n), np.linspace(0, 1, n), kappas[i])
    return kappas

def get_kappas(p: float, specie: str, x: float, T: float):
    id_x = np.searchsorted(X_GRID, x, side='right') - 1
    id_T = np.searchsorted(T_GRID, T, side='right') - 1
    if x == X_GRID[id_x] and T == T_GRID[id_T]:
        return np.load(f"{ABSC_DB}/{p:04.1f}/{specie}/{x:.2f}_{int(T):04d}.npy")
    elif x == X_GRID[id_x]:
        Tl = T_GRID[id_T]
        Tr = T_GRID[id_T + 1]
        kl = np.load(f"{ABSC_DB}/{p:04.1f}/{specie}/{x:.2f}_{int(Tl):04d}.npy")
        kr = np.load(f"{ABSC_DB}/{p:04.1f}/{specie}/{x:.2f}_{int(Tr):04d}.npy")
        kl, kr = to_same_grid([kl, kr])
        return kl + (kr - kl) * (T - Tl) / (Tr - Tl)
    elif T == T_GRID[id_T]:
        xl = X_GRID[id_x]
        xr = X_GRID[id_x + 1]
        kl = np.load(f"{ABSC_DB}/{p:04.1f}/{specie}/{xl:.2f}_{int(T):04d}.npy")
        kr = np.load(f"{ABSC_DB}/{p:04.1f}/{specie}/{xr:.2f}_{int(T):04d}.npy")
        kl, kr = to_same_grid([kl, kr])
        return kl + (kr - kl) * (x - xl) / (xr - xl)
    else:
        xl = X_GRID[id_x]
        xr = X_GRID[id_x + 1]
        Tl = T_GRID[id_T]
        Tr = T_GRID[id_T + 1]
        kll = np.load(f"{ABSC_DB}/{p:04.1f}/{specie}/{xl:.2f}_{int(Tl):04d}.npy")
        klr = np.load(f"{ABSC_DB}/{p:04.1f}/{specie}/{xl:.2f}_{int(Tr):04d}.npy")
        krl = np.load(f"{ABSC_DB}/{p:04.1f}/{specie}/{xr:.2f}_{int(Tl):04d}.npy")
        krr = np.load(f"{ABSC_DB}/{p:04.1f}/{specie}/{xr:.2f}_{int(Tr):04d}.npy")
        kll, klr, krl, krr = to_same_grid([kll, klr, krl, krr])
        kl = kll + (klr - kll) * (T - Tl) / (Tr - Tl)
        kr = krl + (krr - krl) * (T - Tl) / (Tr - Tl)
        return kl + (kr - kl) * (x - xl) / (xr - xl)


if __name__=="__main__":
    print(get_emis_abso(1.0, 300.0, 310.0, 100.0, 0.1, 0.1, 0.1))
    print(np.allclose(get_kappas("H2O", 0.25, 300.0), np.load("spectrum/01.0/H2O/0.25_0300.npy"), rtol=1e-2))

