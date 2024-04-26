# qosst-sim - Simulation module of the Quantum Open Software for Secure Transmissions.
# Copyright (C) 2021-2024 Mayeul Chavanne
# Copyright (C) 2021-2024 Yoann Piétri

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Some utils for the qosst_sim module.
"""
from math import log2

# import matplotlib.pyplot as plt


# TO REMOVE ? So we can avoid aving matplotlib as a dependency
# def plot_matrix(M):
#     """plot a matrix with non negative coeffs, using a colorbar"""
#     M = abs(M)
#     plt.imshow(M)
#     plt.colorbar()
#     plt.show()


def kron(i: int, j: int) -> int:
    """Kronecker's delta on int

    Args:
        i (int): first int,
        j (int): second int.

    Returns:
        int: 1 if i=j and 0 otherwise.
    """
    return int(i == j)


def g(x: float) -> float:
    """Function of the Von Neumann entropy of a thermal state

    Args:
        x (float): input.

    Returns:
        float: output of the g function.
    """
    return (x + 1) * log2(x + 1) - x * log2(x)


def delta(v: float, w: float, z: float) -> float:
    """Coeficent needed to compute the sympleptic values of the 2-modes covariance matrix

    Args:
        V (float): coefficient of the first diagonal block of the covariance matrix.
        W (float): coefficient of the second diagonal block of the covariance matrix.
        Z (float): coefficient of the antidiagonal blocks of the covariance matrix.

    Returns:
        float: delta coefficient needed to compute the symplectic values.
    """
    return v**2 + w**2 - 2 * z**2


def gamma(v: float, w: float, z: float) -> float:
    """Coeficent needed to compute the sympleptic values of the 2-modes covariance matrix

    Args:
        V (float): coefficient of the first diagonal block of the covariance matrix.
        W (float): coefficient of the second diagonal block of the covariance matrix.
        Z (float): coefficient of the antidiagonal blocks of the covariance matrix.

    Returns:
        float: gamma coefficient needed to compute the symplectic values.
    """
    return v**2 * w**2 - 2 * v * w * z**2 + z**4


def transmission(distance: float) -> float:
    """Transmission as a function of the distance in usual optical fiber.

    This assumes an attenuation coefficient of 0.2 dB/km.

    Args:
        distance (float): distance in km.

    Returns:
        float: associated ttenuation.
    """
    return 10 ** (-0.02 * distance)
