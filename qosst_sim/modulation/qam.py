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
QAM modulation.
"""
from math import sqrt, exp, factorial

import numpy as np
import scipy.linalg as alg

from qosst_sim.utils import kron
from qosst_sim.modulation.modulation import Modulation


# pylint: disable=too-few-public-methods
class QAM(Modulation):
    """
    Quadrature and Amplitude Modulation (QAM).
    """

    dim: int  #: Dimension of the Fock space.
    distribution: np.ndarray  #: Distribution of probability over the constellation.
    constellation: np.ndarray  #: Sqaare constellation of qam points.
    tau_half: (
        np.ndarray
    )  #: Matrix defined in Denys, A., Brown, P., & Leverrier, A. (2021).
    a_tau: (
        np.ndarray
    )  #: Matrix defined in Denys, A., Brown, P., & Leverrier, A. (2021).
    size: int

    def __init__(self, dim: int, va: float):
        """
        Child class from modulation, defining the subset of QAM modulation, which
        corresponds to a finite set of accessible complex values, distributed on
        a regular square grid {x +  i*y | x, y = = (-m + 1), (-m + 3), . . . , (m - 1) }.
        The coherent states corresponding to these complexe numbers are element of
        the Fock space Span({| n >, n >= 0 }), but one need to truncate the dimension
        of the space to dim.

        Args:
            dim (int): dimension of the Fock space.
            va (float): variance of the modulation over the complex plan. The variance is
                scalar beacuse the covariance matrix of the modulation are always of
                the form VA*Id. The real and imaginary parts of the modulation are
                indentically and independantly distributed.
                VA = 2*<N> : <N> average number of photons
        """
        super().__init__(va)
        self.dim = dim

        # matrix of the anihilation operator of the Fock space in the truncated basis
        self.a = np.zeros((dim, dim), dtype=complex)

        tau = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            for j in range(dim):
                self.a[i, j] = sqrt(j) * kron(j - 1, i)
                tau[i, j] = sum(
                    weight * exp(-abs(alpha) ** 2) * alpha**i * alpha.conjugate() ** j
                    for weight, alpha in zip(self.distribution, self.constellation)
                ) / (sqrt(factorial(i)) * sqrt(factorial(j)))

        # matrix square root of the modulation matrix tau in the truncated Fock basis
        self.tau_half = alg.sqrtm(tau)
        # matrix of a_tau in the truncated Fock basis
        self.a_tau = self.tau_half @ self.a @ alg.pinvh(self.tau_half)

        # w : number (defined in Denys, A., Brown, P., & Leverrier, A. (2021). Explicit
        # asymptotic secret key rate of continuous-variable quantum key distribution
        # with an arbitrary modulation. Quantum, 5, 540.) that quantifies how much
        # weight from a random input coherent state of the modulation tau is mapped by
        # a_tau, onto a subspace orthogonal from the input coherent state.

        w = 0
        a_tau_dag = self.a_tau.T.conjugate()

        for weight, alpha in zip(self.distribution, self.constellation):
            coherent_state = self.coherent_state(alpha)
            coherent_state_dag = coherent_state.T.conjugate()

            w += weight * (
                coherent_state_dag @ a_tau_dag @ self.a_tau @ coherent_state
                - abs(coherent_state_dag @ self.a_tau @ coherent_state) ** 2
            )

        self.w = w.real

    def coherent_state(self, alpha: complex) -> np.ndarray:
        """
        Args:
            alpha (complex): eigenvalue of the coherent state for the anihilation operator a.

        Returns:
            np.ndarray: vector of the alpha coherent state in the truncated Fock basis of size dim.
        """
        return exp(-abs(alpha) ** 2 / 2) * np.array(
            [alpha**n / sqrt(factorial(n)) for n in range(self.dim)]
        )
