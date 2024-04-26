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
Binomial QAM modulation.
"""
from math import sqrt, comb
from itertools import product

import numpy as np

from qosst_sim.modulation.qam import QAM


# pylint: disable=too-few-public-methods
class BinomialQAM(QAM):
    """
    Binomial QAM modulation.
    """

    def __init__(self, dim: int, size: int, va: float):
        """
        Args:
            dim (int): dimension of truncation of the Fock basis
            m (int): side length of the QAM
            va (float): variance of the modulation over the complex plan. The variance is scalar beacuse the covariance matrix of the modulation are always of the form va*Id. The real and imaginary parts of the modulation are indentically and independantly distributed. va = 2*<n> : <n> average number of photons
        """
        self.size = size

        # the constellation and the distribution of the modulation
        quadratures = range(-size + 1, size, 2)
        constellation = np.array(
            [x + 1j * y for x, y in product(quadratures, quadratures)]
        )

        # probability distribution
        self.distribution = np.array(
            [
                2 ** (-2 * (size - 1)) * comb(size - 1, k) * comb(size - 1, l)
                for k, l in product(range(size), range(size))
            ]
        )
        # renormalised constellation
        self.constellation = constellation * sqrt(va / (4 * (size - 1)))
        # initialisation of all the matrices and and parameters depending of the distribution
        super().__init__(dim, va)
