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
Gaussian QAM modulation.
"""
from itertools import product

import numpy as np

from qosst_sim.modulation.qam import QAM


# pylint: disable=too-few-public-methods
class GaussianQAM(QAM):
    """
    QAM approximating a Gaussian distribution.
    """

    def __init__(self, dim: int, size: int, va: float, nu: float):
        """
        Args:
            dim (int): dimension of truncation of the Fock basis.
            m (int): side length of the QAM.
            va (float): variance of the modulation over the complex plan. The variance is
                scalar beacuse the covariance matrix of the modulation are always of
                the form va*Id. The real and imaginary parts of the modulation are
                indentically and independantly distributed. va = 2*<n> : <n> average number of photons
            nu (float): positive number meant to be optimised according to certain cost function.
        """
        self.size = size
        self.nu = nu

        # the constellation and the distribution of the modulation
        quadratures = range(-size + 1, size, 2)
        constellation = np.array(
            [x + 1j * y for x, y in product(quadratures, quadratures)]
        )
        total_weight = sum(np.exp(-nu * abs(constellation) ** 2))

        # probability distribution
        self.distribution = np.exp(-nu * abs(constellation) ** 2) / total_weight
        # renormalised constellation
        self.constellation = constellation * np.sqrt(
            va / (2 * np.dot(np.abs(constellation) ** 2, self.distribution))
        )
        # initialisation of all the matrices and and parameters depending of the distribution
        super().__init__(dim, va)
