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
Modulation base class and Gaussian modulation.
"""


# pylint: disable=too-few-public-methods
class Modulation:
    """
    Base class for modulation.
    """

    va: float  #: Alice's modulation variance.
    w: float  #: Number defined in Denys, A., Brown, P., & Leverrier, A. (2021)

    def __init__(self, va: float):
        """
        Abstract class, meant for subclassing

        Args:
            va (float): variance of the modulation over the complex plan. The variance is
                scalar beacuse the covariance matrix of the modulation are always of
                the form VA*Id. The real and imaginary parts of the modulation are
                indentically and independantly distributed.
                VA = 2*<N> : <N> average number of photons
        """
        self.va = va


# pylint: disable=too-few-public-methods
class GaussianModulation(Modulation):
    """
    Gaussian modulation.
    """

    def __init__(self, va: float):
        """
        w : number (defined in Denys, A., Brown, P., & Leverrier, A. (2021). Explicit
        asymptotic secret key rate of continuous-variable quantum key distribution
        with an arbitrary modulation. Quantum, 5, 540.) that quantifies how much
        weight from a random input coherent state of the modulaytion tau is mapped by
        a_tau, onto a subspace orthogonal from the input coherent state.
        In the particular case of a Gaussian modulation, since a_tau is
        proportional to a, each coherent state is eigen vector for a_tau, and
        is thus mapped by a_tau on its own linear span. Thus w = 0.

        Args:
            va (float): variance of the modulation over the complex plan. The variance is scalar beacuse the covariance matrix of the modulation are always of the form VA*Id. The real and imaginary parts of the modulation are  indentically and independantly distributed. VA = 2*<N> : <N> average number of photons.
        """

        super().__init__(va)

        self.w = 0
