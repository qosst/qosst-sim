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
Module containing the class of detectors.
"""
import abc
from typing import Tuple
from math import sqrt

from qosst_sim.utils import delta, gamma, g


# pylint: disable=invalid-name
class Detector(abc.ABC):
    """
    Abstract class for detectors.
    """

    eta: float  #: Efficiency of the detector.
    vel: float  #: Electronic noise of the detector.

    @abc.abstractmethod
    def sympl(self, V: float, W: float, Z: float) -> Tuple:
        """Compute and return the symplectic eigenvalues.

        Args:
            V (float): coefficient of the first diagonal block of the covariance matrix.
            W (float): coefficient of the second diagonal block of the covariance matrix.
            Z (float): coefficient of the antidiagonal blocks of the covariance matrix.

        Returns:
            Tuple: a tuple containing the eigenvalues. The number of elements depends on the detector.
        """

    @abc.abstractmethod
    def holevo_bound(self, V: float, W: float, Z: float) -> float:
        """Compute the Holevo's bound using the symplectic eigenvalues.

        Args:
            V (float): coefficient of the first diagonal block of the covariance matrix
            W (float): coefficient of the second diagonal block of the covariance matrix
            Z (float): coefficient of the antidiagonal blocks of the covariance matrix

        Returns:
            float: Holevo's bound.
        """


class IdealHeterodyneDetector(Detector):
    """
    Class containing the caractising datas of an Ideal Heterodyne Detector, which
    has a quantum efficiency eta = 1, and an electric noise Vel = 0. The functions
    that compute the sympleptic eigenvalues and the Holevo bound of the
    covariance matrix describded by V, W and Z. These three quantities depends
    of the simulation.
    """

    def __init__(self):
        """
        Assign eta to be 1 and vel to be 0 in the case of the ideal detector.
        """
        self.eta = 1
        self.vel = 0

    def sympl(self, V: float, W: float, Z: float) -> Tuple[float, float, float]:
        """
        Compute the two symplectic eigenvalues v1 and v2 of the two-modes covariance
        matrix, and the symplectic eigenvalue v3 of Alice’s state given that Bob
        performed a heterodyne detection on his part of the state.

        Args:
            V (float): coefficient of the first diagonal block of the covariance matrix
            W (float): coefficient of the second diagonal block of the covariance matrix
            Z (float): coefficient of the antidiagonal blocks of the covariance matrix

        Returns:
            Tuple[float, float, float]: tuple containing v1, v2, v3. v1, v2: symplectic eigenvalues of the two-modes covariance matrix. v3: symplectic eigenvalue of Alice's state, condinitioned by Bob's measurement.
        """
        delt = delta(V, W, Z)
        gam = gamma(V, W, Z)

        v1 = sqrt((delt + sqrt(delt**2 - 4 * gam)) / 2)
        v2 = sqrt((delt - sqrt(delt**2 - 4 * gam)) / 2)
        v3 = V - Z**2 / (W + 1)

        return v1, v2, v3

    def holevo_bound(self, V: float, W: float, Z: float) -> float:
        """
        Computes the Holevo bound from the symplectic eigenvalues and the two mode
        covariance matrix.

        Args:
            V (float): coefficient of the first diagonal block of the covariance matrix
            W (float): coefficient of the second diagonal block of the covariance matrix
            Z (float): coefficient of the antidiagonal blocks of the covariance matrix

        Returns:
            float: Holevo bound of the mutual information between Eve and Bob.
        """
        v1, v2, v3 = self.sympl(V, W, Z)

        return g((v1 - 1) / 2) + g((v2 - 1) / 2) - g((v3 - 1) / 2)


class NoisyHeterodyneDetector(Detector):
    """
    Class containing the caractising datas of an Noisy Heterodyne Detector, which
    has a quantum efficiency eta, and an electric noise Vel. The functions
    that compute the sympleptic eigenvalues and the Holevo bound of the
    covariance matrix describded by V, W and Z. These three quantities depends
    of the simulation.
    """

    def __init__(self, eta: float, Vel: float):
        """
        Args:
            eta (float): efficiency of the detector.
            Vel (float): electronic noise of the detector.
        """
        self.eta = eta
        self.vel = Vel

    def sympl(self, V: float, W: float, Z: float) -> Tuple[float, float, float, float]:
        """
        Compute the two symplectic eigenvalues v1 and v2 of the two-modes covariance
        matrix, and the symplectic eigenvalues v3 and v4 of the covariance matrix of the
        state AFG|b exiting the beam splitter, after mixing the state AB with an EPR
        state FG of variance nu = 1 + 2*Vel/(1 - eta) ????

        Args:
            V (float): coefficient of the first diagonal block of the covariance matrix AB
            W (float): coefficient of the second diagonal block of the covariance matrix AB
            Z (float): coefficient of the antidiagonal blocks of the covariance matrix AB

        Returns:
            Tuple[float, float, float, float]: v1, v2, v3, v4. v1, v2: symplectic eigenvalues of the two-modes covariance matrix AB. v3, v4: two first symplectic eigenvalues of the two-modes covariance matrix AFG|b
        """
        eta = self.eta
        vel = self.vel

        delt = delta(V, W, Z)
        gam = gamma(V, W, Z)

        v1 = sqrt((delt + sqrt(delt**2 - 4 * gam)) / 2)
        v2 = sqrt((delt - sqrt(delt**2 - 4 * gam)) / 2)

        nu = 1 + 2 * vel / (1 - eta)
        nu = 1

        r1 = (
            (V * (1 + eta * W + nu - eta * nu) - eta * Z**2) ** 2
            + (eta * nu + W * (1 - eta + nu)) ** 2
            + (1 + nu + eta * (W * nu - 1)) ** 2
            - 2 * (1 - eta) * (nu + 1) ** 2 * Z**2
            - 2 * eta * (1 - eta) * (nu**2 - 1) * Z**2
            - 2 * eta * (nu**2 - 1) * (1 + W) ** 2
        ) / ((1 + W * eta + nu - eta * nu) ** 2)

        r1 -= 1
        r2 = (Z**2 - V * (W + eta) + (V * W - Z**2) * (-1 + eta) * nu) ** 2 / (
            1 + W * eta + nu - eta * nu
        ) ** 2

        v3 = sqrt(0.5 * (r1 + sqrt(r1**2 - 4 * r2)))
        v4 = sqrt(0.5 * (r1 - sqrt(r1**2 - 4 * r2)))
        return v1, v2, v3, v4

    def holevo_bound(self, V: float, W: float, Z: float) -> float:
        """
        Computes the Holevo bound from the symplectic eigenvalues and the
        covariance matrices of the bipartite system, before interfering with the
        EPR source, and after, under the condition of Bob's measurement.

        Args:
        V (float): coefficient of the first diagonal block of the covariance matrix
        W (float): coefficient of the second diagonal block of the covariance matrix
        Z (float): coefficient of the antidiagonal blocks of the covariance matrix

        Returns:
            float: Holevo bound of the mutual information between Eve and Bob.

        """

        v1, v2, v3, v4 = self.sympl(V, W, Z)

        return g((v1 - 1) / 2) + g((v2 - 1) / 2) - g((v3 - 1) / 2) - g((v4 - 1) / 2)
