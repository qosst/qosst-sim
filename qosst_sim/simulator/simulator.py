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
Abstract simulator class.
"""
import abc
from math import log2
from typing import Tuple

import numpy as np

from qosst_sim.modulation.modulation import Modulation
from qosst_sim.channel import GaussianChannel
from qosst_sim.detector import Detector


class Simulator(abc.ABC):
    """
    Abstract class meant to be sub-classed, containing the general functions that does
    not depend on the choice of the detector(ideal/noisy), nor on the type of simulation
    (asymptotic/finite size).
    """

    modulation: (
        Modulation  #: Modulation instance (Gaussian, QAM, Binomial QAM, Gaussian QAM).
    )
    detector: Detector  #: Instance of the detector (Perfect detector, noisy detector).
    beta: float  #: Efficiency of the reconciliation algorithm.
    c1: float  #: Quantity defined in Denys, A., Brown, P., & Leverrier, A. (2021).
    c2: float  #: Quantity defined in Denys, A., Brown, P., & Leverrier, A. (2021).
    n_B: float  # Quantity defined in Denys, A., Brown, P., & Leverrier, A. (2021).

    def __init__(
        self,
        modulation: Modulation,
        channel: GaussianChannel,
        detector: Detector,
        beta: float = 0.95,
    ):
        """
        Args:
            modulation (Modulation): class modulation object modulation chosen by Alice to sample the state that she wil send to Bob.
            channel (GaussianChannel): Instance of gaussian channel with given transmittance and excess noise.
            detector (Detector): detector instance used by Bob to measure the states.
            beta (float): reconciliation efficiency; a parameter that quantifies how much extra information Bob needs to send to Alice through the authenticated classical channel for her to correctly infer the value of Y, typically equal to 0.95 in practice.
        """
        self.modulation = modulation
        self.detector = detector
        self.channel = channel
        self.beta = beta

    def covariance(self) -> Tuple[float, float, float]:
        """
        Thanks to the Gaussian extremality properties of Gaussian states, it is
        known that this quantity can be upper bounded by its value computed for
        a Gaussian state with the same covariance matrix as the true state.
        Symmetry arguments (see : Anthony Leverrier. Composable security proof
        for continuous-variable quantum key distribution with coherent states.
        Phys. Rev. Lett., 114:070501, 2015. DOI: 10.1103/PhysRevLett.114.070501.)
        show that the covariance matrix of the entenglement-based version of the
        protocol can be safely replaced by a gaussian state matrix of the form :

            Gamma = [[ V * Id_2       Z * sigma_z ]
                     [ Z * sigma_z    W * Id_2    ]]


            when computing the secret key rate.

        The scalar parameters V := 1/2(<x_A**2> + <p_A**2>) = 1 + 2*<N_A>
                              W := 1/2(<x_B**2> + <p_B**2>) = 1 + 2*<N_B>

        can be estimated easily, and one need their upper bounds to compute the
        Holevo bound. As for  Z := 1/4(<x_A , x_B> + <p_A, p_B>) = <a*b + a_dag*b_dag>

        one needs a lower bound on this quantity, which is very difficult to
        precisely estimante. A bound is given by the Denys-Brown-Leverrier bound
        Z_star defined above.


        Returns:
            Tuple[float, float, float]: V, W, Z_star. V:  exact value af V known by Alice. W: estimate of W based on the estimation of nB. Z_star: Denys-Brown-Leverrier's bound on Z
        """
        v = self.modulation.va + 1
        w = 2 * self.n_B + 1
        racine = self.modulation.w * (self.n_B - 2 * self.c2**2 / (self.modulation.va))
        z_star = (2 * self.c1 - 2 * np.sqrt(racine)).real
        return v, w, z_star

    def skr(self) -> float:
        """
        Estimate of the Secret Key Rate of the protocol, based on the Holevo bound
        of the mutual information between Bob and Eve, computed from the bounds
        on the covariance matrix, and on the value of the matual information between
        Alice and Bob, computed form the Shot Noise Ratio of the simulation.

        Returns:
            float: estimate of the secret key rate.
        """
        v, w, z = self.covariance()
        holevo_bound = self.detector.holevo_bound(v, w, z)
        return self.beta * log2(1 + self.snr()) - holevo_bound

    @abc.abstractmethod
    def snr(self) -> float:
        """
        Returns:
            float: Shot Noise Ratio of the protocol, which is computed with the theoretical formula above, and which depends on detection.
        """
