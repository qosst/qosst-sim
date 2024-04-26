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
Module containg the class of channels.
"""

import numpy as np
from qosst_sim.detector import Detector


# pylint: disable=too-few-public-methods
class GaussianChannel:
    """
    Class containing the caractising datas of a Gaussian Channel, and the sampling
    function that draws its the output of the channel, depending also on the choice
    of detector used.
    """

    t: float  #: Transmittance of the channel.
    xi: float  #: Excess noise of the channel.

    def __init__(self, t: float, xi: float):
        """
        Args:
            T (float): transmission of the channel.
            xi (float): additive noise of the channel.
        """
        self.t = t
        self.xi = xi

    def sample_output(self, symbols: np.ndarray, detector: Detector) -> np.ndarray:
        """
        Pseudo-random sampler of the output of a Gaussian Channel. If the symbols at the entrance
        are x_k, the output is :

            y_k = sqrt(eta*T/2) * x_k + w

        where w is sampled according to a symetric complex gaussian of mean 0 and variance
        1 + Vel + eta*T/2, which is equivalent to say that it real and imaginary parts are
        independantly sampled according toa real gaussian distribution of mean 0 and variance
        1 + Vel + eta*T/2.

        Args:
            symbols (np.ndarray): array of the N symboles sampled by Alice according to her modulation.
            detector (Detector): container of the values of eta and Vel.

        Returns:
            np.ndarray: corresponding symboles that Bob receives at the exit of the channel.
        """

        num_symbols = len(symbols)
        eta = detector.eta
        vel = detector.vel
        noise_r = np.random.normal(
            0, 0.5 * np.sqrt(1 + vel + eta * self.t * self.xi / 2), num_symbols
        )
        noise_i = np.random.normal(
            0, 0.5 * np.sqrt(1 + vel + eta * self.t * self.xi / 2), num_symbols
        )

        return np.sqrt(self.t * eta / 2) * symbols + noise_r + 1j * noise_i
