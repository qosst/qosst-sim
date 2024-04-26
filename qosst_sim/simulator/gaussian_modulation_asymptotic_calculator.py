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
Class for simulating the case of a Gaussian modulation and a Gaussian channel.
"""

from math import sqrt

from qosst_sim.simulator.simulator import Simulator
from qosst_sim.modulation.modulation import GaussianModulation
from qosst_sim.channel import GaussianChannel
from qosst_sim.detector import Detector


class GaussianModulationAsymptoticCalculator(Simulator):
    """
    Class inheriting from simulator, describing the particular case where
    the channel and the modulation are supposed to be gaussian. In this very
    special case, one can have theorical values of the parameters c1, c2
    and nB (defined in Denys, A., Brown, P., & Leverrier, A. (2021).
    Explicit asymptotic secret key rate of continuous-variable quantum key
    distribution with an arbitrary modulation. Quantum, 5, 540.), in the
    asymptotical limit, which is when Alice and Bob exchange an infinite
    number of symbols.
    """

    def __init__(
        self,
        modulation: GaussianModulation,
        channel: GaussianChannel,
        detector: Detector,
        beta: float = 0.95,
    ):
        """
        Args:
            modulation (GaussianModulation): class GaussianModulation object modulation chosen by Alice, which must be gaussian.
            channel (GaussianChannel): class GaussianChannel object channel used, which must be gaussian.
            detector (Detector): class Detector object detecor used by Bob, which can be ideal or noisy.
            beta (float): reconciliation efficiency; a parameter that quantifies how much extra information Bob needs to send to Alice through the authenticated classical channel for her to correctly infer the value of Y, typically equal to 0.95 in practice.
        """
        super().__init__(modulation, channel, detector, beta)

        t = self.channel.t
        va = self.modulation.va
        self.n_B = t * (va + self.channel.xi) / 2
        self.c1 = sqrt(t * va / 2 * (va / 2 + 1))
        self.c2 = sqrt(t) * va / 2

    def snr(self) -> float:
        """
        Returns:
            float: Shot Noise Ratio of the protocol, which is computed with the theoretical formula above, and which depends on detection.
        """
        t = self.channel.t
        eta = self.detector.eta
        return (
            t
            * self.modulation.va
            * eta
            / (2 + 2 * self.detector.vel + eta * t * self.channel.xi)
        )
