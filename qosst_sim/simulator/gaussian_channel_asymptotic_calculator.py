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
Simulator for the asymptotic case with a Guassian channel.
"""

from math import sqrt

from qosst_sim.simulator.simulator import Simulator
from qosst_sim.modulation.qam import QAM
from qosst_sim.channel import GaussianChannel
from qosst_sim.detector import Detector


class GaussianChannelAsymptoticCalculator(Simulator):
    """
    Class inheriting from simulator, describing the particular case where
    the channel is supposed to be gaussian. In this special case, one can have
    theorical values of the parameters c1, c2 and nB (defined in Denys, A., Brown,
    P., & Leverrier, A. (2021). Explicit asymptotic secret key rate of
    continuous-variable quantum key distribution with an arbitrary modulation.
    Quantum, 5, 540.), in the asymptotical limit, which is when Alice and Bob
    exchange an infinite number of symbols.
    """

    modulation: QAM

    def __init__(
        self,
        modulation: QAM,
        channel: GaussianChannel,
        detector: Detector,
        beta: float = 0.95,
    ):
        """
        Args:
            modulation (QAM): class QAM object modulation chosen by Alice, which must be discrete (QAM).
            channel (GaussianChannel): class GaussianChannel object channel used, which must be gaussian
            detector (Detector): class Detector object detecor used by Bob, which can be ideal or noisy.
            beta (float): reconciliation efficiency; a parameter that quantifies how much extra information Bob needs to send to Alice through the authenticated classical channel for her to correctly infer the value of Y, typically equal to 0.95 in practice.
        """
        super().__init__(modulation, channel, detector, beta)

        tau_half = self.modulation.tau_half
        a = self.modulation.a
        a_dag = a.T  # transpose
        t = self.channel.t  # transmission

        self.n_B = t * (self.modulation.va + self.channel.xi) / 2
        self.c1 = sqrt(t) * (tau_half @ a @ tau_half @ a_dag).trace().real
        self.c2 = sqrt(t) * self.modulation.va / 2

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
