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
Simulations with finite size effect.
"""
from collections import Counter

import numpy as np
from scipy import stats

from qosst_sim.simulator.simulator import Simulator
from qosst_sim.modulation.qam import QAM
from qosst_sim.channel import GaussianChannel
from qosst_sim.detector import Detector


class FiniteSizeSimulator(Simulator):
    """
    Class inheriting from simulator, describing the case where the numbner
    of symbols transmitted between Alice and Bob is supposed to be finite.
    No hypothesis needs to be made on the channel, nor on the type of detection,
    and the modulation only need to be discrete. One can have good
    oestimated values of the parameters c1, c2 and n_B (defined in Denys, A., Brown,
    P., & Leverrier, A. (2021). Explicit asymptotic secret key rate of
    continuous-variable quantum key distribution with an arbitrary modulation.
    Quantum, 5, 540.), in the asymptotical limit, which is when Alice and Bob
    exchange an infinite number of symbols.
    """

    alice_string: np.ndarray
    bob_string: np.ndarray
    betas: np.ndarray
    n_symbols: int
    modulation: QAM

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(
        self,
        modulation: QAM,
        channel: GaussianChannel,
        detector: Detector,
        n_symbols: int,
        beta: float = 0.95,
    ):
        """
        Args:
            modulation (QAM): class QAM object modulation chosen by Alice, which must be discrete (QAM).
            channel (GaussianChannel): class Channel object channel used.
            detector (Detector) : class Detector object detecor used by Bob, which can be ideal or noisy.
            n_symbols (int): number of symbols exchanged.
            beta (float):  reconciliation efficiency; a parameter that quantifies how much extra information Bob needs to send to Alice through the authenticated classical channel for her to correctly infer the value of Y, typically equal to 0.95 in practice.
        """
        super().__init__(modulation, channel, detector, beta)
        self.n_symbols = n_symbols

        # first, sample Alice's data according to the distribution
        # we must assume that we have already defined the constellation and the distribution
        custm = stats.rv_discrete(
            name="custm",
            values=(range(self.modulation.size**2), self.modulation.distribution),
        )

        # very important step: we sort the array, so that we can compute c1, c2 and n_B easily
        rvs = custm.rvs(size=self.n_symbols)
        raw_data = np.sort(rvs)

        # this random sample is then mapped to a random choice of symbols...
        self.alice_string = np.array(
            [self.modulation.constellation[x] for x in raw_data]
        )

        # then, we sample Bob's data accodring to the law of the chosen channel
        measured_quadratures = self.channel.sample_output(
            self.alice_string, self.detector
        )

        # Bob's string
        self.bob_string = np.sqrt(2 / self.detector.eta) * measured_quadratures  # ???

        raw_datas = Counter(raw_data)

        # compute the estimated values of the symbols beta_k detected by Bob, and of nB
        betas = np.zeros(self.modulation.size**2, dtype=np.complex64)
        n_B = 0.0  # test_1.2
        #       n_B = -1 # test_1.1 et test_1.3

        i = 0
        for x in raw_datas:  # i.e for x in range(self.m ** 2)
            freq = raw_datas[x]
            betas[x] = np.mean(self.bob_string[i : i + freq])
            n_B += self.modulation.distribution[x] * np.mean(
                np.abs(self.bob_string[i : i + freq]) ** 2
            )
            i += freq

        #        nB -= (1 + self.detector.Vel) / self.detector.eta - 1 # test_1.1
        n_B -= (1 + self.detector.vel) / self.detector.eta  # test_1.2

        self.betas = betas
        self.n_B = n_B

        c1 = 0
        c2 = 0
        a_tau = self.modulation.a_tau

        for weight, alpha, betak in zip(
            self.modulation.distribution, self.modulation.constellation, self.betas
        ):
            coherent_state = self.modulation.coherent_state(alpha)
            coherent_state_dag = coherent_state.T.conjugate()

            c1 += (
                weight
                * (coherent_state_dag @ a_tau @ coherent_state).conjugate()
                * betak
            )
            c2 += weight * alpha.conjugate() * betak

        self.c1 = c1.real
        self.c2 = c2.real

    def snr(self) -> float:
        """
        Returns:
            float: Shot Noise Ratio of the protocol, which is computed with the empirical estimator.
        """
        alice = self.alice_string
        bob = self.bob_string

        rho_hat = (
            np.dot(alice.real, bob.real) + np.dot(alice.imag, bob.imag)
        ) / np.sum(abs(alice) ** 2)
        qk_alice = alice.real
        pk_alice = alice.imag

        qk_bob = bob.real
        pk_bob = bob.imag
        return 1 / (
            sum(qk_bob**2 + pk_bob**2) / sum(rho_hat**2 * (qk_alice**2 + pk_alice**2))
            - 1
        )
