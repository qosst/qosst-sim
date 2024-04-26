# Tutorial and example

## Tutorial

The simulation package allows you to compute the secret key rate of a CV-QKD system by giving the parameters of emission, of the channel and of detection.

The parameters of emission are given by choosing a modulation (see {doc}`here <../api/modulation>` for a list of possible modulations) along with the parameters, including the variance for instance, but also the dimension of discrete modulations when applicable.

The parameters of the channel are given using a channel object, and currently only the {py:class}`qosst_sim.channel.GaussianChannel` class is available, and takes as the transmittance and the excess noise as parameters.

Finally the parameters of detection are given using a Detector object (see {doc}`here <../api/detector>` for a list of available detectors). For the noisy detector, this include the efficiency and the electronic noise.

Those 3 objects can be fed into one of the simulators, along with some additional parameters. The list of simulators is available {doc}`here <../api/simulator>`.

```{note}

One might ask the difference between sim and skr packages. The SKR packages is here to calculate the key rate from the parameters obtained from a real experiment, whereas the sim package will actually generate symbols, and make them follow a simple noise level to then estimate the values of the channel and compute the key rate.
```

## Example

Here below is an example to simulate the key rate for 64-PCSQAM, in asymptotic and finite size scenarios.

```{eval-rst}
.. plot:: pyplots/example.py
    :include-source: true
    :align: center
```