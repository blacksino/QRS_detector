import neurokit2 as nk
from matplotlib import pyplot as plt
import numpy as np

simulated_signal = nk.ecg_simulate(length=1000,sampling_rate=250, noise=1, heart_rate=80,heart_rate_std=10)
plt.plot(simulated_signal)
plt.show()

_,r_peak = nk.ecg_peaks()
ecg = nk.ecg_clean()