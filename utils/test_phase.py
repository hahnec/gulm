import matplotlib.pyplot as plt
import numpy as np

from multimodal_emg import emg_wave_model


t = np.arange(-1.6383999999999998e-05/2, 1.6383999999999998e-05/2, 1.2800000000000001e-09/2)

fs = 15625000.0

res0 = emg_wave_model(alpha=1, mu=0, sigma=2*1e-7, eta=0, fkhz=fs/1e3, phi=0, x=t)
res1 = emg_wave_model(alpha=1, mu=0, sigma=2*1e-7, eta=0, fkhz=fs/1e3, phi=-np.pi*1/2, x=t)
res2 = emg_wave_model(alpha=1, mu=0, sigma=2*1e-7, eta=0, fkhz=fs/1e3, phi=+np.pi*1/2, x=t)
res3 = emg_wave_model(alpha=1, mu=0, sigma=2*1e-7, eta=0, fkhz=fs/1e3, phi=+np.pi*3/4, x=t)
res4 = emg_wave_model(alpha=1, mu=0, sigma=2*1e-7, eta=0, fkhz=fs/1e3, phi=-np.pi*1/4, x=t)

plt.figure()

plt.plot(t, res0, label='0')
plt.plot(t, res1, label='-1/2')
plt.plot(t, res2, label='+1/2')
plt.plot(t, res3, label='+3/4')
plt.plot(t, res4, label='-1/4')
plt.plot([0, 0], [-1, +1], 'k')
plt.legend()

plt.show()