# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt

# # 1.a

# +
N = 128
f1 = 1  # in Hz
f2 = N + f1

t = np.linspace(0, 1, N, endpoint=False)
x1_t = np.cos(2 * np.pi * t * f1)
x2_t = np.cos(2 * np.pi * t * f2)

fig, (ax1, ax2) = plt.subplots(
    1, 2, constrained_layout=True, sharey=True, figsize=(10, 5)
)
fig.suptitle("N={0} Sample points".format(N), fontsize=16)
ax1.set_ylabel("Signal (V)")

ax1.set_title("f1={0}Hz".format(f1))
ax1.set_xlabel("Time (s)")
ax1.plot(t, x1_t, ".-")

ax2.set_title("f2={0}Hz".format(f2))
ax2.set_xlabel("Time (s)")
ax2.plot(t, x2_t, ".-");
# -

# # 1.c

# +
# we only work with the second data from now on
f = f2
x_t = x2_t


def periodogram(x, t):
    delta_t = t[1] - t[0]
    frequencies = np.fft.rfftfreq(
        len(t), delta_t
    )  # frequencies for each element of the periodogram
    X_tilde = np.fft.rfft(x)
    S_per = 2 / len(t) * np.power(np.abs(X_tilde), 2)
    return (frequencies, S_per)


def plot_S(fs, S_f, title="$S^{per}_N(f_i)$"):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle(title, fontsize=16)
    ax.set_ylabel("(W/Hz)")  # is this correct?
    ax.set_xlabel("f (Hz)")
    ax.plot(fs, S_f, ".")
    ax.set_ylim(bottom=0)


# -

fs, S_f = periodogram(x_t, t)
plot_S(fs, S_f)

# # 1.d

# +
x_noise_t = np.random.normal(loc=0, scale=1, size=len(t))

fs, S = periodogram(x_noise_t, t)
plot_S(fs, S, title="$^{noise}S^{per}_N(f_i)$")

print("→ doesn't seem to be constant (as one would expect from white noise)")
# -

# # 1.e

# +
stretch_factor = 100
t_long = np.linspace(0, 1 * stretch_factor, N * stretch_factor, endpoint=False)
x_longnoise_t = np.random.normal(loc=0, scale=1, size=len(t_long))

fs, S_long = periodogram(x_longnoise_t, t_long)
plot_S(fs, S_long, title="$^{long noise}S^{per}_N(f_i)$")

print("→ still not constant")
# -

print("variance of noise:     ", np.var(S))
print("variance of long noise:", np.var(S_long))
print(" ⇒ both are similar and ≈4*𝜎(x)=4")  # I excpected a decrease

# # 1.f

# +
sigmas = np.arange(0, 10, 1)
vars = []
for sigma in sigmas:
    x_noise_t = np.random.normal(loc=0, scale=sigma, size=len(t_long))
    fs, S = periodogram(x_noise_t, t_long)
    vars.append(np.var(S))
vars = np.array(vars)

fig, (ax1, ax2) = plt.subplots(
    2, 1, constrained_layout=True, sharex=True, figsize=(7, 7)
)
fig.suptitle("variance of periodogram (v)", fontsize=16)
ax1.set_ylabel("v")
s = np.linspace(sigmas[0], sigmas[-1])
ax1.plot(s, 4 * np.power(s, 4), label="$4𝜎^4$")
ax1.plot(sigmas, vars, ".", label="variance")
ax1.legend()

ax2.set_xlabel("std. deviation of signal")
ax2.set_ylabel("fit plot: $(v/4)^{1/4}$")
s = np.linspace(sigmas[0], sigmas[-1])
ax2.plot(s, s, label="$𝜎$")
ax2.plot(sigmas, np.power(vars / 4, 1 / 4), ".", label="$(v/4)^{1/4}$")
ax2.legend();
# -

# # 1.g

# +
S_sum = np.zeros(len(t) // 2 + 1)
n = 10000
for i in range(n):
    x_noise_t = np.random.normal(loc=0, scale=1, size=len(t))
    fs, S = periodogram(x_noise_t, t)
    S_sum += S
S_avg = S_sum / n

plot_S(fs, S_avg, title="$^{avg}S^{per}_N(f_i)$")

print("variance of avg of S_avg:", np.var(S_avg))
# -


