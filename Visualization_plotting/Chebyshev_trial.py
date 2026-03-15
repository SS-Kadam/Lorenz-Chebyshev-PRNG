# =============================================================================
# Copyright (C) 2026 SS Kadam
# All rights reserved.
#
# Visualization code for the chaotic map systems used as the mathematical
# foundation of the PRNG developed by SS Kadam (2026).
#
# Included for transparency and academic reproducibility purposes.
# Not part of the core PRNG implementation.
#
# Development environment : Local (private development machine)
# Development year        : 2026
#
# Licensed under the MIT License.
# See LICENSE file in the project root for full license text.
# =============================================================================


"""
True Chebyshev polynomial PRNG core (trial / analysis build)
------------------------------------------------------------
- Password -> Argon2id (mandatory)
- 256-bit derivation -> 128-bit x0 seed
- k chosen uniformly at random in [25, 50]
- True Chebyshev map: x_{n+1} = cos(k * arccos(x_n))
- Burn-in to remove transients
- Raw state output (NO bit extraction)
- Visualization: time series + phase plot
- Lyapunov exponent: theoretical + numeric estimate

NOTE:
This code is for *analysis and documentation*, not a final PRNG.
Floating-point is used intentionally to expose geometric structure.
"""

import math
import random
import binascii
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from argon2.low_level import hash_secret_raw, Type

# ------------------------------------------------------------
# USER PARAMETERS
# ------------------------------------------------------------
password = "correct horse battery staple"   # change as needed
salt = b"chebyshev-salt-v2"
burn = 10_000
plot_steps = 2_000

# ------------------------------------------------------------
# ARGON2id KEY DERIVATION (MANDATORY)
# ------------------------------------------------------------
pwd_bytes = password.encode("utf-8")

derived = hash_secret_raw(
    secret=pwd_bytes,
    salt=salt,
    time_cost=2,
    memory_cost=2**16,
    parallelism=1,
    hash_len=32,
    type=Type.ID,
)

print("Derivation summary:")
print(" - Derived bytes (hex):", binascii.hexlify(derived).decode())

# ------------------------------------------------------------
# SEED EXTRACTION
# ------------------------------------------------------------
seed_left = int.from_bytes(derived[:16], "big")
two128 = 1 << 128

# Map seed to x0 in (-1, 1), avoiding boundaries
x0 = (seed_left / two128) * 2.0 - 1.0
x0 = max(min(x0, 0.999999999999), -0.999999999999)

# ------------------------------------------------------------
# RANDOM k SELECTION
# ------------------------------------------------------------
random.seed(int.from_bytes(derived[16:], "big"))
k = random.randint(25, 50)

print(" - k selected:", k)
print(" - Initial x0:", x0)

# ------------------------------------------------------------
# TRUE CHEBYSHEV MAP
# ------------------------------------------------------------
def chebyshev_map(x, k):
    return math.cos(k * math.acos(x))

# ------------------------------------------------------------
# BURN-IN
# ------------------------------------------------------------
x = x0
for _ in range(burn):
    x = chebyshev_map(x, k)

print("Burn complete.")

# ------------------------------------------------------------
# COLLECT RAW STATES FOR ANALYSIS
# ------------------------------------------------------------
x_series = []
x_curr = x
for _ in range(plot_steps):
    x_curr = chebyshev_map(x_curr, k)
    x_series.append(x_curr)

# Save raw values
out_path = Path("/mnt/data/chebyshev_raw_states.txt")
out_path.write_text("\n".join(f"{v:.17e}" for v in x_series))
print("Saved raw states to", out_path)

# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------
plt.figure(figsize=(10, 3))
plt.plot(x_series[:1000], lw=0.8)
plt.title("True Chebyshev map time series (x_n)")
plt.xlabel("n")
plt.ylabel("x_n")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(x_series[:-1], x_series[1:], s=2)
plt.title("Phase plot: (x_n, x_{n+1})")
plt.xlabel("x_n")
plt.ylabel("x_{n+1}")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# LYAPUNOV EXPONENT
# ------------------------------------------------------------
# Theoretical: ln(k)
lyap_theoretical = math.log(k)

# Numeric estimate via nearby trajectory
def numeric_lyapunov(x0, k, delta0=1e-12, steps=2000):
    x = x0
    x_pert = x0 + delta0
    s = 0.0
    count = 0

    for _ in range(steps):
        x = chebyshev_map(x, k)
        x_pert = chebyshev_map(x_pert, k)
        d = abs(x - x_pert)
        if d == 0.0:
            break
        s += math.log(abs(d / delta0))
        count += 1
        x_pert = x + (x_pert - x) / abs(x_pert - x) * delta0

    return s / count if count else float("nan")

lyap_numeric = numeric_lyapunov(x, k)


# ------------------------------------------------------------
# KILLER PLOT: lagged return map
# ------------------------------------------------------------
tau = 20  # try 10, 20, 50

x_n = np.array(x_series[:-tau])
x_tau = np.array(x_series[tau:])

plt.figure(figsize=(5,5))
plt.scatter(x_n, x_tau, s=2)
plt.title(f"Lagged return map: (x_n, x_(n+{tau}))")
plt.xlabel("x_n")
plt.ylabel(f"x_(n+{tau})")
plt.tight_layout()
plt.show()


print("\nLyapunov exponents:")
print(" - theoretical ln(k) =", lyap_theoretical)
print(" - numeric estimate  =", lyap_numeric)
