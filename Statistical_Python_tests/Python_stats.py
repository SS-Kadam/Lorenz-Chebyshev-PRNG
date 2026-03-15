# =============================================================================
# Copyright (C) 2026 SS Kadam
# All rights reserved.
#
# This file is part of the local statistical test suite developed independently
# by SS Kadam to supplement the NIST SP 800-22 test battery.
#
# Tests included in this suite were designed, implemented, and validated
# locally as original work.
#
# Development environment : Local (private development machine)
# Development year        : 2026
#
# Licensed under the MIT License.
# See LICENSE file in the project root for full license text.
# =============================================================================


import numpy as np
from scipy.stats import chisquare, chi2

# ==========================================================
# INPUT FILE
# ==========================================================

FILE = "file-name"

with open(FILE, "rb") as f:
    data = f.read()

bytes_arr = np.frombuffer(data, dtype=np.uint8)
bits = np.unpackbits(bytes_arr)

n_bits = len(bits)
n_bytes = len(bytes_arr)

print("\nLoaded file:", FILE)
print("Total bits :", n_bits)
print("Total bytes:", n_bytes)


# ==========================================================
# 1. SERIAL CORRELATION
# ==========================================================

def serial_correlation(bits):
    x = bits[:-1]
    y = bits[1:]
    return np.corrcoef(x, y)[0,1]

sc = serial_correlation(bits)


# ==========================================================
# 2. SHANNON ENTROPY (per bit)
# ==========================================================

def shannon_entropy(bits):
    p0 = np.mean(bits == 0)
    p1 = 1 - p0

    H = 0
    if p0 > 0:
        H -= p0 * np.log2(p0)
    if p1 > 0:
        H -= p1 * np.log2(p1)

    return H

entropy = shannon_entropy(bits)


# ==========================================================
# 3. TRUE CHI-SQUARE TEST (256 BYTE VALUES)
# ==========================================================


def chi_square_bits(bits):
    n = len(bits)

    zeros = np.sum(bits == 0)
    ones = np.sum(bits == 1)

    observed = [zeros, ones]
    expected = [n / 2, n / 2]

    chi_stat, p_val = chisquare(f_obs=observed, f_exp=expected)

    return chi_stat, p_val

chi, p = chi_square_bits(bits)

# ==========================================================
# 4. FAST LEMPEL-ZIV COMPLEXITY
# ==========================================================

def lz_complexity_fast(bits):

    dictionary = set()
    w = ""
    complexity = 0

    for bit in bits:
        wc = w + str(bit)
        if wc not in dictionary:
            dictionary.add(wc)
            complexity += 1
            w = ""
        else:
            w = wc

    if w:
        complexity += 1

    return complexity

lz = lz_complexity_fast(bits)

n = len(bits)

lz_norm = lz * np.log2(n) / n

# ==========================================================
# 5. POKER TEST
# ==========================================================

def poker_test(bits, m=4):

    n_blocks = len(bits) // m
    blocks = bits[:n_blocks*m].reshape((n_blocks, m))

    values = blocks.dot(1 << np.arange(m)[::-1])

    freq = np.bincount(values, minlength=2**m)

    k = 2**m

    X = (k / n_blocks) * np.sum(freq**2) - n_blocks

    p_val = 1 - chi2.cdf(X, k - 1)

    return X, p_val

poker_stat, poker_p = poker_test(bits)


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n================ PRNG TEST RESULTS ================\n")

print("[1] Serial Correlation")
print("Value:", sc)

print("\n[2] Shannon Entropy")
print("Entropy (bits):", entropy)

print("\n[Chi-Square Test (Bit level)]")
print("Chi-square:", chi)
print("p-value:", p)

print("\n[Lempel–Ziv Complexity]")
print("LZ:", lz)
print("Normalized LZ:", lz_norm)


print("\n[5] Poker Test")
print("Statistic:", poker_stat)
print("p-value :", poker_p)

print("\n===================================================")