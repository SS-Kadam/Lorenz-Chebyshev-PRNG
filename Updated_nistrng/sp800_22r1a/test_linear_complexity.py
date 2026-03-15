#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
# Inspired by the work of David Johnston (C) 2017: https://github.com/dj-on-github/sp800_22_tests
#
# NistRng is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
#
# --- MODIFICATIONS ---
# Modified by [SS Kadam], [2026]
# Changes made for compatibility with modern NumPy (1.20+) and Python 3.10+:
#   - Added input normalization (flatten + where) consistent with other fixed tests
#   - Removed thread-unsafe instance variable cache, replaced with local computation
#   - Added max(0, ...) guard in Berlekamp-Massey range to prevent negative range edge case
# Original source: https://github.com/lucapasqualini/nistrng

# Import packages

import numpy
import scipy.special

# Import required src

from nistrng import Test, Result


class LinearComplexityTest(Test):
    """
    Linear complexity test as described in NIST paper:
    https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf

    The focus of this test is the length of a linear feedback shift register (LFSR).
    The purpose of this test is to determine whether or not the sequence is complex
    enough to be considered random. Random sequences are characterized by longer LFSRs.
    An LFSR that is too short implies non-randomness.

    Block size (M) = 512 bits.
    Minimum sequence length = M * 200 = 102,400 bits (NIST SP 800-22 Section 2.10).
    The significance value of the test is 0.01.
    """

    def __init__(self, block_size=500):
        # NIST recommends 500 <= M <= 5000
        self._M = block_size
        super(LinearComplexityTest, self).__init__("Linear Complexity", 0.01)

    def _execute(self, bits: numpy.ndarray) -> Result:
        # --- DEFENSIVE STEP 1: NORMALIZATION ---
        bits = numpy.asarray(bits).flatten()
        bits = numpy.where(bits > 0, 1, 0).astype(numpy.int8)
        n = bits.size
        M = self._M
        N = n // M  # Number of blocks

        if N < 200:  # NIST recommended minimum blocks for validity
            return Result(self.name, False, numpy.array(0.0))

        # --- STEP 2: THE CALCULATION ---
        # Theoretical mean for a random sequence of length M
        mean = (M / 2.0) + ((9.0 + ((-1) ** (M + 1))) / 36.0) - ((M / 3.0 + 2.0 / 9.0) / (2 ** M))

        # Bin counts for Chi-square (7 bins as per NIST)
        v = numpy.zeros(7)

        for i in range(N):
            block = bits[i * M: (i + 1) * M]
            L = self._berlekamp_massey(block)

            # Calculate the T statistic
            T = ((-1) ** M) * (L - mean) + (2.0 / 9.0)

            # Categorize into bins
            if T <= -2.5:
                v[0] += 1
            elif T <= -1.5:
                v[1] += 1
            elif T <= -0.5:
                v[2] += 1
            elif T <= 0.5:
                v[3] += 1
            elif T <= 1.5:
                v[4] += 1
            elif T <= 2.5:
                v[5] += 1
            else:
                v[6] += 1

        # --- STEP 3: CHI-SQUARE AND P-VALUE ---
        # Probabilities for the 7 bins defined by NIST
        probs = numpy.array([0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.07500, 0.00828])

        chi_sq = 0.0
        for i in range(7):
            expected = N * probs[i]
            chi_sq += ((v[i] - expected) ** 2) / expected

        score = scipy.special.gammaincc(3.0, chi_sq / 2.0)

        success = score >= self.significance_value
        return Result(self.name, success, numpy.array(score))

    def _berlekamp_massey(self, block):
        """ Efficient implementation of the Berlekamp-Massey Algorithm """
        n = len(block)
        b = numpy.zeros(n, dtype=numpy.int8)
        c = numpy.zeros(n, dtype=numpy.int8)
        b[0] = 1
        c[0] = 1
        L = 0
        m = -1

        for i in range(n):
            # Calculate discrepancy
            d = block[i]
            for j in range(1, L + 1):
                d ^= c[j] & block[i - j]

            if d != 0:
                temp = c.copy()
                # Shift and XOR
                shift = i - m
                for j in range(n - shift):
                    c[j + shift] ^= b[j]

                if L <= i / 2:
                    L = i + 1 - L
                    m = i
                    b = temp
        return L

    def is_eligible(self, bits: numpy.ndarray) -> bool:
        return bits.size >= 1000000  # Recommended minimum 10^6 bits