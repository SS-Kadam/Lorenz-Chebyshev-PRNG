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

## --- MODIFICATIONS ---
# Modified by [SS Kadam], [2026]
# Changes made for compatibility with modern NumPy (1.20+) and Python 3.10+:
#   - Added input normalization (flatten + where) consistent with other fixed tests

# Import packages

import numpy
import scipy.special
from nistrng import Test, Result

class FrequencyWithinBlockTest(Test):
    """
    Frequency within block test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of the test is the proportion of ones within M-bit blocks. The purpose of this test is to determine whether the frequency of
    ones in an M-bit block is approximately M/2, as would be expected under an assumption of randomness.
    For block size M=1, this test degenerates to the Frequency (Monobit) test.

    """

    def __init__(self, block_size=128):
        # NIST recommends M >= 20 and n >= 100. M=128 is a standard for crypto.
        self._block_size_fixed = block_size
        super(FrequencyWithinBlockTest, self).__init__("Frequency Within Block", 0.01)

    def _execute(self, bits: numpy.ndarray) -> Result:
        # --- DEFENSIVE STEP 1: NORMALIZATION ---
        bits = numpy.asarray(bits).flatten()
        bits = numpy.where(bits > 0, 1, 0)
        n = bits.size

        # --- STEP 2: BLOCK PARAMETERS ---
        M = self._block_size_fixed
        N = n // M  # Number of blocks

        if N == 0:
            return Result(self.name, False, numpy.array(0.0))

        # --- STEP 3: VECTORIZED PROCESSING ---
        # Instead of a slow 'for' loop, we reshape and sum.
        # This handles millions of bits in milliseconds.
        usable_bits = n - (n % M)
        blocks = bits[:usable_bits].reshape((N, M))

        # Calculate proportion of ones in each block
        pi_i = numpy.sum(blocks, axis=1) / float(M)

        # --- STEP 4: CHI-SQUARE CALCULATION ---
        chi_square = 4.0 * M * numpy.sum((pi_i - 0.5) ** 2)

        # --- STEP 5: P-VALUE (Score) ---
        # Using the upper incomplete gamma function as per NIST SP 800-22
        score = scipy.special.gammaincc(N / 2.0, chi_square / 2.0)

        success = score >= self.significance_value
        return Result(self.name, success, numpy.array(score))

    def is_eligible(self, bits: numpy.ndarray) -> bool:
        return bits.size >= 100