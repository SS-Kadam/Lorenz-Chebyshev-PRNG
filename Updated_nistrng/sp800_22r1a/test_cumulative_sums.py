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
#   - Fixed critical overflow: bits_copy now cast to int32 before accumulation
#     int8 overflows after ~128 steps; int32 safely handles 10M+ bit sequences
#   - Replaced slow Python for loop with numpy cumsum for forward/backward walks
#     This is both faster and overflow-safe when combined with int32 casting
#   - Added input normalization (flatten + where) consistent with other fixed tests
#   - Added minimum sequence length guard (n >= 100 per NIST recommendation)
# Original source: https://github.com/lucapasqualini/nistrng

# Import packages

import numpy
import math

# Import required src

from nistrng import Test, Result


class CumulativeSumsTest(Test):
    """
    Cumulative sums test as described in NIST paper:
    https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf

    The focus of this test is the maximal excursion (from zero) of the random walk
    defined by the cumulative sum of adjusted (-1, +1) digits in the sequence.

    The purpose of the test is to determine whether the cumulative sum of the partial
    sequences occurring in the tested sequence is too large or too small relative to
    the expected behavior of that cumulative sum for random sequences.

    Both forward (mode 0) and backward (mode 1) cumulative sums are tested.
    Both p-values must pass for the test to succeed.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Generate base Test class
        super(CumulativeSumsTest, self).__init__("Cumulative Sums", 0.01)

    def _execute(self, bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # --- STEP 1: NORMALIZATION ---
        # Flatten to 1D, ensure clean binary values, cast to int32
        # int32 is REQUIRED here — cumulative sum of 10M values of ±1
        # exceeds int8 range (max 127) almost immediately, causing overflow
        bits = numpy.asarray(bits).flatten()
        bits = numpy.where(bits > 0, 1, -1).astype(numpy.int32)  # Map 0→-1, 1→+1
        n = bits.size

        if n < 100:
            return Result(self.name, False, numpy.array([0.0, 0.0]))

        # --- STEP 2: FORWARD CUMULATIVE SUM ---
        # numpy.cumsum is vectorized and overflow-safe with int32
        forward_walk = numpy.cumsum(bits)
        forward_max = int(numpy.max(numpy.abs(forward_walk)))

        # --- STEP 3: BACKWARD CUMULATIVE SUM ---
        backward_walk = numpy.cumsum(bits[::-1])
        backward_max = int(numpy.max(numpy.abs(backward_walk)))

        # --- STEP 4: COMPUTE P-VALUES ---
        score_1: float = self._compute_p_value(n, forward_max)
        score_2: float = self._compute_p_value(n, backward_max)

        # Both forward and backward must pass
        success = score_1 >= self.significance_value and score_2 >= self.significance_value
        return Result(self.name, success, numpy.array([score_1, score_2]))

    def is_eligible(self, bits: numpy.ndarray) -> bool:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        return bits.size >= 100

    @staticmethod
    def _compute_p_value(sequence_size: int, max_excursion: int) -> float:
        """
        Compute P-Value given the sequence size and the max excursion.
        Formula per NIST SP 800-22 Section 2.13.

        :param sequence_size: the length of the sequence of bits (n)
        :param max_excursion: the max excursion forward or backward (z)
        :return: the computed float P-Value
        """
        n = sequence_size
        z = max_excursion

        # Guard against zero excursion (theoretically impossible for n > 0)
        if z == 0:
            return 1.0

        # --- First sum ---
        # k range: floor((-n/z + 1) / 4) to floor((n/z - 1) / 4)
        sum_a: float = 0.0
        start_k: int = int(math.floor((((-n / z) + 1.0) / 4.0)))
        end_k:   int = int(math.floor((((n / z) - 1.0) / 4.0)))
        for k in range(start_k, end_k + 1):
            c = 0.5 * math.erfc(-((4.0 * k + 1.0) * z) / math.sqrt(n) * math.sqrt(0.5))
            d = 0.5 * math.erfc(-((4.0 * k - 1.0) * z) / math.sqrt(n) * math.sqrt(0.5))
            sum_a += c - d

        # --- Second sum ---
        # k range: floor((-n/z - 3) / 4) to floor((n/z - 1) / 4)
        sum_b: float = 0.0
        start_k = int(math.floor((((-n / z) - 3.0) / 4.0)))
        end_k   = int(math.floor((((n / z) - 1.0) / 4.0)))
        for k in range(start_k, end_k + 1):
            c = 0.5 * math.erfc(-((4.0 * k + 3.0) * z) / math.sqrt(n) * math.sqrt(0.5))
            d = 0.5 * math.erfc(-((4.0 * k + 1.0) * z) / math.sqrt(n) * math.sqrt(0.5))
            sum_b += c - d

        return 1.0 - sum_a + sum_b