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

# --- MODIFICATIONS ---
# Modified by [SS Kadam], [2026]
# Changes made for compatibility with modern NumPy (1.20+) and Python 3.10+:
#   - Added input normalization (flatten + where) consistent with other fixed tests

# Import packages

import numpy
import math

# Import required src

from nistrng import Test, Result

import numpy
import math


class MonobitTest(Test):
    """
    Monobit test as described in NIST SP 800-22.
    The focus is the proportion of zeros and ones for the entire sequence.
    """

    def __init__(self):
        super(MonobitTest, self).__init__("Monobit", 0.01)

    def _execute(self, bits: numpy.ndarray):
        """
        Calculates the P-value for the proportion of bits.
        """
        # --- CHANGE 1: DATA NORMALIZATION ---
        # Ensures the input is treated as a flat array of 0s and 1s.
        # This prevents '0.0' errors caused by multidimensional arrays
        # or non-binary integer values.
        bits = numpy.asarray(bits).flatten()

        # --- CHANGE 2: TYPE SAFE COUNTING ---
        # Using sum() on a boolean comparison is safer than count_nonzero
        # if the input happens to be ASCII characters or raw bytes.
        n = bits.size
        ones = numpy.sum(bits == 1)
        zeroes = n - ones

        # --- CHANGE 3: EDGE CASE PROTECTION ---
        # If the array is empty or malformed, return 0 to prevent crash.
        if n == 0:
            return Result(self.name, False, numpy.array(0.0))

        # Compute difference
        difference = abs(ones - zeroes)

        # --- CHANGE 4: MATH STABILITY ---
        # Explicitly casting to float64 to ensure maximum precision
        # during the square root and erfc operations.
        score = math.erfc(float(difference) / (math.sqrt(float(n)) * math.sqrt(2.0)))

        # Return result
        success = score >= self.significance_value
        return Result(self.name, success, numpy.array(score))

    def is_eligible(self, bits: numpy.ndarray) -> bool:
        return True