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
from nistrng import Test, Result

class RunsTest(Test):
    """
    Runs test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of this test is the total number of runs in the sequence, where a run is an uninterrupted sequence of identical bits.
    A run of length k consists of exactly k identical bits and is bounded before and after with a bit of the opposite value.
    The purpose of the runs test is to determine whether the number of runs of ones and zeros of various lengths is as expected
    for a random sequence. In particular, this test determines whether the oscillation between such zeros and ones is too fast or too slow.


    """

    def __init__(self):
        super(RunsTest, self).__init__("Runs", 0.01)

    def _execute(self, bits: numpy.ndarray):
        # --- DEFENSIVE STEP 1: NORMALIZATION ---
        # Same as Monobit: ensure we have a flat array of actual 0s and 1s
        bits = numpy.asarray(bits).flatten()
        bits = numpy.where(bits > 0, 1, 0)
        n = bits.size

        if n < 2:
            return Result(self.name, False, numpy.array(0.0))

        # --- STEP 2: CALCULATE PROPORTION (pi) ---
        ones = numpy.sum(bits == 1)
        pi = float(ones) / float(n)

        # --- STEP 3: PRE-TEST ELIGIBILITY ---
        # NIST Requirement: If the proportion is too far from 0.5,
        # the Runs test is not even performed and results in a failure.
        tau = 2.0 / math.sqrt(float(n))
        if abs(pi - 0.5) >= tau:
            return Result(self.name, False, numpy.array(0.0))

        # --- STEP 4: COUNT OBSERVED RUNS ---
        # A run is a sequence of identical bits.
        # We count transitions and add 1.
        v_n_obs = numpy.sum(bits[:-1] != bits[1:]) + 1

        # --- STEP 5: CALCULATE P-VALUE ---
        # We add a tiny epsilon (1e-15) to the denominator to prevent
        # ZeroDivisionError, though Step 3 should catch most cases.
        denominator = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)

        if denominator == 0:
            return Result(self.name, False, numpy.array(0.0))

        numerator = abs(float(v_n_obs) - (2.0 * n * pi * (1.0 - pi)))
        score = math.erfc(numerator / denominator)

        success = score >= self.significance_value
        return Result(self.name, success, numpy.array(score))

    def is_eligible(self, bits: numpy.ndarray) -> bool:
        return True